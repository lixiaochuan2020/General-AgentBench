"""LiteLLM API - Supports multiple LLM providers (including AWS Bedrock)

API Rate Limit retry strategy is consistent with the original benchmark:
- OpenHands (SWE-Bench): num_retries=5, retry_min_wait=8s, retry_max_wait=64s
- Terminal-Bench: stop_after_attempt=3, wait_exponential(min=4, max=15)

This implementation uses the OpenHands standard (more conservative strategy)
"""

import copy
import json
import logging
import os
import re
from typing import Any, List

logger = logging.getLogger(__name__)

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMAPI
from ..agent import LLMResponse, ToolCall

# Lazy import litellm
_litellm = None

def get_litellm():
    global _litellm
    if _litellm is None:
        import litellm
        litellm.drop_params = True  # Automatically drop unsupported parameters
        litellm.modify_params = True  # Allow parameter modification to adapt to different models (e.g. Bedrock)
        _litellm = litellm
    return _litellm


# OpenHands standard retry configuration
RETRY_NUM_RETRIES = 8  # Maximum number of retries
RETRY_MULTIPLIER = 8   # Exponential backoff multiplier
RETRY_MIN_WAIT = 8     # Minimum wait time (seconds)
RETRY_MAX_WAIT = 64    # Maximum wait time (seconds)
# Total wait time approx: 8 + 16 + 32 + 64 + 64 + 64 + 64 + 64 = 376 seconds


def parse_deepseek_text_tool_calls(content: str) -> List[dict]:
    """
    Parse DeepSeek R1 text-format tool calls
    
    DeepSeek R1 sometimes returns tool calls in text format within message.content,
    instead of the standard OpenAI tool_calls format. This function parses that format.
    
    Format example:
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>tool_name
    ```json
    {"arg": "value"}
    ```
    <｜tool▁call▁end｜><｜tool▁calls▁end｜>
    
    Args:
        content: The content field of the LLM response
        
    Returns:
        List of parsed tool calls, in OpenAI standard format
    """
    tool_calls = []
    
    # Quick check: return immediately if DeepSeek signature markers are not present
    if not content or '<｜tool▁calls▁begin｜>' not in content:
        return tool_calls
    
    # Extract the tool_calls block
    match = re.search(
        r'<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>',
        content,
        re.DOTALL
    )
    if not match:
        return tool_calls
    
    tool_calls_block = match.group(1)
    
    # Parse each tool_call
    # Format: <｜tool▁call▁begin｜>function<｜tool▁sep｜>tool_name\n```json\n{...}\n```<｜tool▁call▁end｜>
    tool_call_pattern = re.compile(
        r'<｜tool▁call▁begin｜>(\w+)<｜tool▁sep｜>(\w+)\s*```json\s*(.*?)\s*```\s*<｜tool▁call▁end｜>',
        re.DOTALL
    )
    
    for i, m in enumerate(tool_call_pattern.finditer(tool_calls_block)):
        tool_type = m.group(1)  # "function"
        tool_name = m.group(2)
        args_str = m.group(3).strip()
        
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = {}
        
        tool_calls.append({
            'id': f'deepseek_tc_{i}',
            'type': tool_type,
            'function': {
                'name': tool_name,
                'arguments': json.dumps(arguments)
            }
        })
    
    if tool_calls:
        logger.info(f"[DeepSeek] Parsed {len(tool_calls)} tool call(s) from text format")
    
    return tool_calls


def parse_plaintext_tool_calls(content: str, available_tools: List[str] = None) -> List[dict]:
    """
    Parse tool calls from plain text (Bedrock DeepSeek-R1 compatible)
    
    When DeepSeek-R1 on Bedrock does not support native function calling,
    the model may write tool names directly in plain text, e.g.:
    
        swebench_finish
        The fix has been applied...
    
    This function detects such patterns and extracts tool calls.
    
    Args:
        content: The content field of the LLM response
        available_tools: List of available tool names (for matching)
        
    Returns:
        List of parsed tool calls
    """
    tool_calls = []
    
    if not content:
        return tool_calls
    
    # Default supported swebench termination tools
    finish_tools = ['swebench_finish', 'terminalbench_finish']
    
    # Check if a finish tool was called at the end of text or on a standalone line
    lines = content.strip().split('\n')
    
    for tool_name in finish_tools:
        # Check if the last few lines contain a tool name (standalone call)
        for line in lines[-5:]:  # Check last 5 lines
            line_stripped = line.strip()
            if line_stripped == tool_name or line_stripped.startswith(f'{tool_name}\n'):
                # Extract content after the tool name as arguments
                idx = content.rfind(tool_name)
                if idx != -1:
                    remaining = content[idx + len(tool_name):].strip()
                    # First line as message
                    message_lines = remaining.split('\n')
                    message = message_lines[0].strip() if message_lines else ""
                    
                    tool_calls.append({
                        'id': f'plaintext_tc_0',
                        'type': 'function',
                        'function': {
                            'name': tool_name,
                            'arguments': json.dumps({'message': message} if message else {})
                        }
                    })
                    logger.info(f"[Plaintext] Parsed tool call: {tool_name}")
                    return tool_calls  # Only return the first match
    
    return tool_calls


def _extract_balanced_json(text: str, start_pos: int) -> tuple:
    """
    Extract a balanced JSON object starting from start_pos
    
    Uses a bracket balancing algorithm, correctly handling nested brackets and escape characters.
    
    Args:
        text: The original text
        start_pos: JSON start position (must be '{')
        
    Returns:
        (json_str, end_pos) or (None, start_pos)
    """
    if start_pos >= len(text) or text[start_pos] != '{':
        return None, start_pos
    
    depth = 0
    in_string = False
    i = start_pos
    
    while i < len(text):
        c = text[i]
        
        # Handle escape characters
        if c == '\\' and i + 1 < len(text):
            i += 2  # Skip the escape character and the next character
            continue
        
        # Handle string boundaries
        if c == '"':
            in_string = not in_string
        elif not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start_pos:i+1], i + 1
        
        i += 1
    
    return None, start_pos


def parse_oss_content_tool_calls(content: str) -> List[dict]:
    """
    Parse tool call format output by OpenAI-oss-120B in content
    
    OpenAI-oss models sometimes write tool calls in content instead of the tool_calls field,
    in the following formats:
    
    Format 1: analysisXXX...assistantcommentary to=functions.TOOL_NAME json{JSON_ARGS}
    Format 2: <|start|>assistant<|channel|>commentary to=functions.TOOL_NAME <|message|>{JSON_ARGS}<|call|>
    
    Args:
        content: The content field of the LLM response
        
    Returns:
        List of parsed tool calls
    """
    tool_calls = []
    
    if not content:
        return tool_calls
    
    # Pattern: to=functions.TOOL_NAME or to=repo_browser.TOOL_NAME
    # Followed by optional json/code keywords, then a JSON object
    pattern = r'to=(?:functions|repo_browser)\.(\w+)'
    
    found_calls = []
    for match in re.finditer(pattern, content):
        tool_name = match.group(1)
        end_pos = match.end()
        
        # Skip possible keywords: json, code, <|message|>, etc.
        remaining = content[end_pos:]
        
        # Find the position of the next '{'
        json_start = -1
        for i, c in enumerate(remaining):
            if c == '{':
                json_start = end_pos + i
                break
            elif c == '\n' or (i > 50):  # Give up if '{' is not found within 50 characters
                break
        
        if json_start != -1:
            json_str, _ = _extract_balanced_json(content, json_start)
            if json_str:
                try:
                    arguments = json.loads(json_str)
                    found_calls.append({
                        'tool_name': tool_name,
                        'arguments': arguments,
                        'position': match.start()
                    })
                except json.JSONDecodeError as e:
                    logger.debug(f"[OSS] Failed to parse JSON for {tool_name}: {e}")
                    # If JSON parsing fails, add empty arguments
                    found_calls.append({
                        'tool_name': tool_name,
                        'arguments': {},
                        'position': match.start()
                    })
    
    # Return all valid tool calls (sorted by position)
    if found_calls:
        found_calls.sort(key=lambda x: x['position'])
        for i, call in enumerate(found_calls):
            tool_calls.append({
                'id': f'oss_tc_{i}',
                'type': 'function',
                'function': {
                    'name': call['tool_name'],
                    'arguments': json.dumps(call['arguments'])
                }
            })
        tool_names = [c['tool_name'] for c in found_calls]
        logger.info(f"[OSS] Parsed {len(tool_calls)} tool call(s) from content: {tool_names}")
    
    return tool_calls


def _fix_schema_for_gemini(schema: dict) -> dict:
    """
    Fix tool schema for Gemini API compatibility
    
    Gemini API has stricter requirements for JSON Schema:
    1. type: array must have items defined
    2. Does not support certain JSON Schema extension fields
    
    This function recursively fixes schema issues.
    """
    if not isinstance(schema, dict):
        return schema
    
    result = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            # Recursively process nested objects
            fixed_value = _fix_schema_for_gemini(value)
            
            # If type is array but items is missing, add default items
            if fixed_value.get("type") == "array" and "items" not in fixed_value:
                fixed_value["items"] = {"type": "string"}  # Default to string array
            
            result[key] = fixed_value
        elif isinstance(value, list):
            # Process each element in the list
            result[key] = [_fix_schema_for_gemini(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    
    # Check if the current level is array type but missing items
    if result.get("type") == "array" and "items" not in result:
        result["items"] = {"type": "string"}
    
    return result


def _fix_tools_for_gemini(tools: list[dict]) -> list[dict]:
    """
    Fix tool list for Gemini API compatibility
    
    Applies schema fixes to function.parameters of each tool
    """
    if not tools:
        return tools
    
    fixed_tools = []
    for tool in tools:
        tool_copy = copy.deepcopy(tool)
        
        # Fix function.parameters
        if "function" in tool_copy and "parameters" in tool_copy["function"]:
            tool_copy["function"]["parameters"] = _fix_schema_for_gemini(
                tool_copy["function"]["parameters"]
            )
        
        fixed_tools.append(tool_copy)
    
    return fixed_tools


class LiteLLMAPI(BaseLLMAPI):
    """
    LiteLLM API
    
    Supports multiple LLM providers:
    - OpenAI: openai/gpt-4, openai/gpt-3.5-turbo
    - Anthropic: anthropic/claude-3-opus, anthropic/claude-3-sonnet
    - AWS Bedrock: bedrock/openai.gpt-oss-120b-1:0, bedrock/anthropic.claude-v2
    - Azure: azure/gpt-4
    - Gemini: gemini/gemini-pro
    - See LiteLLM documentation for more
    
    AWS Bedrock configuration requires setting environment variables:
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
        AWS_REGION_NAME
    """
    
    def __init__(
        self,
        model: str = "bedrock/openai.gpt-oss-120b-1:0",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize LiteLLM API
        
        Args:
            model: LiteLLM model name, in the format provider/model-name
                   e.g.: bedrock/openai.gpt-oss-120b-1:0
            temperature: Sampling temperature
            max_tokens: Maximum number of generated tokens
            **kwargs: Additional arguments passed to litellm.completion
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs
        
        # Validate Bedrock configuration
        if model.startswith("bedrock/"):
            self._validate_bedrock_config()
    
    def _validate_bedrock_config(self):
        """Validate AWS Bedrock configuration"""
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            logger.warning(f"AWS Bedrock requires the following environment variables: {missing}")
            logger.warning("Please set these environment variables or load a .env file via --env-file")
    
    async def _collect_stream_response(self, stream):
        """
        Collect streaming response chunks into a complete response.
        
        For Fireworks AI with max_tokens > 4096, streaming is required.
        This method collects all chunks and reconstructs a standard response.
        """
        from litellm import ModelResponse
        from litellm.types.utils import Choices, Message, Usage
        
        full_content = ""
        tool_calls_list = []  # List of {id, type, function: {name, arguments}}
        finish_reason = None
        usage = None
        model = None
        
        async for chunk in stream:
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            model = chunk.model if hasattr(chunk, 'model') else model
            
            # Collect content
            if hasattr(delta, 'content') and delta.content:
                full_content += delta.content
            
            # Collect tool calls - use index to track position
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc in delta.tool_calls:
                    # Get the index (position in tool_calls array)
                    idx = tc.index if hasattr(tc, 'index') and tc.index is not None else 0
                    
                    # Extend list if needed
                    while len(tool_calls_list) <= idx:
                        tool_calls_list.append({
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    
                    # Update the tool call at this index
                    if tc.id:
                        tool_calls_list[idx]["id"] = tc.id
                    if hasattr(tc, 'type') and tc.type:
                        tool_calls_list[idx]["type"] = tc.type
                    if hasattr(tc, 'function') and tc.function:
                        if hasattr(tc.function, 'name') and tc.function.name:
                            tool_calls_list[idx]["function"]["name"] += tc.function.name
                        if hasattr(tc.function, 'arguments') and tc.function.arguments:
                            tool_calls_list[idx]["function"]["arguments"] += tc.function.arguments
            
            # Get finish reason from last chunk
            if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            
            # Get usage from last chunk if available
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = chunk.usage
        
        # Construct tool_calls list with SimpleNamespace
        tool_calls = None
        if tool_calls_list:
            from types import SimpleNamespace
            tool_calls = []
            for i, tc_data in enumerate(tool_calls_list):
                # Skip if no name (invalid tool call)
                if not tc_data["function"]["name"]:
                    continue
                tc = SimpleNamespace()
                tc.id = tc_data["id"] or f"call_{i}"
                tc.type = tc_data["type"]
                tc.function = SimpleNamespace()
                tc.function.name = tc_data["function"]["name"]
                tc.function.arguments = tc_data["function"]["arguments"]
                tool_calls.append(tc)
        
        # Create a response-like object
        from types import SimpleNamespace
        class StreamedResponse:
            def __init__(self):
                self.choices = [SimpleNamespace()]
                self.choices[0].message = SimpleNamespace()
                self.choices[0].message.content = full_content if full_content else None
                self.choices[0].message.tool_calls = tool_calls if tool_calls else None
                self.choices[0].finish_reason = finish_reason
                self.model = model
                self.usage = usage
        
        return StreamedResponse()

    async def _call_with_retry(self, kwargs: dict) -> Any:
        """
        Call LiteLLM using OpenHands standard retry strategy
        
        Retry strategy (consistent with OpenHands SWE-Bench):
        - num_retries: 5
        - retry_min_wait: 8 seconds
        - retry_max_wait: 64 seconds
        - retry_multiplier: 8
        - Total wait time approx: 8 + 16 + 32 + 64 = 120 seconds
        
        Retryable exceptions:
        - RateLimitError (429)
        - APIConnectionError
        - ServiceUnavailableError (503)
        - Timeout
        - InternalServerError (500)
        """
        import asyncio
        import random
        from litellm.exceptions import (
            APIConnectionError,
            APIError,
            BadRequestError,
            RateLimitError,
            ServiceUnavailableError,
        )
        import litellm
        
        litellm_module = get_litellm()
        
        # Retryable exception types
        # Note: BadRequestError is included because Bedrock OpenAI-oss-120B has intermittent
        # "Unexpected token XXXX while expecting start token 200006" errors
        # This is a backend issue and retrying usually succeeds
        RETRY_EXCEPTIONS = (
            RateLimitError,
            APIConnectionError,
            APIError,
            BadRequestError,  # Bedrock intermittent token parsing errors
            ServiceUnavailableError,
            litellm.Timeout if hasattr(litellm, 'Timeout') else TimeoutError,
        )
        
        last_exception = None
        is_streaming = kwargs.get("stream", False)
        
        for attempt in range(RETRY_NUM_RETRIES):
            try:
                response = await litellm_module.acompletion(**kwargs)
                
                # Handle streaming response
                if is_streaming:
                    response = await self._collect_stream_response(response)
                
                # Check for empty response (Gemini may return empty due to content safety filtering)
                if not response.choices:
                    raise RuntimeError(f"LLM returned empty response (no choices)")
                
                if attempt > 0:
                    print(f"LiteLLM: Succeeded on retry {attempt + 1}/{RETRY_NUM_RETRIES}")
                return response
            except (RuntimeError,) + RETRY_EXCEPTIONS as e:
                last_exception = e
                
                if attempt < RETRY_NUM_RETRIES - 1:
                    # OpenHands standard exponential backoff: multiplier * 2^attempt
                    # min_wait=8, max_wait=64, multiplier=8
                    # Actual wait: min(8 * 2^attempt, 64) = 8, 16, 32, 64, 64
                    base_delay = RETRY_MULTIPLIER * (2 ** attempt)
                    delay = min(base_delay, RETRY_MAX_WAIT)
                    delay = max(delay, RETRY_MIN_WAIT)
                    
                    # Add random jitter (0-20%)
                    jitter = random.uniform(0, 0.2)
                    delay = delay * (1 + jitter)
                    
                    error_type = type(e).__name__
                    print(f"LiteLLM: {error_type}, attempt {attempt + 1}/{RETRY_NUM_RETRIES}, waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                
                # Last retry failed
                raise
            except Exception as e:
                # Non-retryable exception, raise immediately
                raise
        
        # All retries exhausted
        raise last_exception

    async def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> LLMResponse:
        """
        Call LiteLLM API to generate a response
        
        Args:
            messages: List of conversation messages
            tools: List of available tools (OpenAI format)
            
        Returns:
            LLMResponse: Model response
        """
        litellm = get_litellm()
        
        # Build request parameters
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_kwargs
        }
        
        # Fireworks AI requires stream=True for max_tokens > 4096
        if "fireworks-ai" in self.model and self.max_tokens > 4096:
            kwargs["stream"] = True
            # Request usage stats in streaming response (critical for budget tracking!)
            kwargs["stream_options"] = {"include_usage": True}
        
        if tools:
            # For Gemini models, fix tool schema compatibility issues
            if self.model.startswith("gemini/"):
                kwargs["tools"] = _fix_tools_for_gemini(tools)
            else:
                kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
            # Bedrock DeepSeek models: LiteLLM does not list tools/tool_choice as supported params,
            # causing these params to be silently dropped when drop_params=True.
            # Explicitly declare support via allowed_openai_params to bypass drop_params filtering.
            # Note: Only set when tools exist, otherwise LiteLLM tries to process None tools and crashes.
            if "bedrock" in self.model and "deepseek" in self.model:
                kwargs["allowed_openai_params"] = [
                    "tools", "tool_choice",
                    "temperature", "max_tokens", "top_p", "stop",
                ]
        
        # Use OpenHands standard retry strategy (includes empty response retries)
        response = await self._call_with_retry(kwargs)
        
        choice = response.choices[0]
        message = choice.message
        
        # Parse tool calls
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments
                ))
        
        # Fallback 1: Parse DeepSeek R1 text-format tool calls
        # Triggered when message.tool_calls is empty but content contains DeepSeek format
        if not tool_calls and message.content:
            parsed_tcs = parse_deepseek_text_tool_calls(message.content)
            for tc in parsed_tcs:
                try:
                    arguments = json.loads(tc['function']['arguments'])
                except json.JSONDecodeError:
                    arguments = {}
                
                tool_calls.append(ToolCall(
                    id=tc['id'],
                    name=tc['function']['name'],
                    arguments=arguments
                ))
        
        # Fallback 2: Parse Bedrock DeepSeek-R1 plain text tool calls
        # When the above fallback also fails to parse tool calls, try extracting finish tools from plain text
        if not tool_calls and message.content:
            parsed_tcs = parse_plaintext_tool_calls(message.content)
            for tc in parsed_tcs:
                try:
                    arguments = json.loads(tc['function']['arguments'])
                except json.JSONDecodeError:
                    arguments = {}
                
                tool_calls.append(ToolCall(
                    id=tc['id'],
                    name=tc['function']['name'],
                    arguments=arguments
                ))
        
        # Fallback 3: Parse OpenAI-oss tool calls output in content
        # When the model writes tool calls in content instead of the tool_calls field
        if not tool_calls and message.content:
            parsed_tcs = parse_oss_content_tool_calls(message.content)
            for tc in parsed_tcs:
                try:
                    arguments = json.loads(tc['function']['arguments'])
                except json.JSONDecodeError:
                    arguments = {}
                
                tool_calls.append(ToolCall(
                    id=tc['id'],
                    name=tc['function']['name'],
                    arguments=arguments
                ))
        
        # Get token usage statistics
        usage = getattr(response, 'usage', None)
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
        total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0
        
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            prompt_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
