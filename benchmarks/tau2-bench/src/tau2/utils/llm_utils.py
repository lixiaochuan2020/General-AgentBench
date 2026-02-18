import json
import re
import time
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage, Choices
from litellm.types.llms.openai import ChatCompletionAssistantMessage
from litellm.exceptions import RateLimitError, APIError, BadRequestError
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()

def _clean_tool_arguments(args: dict) -> dict:
    """Clean tool arguments by removing empty string keys and invalid entries."""
    if not isinstance(args, dict):
        return args
    # Remove empty string keys and keys that are just whitespace
    cleaned = {k: v for k, v in args.items() if k and k.strip()}
    return cleaned


def safe_json_loads(s):
    if not s:
        return {}
    s = s.strip()
    result = {}
    try:
        result = json.loads(s)
    except:
        pass
    if not result and "}" in s:
        t = s[: s.index("}") + 1]
        try:
            result = json.loads(t)
        except:
            pass
    if not result:
        try:
            t = s.split("}")[0] + "}"
            result = json.loads(t)
        except:
            pass
    # Clean the parsed arguments (remove empty string keys)
    if isinstance(result, dict):
        result = _clean_tool_arguments(result)
    return result if result else {}


if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "total_tokens": usage.total_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def _sanitize_tool_name(name: str) -> str:
    """
    Sanitize tool name to match regex [a-zA-Z0-9_-]+
    Replaces invalid characters with underscore.
    """
    if not name:
        return "unknown_tool"
    # Replace any character that's not alphanumeric, underscore, or hyphen with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Ensure it's not empty after cleaning
    return sanitized if sanitized else "unknown_tool"


def _fix_tool_call_arguments(tool_calls: list[dict]) -> list[dict]:
    """Fix tool calls with invalid arguments (like empty string keys) and sanitize tool names."""
    fixed_calls = []
    for tc in tool_calls:
        # Sanitize tool name
        if "name" in tc:
            tc["name"] = _sanitize_tool_name(tc["name"])
        if "function" in tc:
            if "name" in tc["function"]:
                tc["function"]["name"] = _sanitize_tool_name(tc["function"]["name"])
            # Fix arguments
            if "arguments" in tc["function"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    # Remove empty string keys
                    if isinstance(args, dict):
                        args = {k: v for k, v in args.items() if k and k.strip()}
                    tc["function"]["arguments"] = json.dumps(args if args else {})
                except:
                    tc["function"]["arguments"] = "{}"
        fixed_calls.append(tc)
    return fixed_calls


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            # Bedrock requires non-empty content field
            content = message.content
            if not content or not content.strip():
                content = " "
            litellm_messages.append({"role": "user", "content": content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
                # Fix invalid arguments
                tool_calls = _fix_tool_call_arguments(tool_calls)
            # Bedrock requires non-empty content field, use placeholder if empty
            content = message.content
            if not content or not content.strip():
                content = " "
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def _fix_tool_schemas(tools: list[dict]) -> list[dict]:
    """Fix tool schemas with invalid parameters and sanitize tool names."""
    fixed_tools = []
    for tool in tools:
        if "function" in tool:
            # Sanitize tool name
            if "name" in tool["function"]:
                tool["function"]["name"] = _sanitize_tool_name(tool["function"]["name"])
            # Fix parameters
            if "parameters" in tool["function"]:
                params = tool["function"]["parameters"]
                if isinstance(params, dict):
                    # Remove empty string keys from parameters
                    tool["function"]["parameters"] = {k: v for k, v in params.items() if k and k.strip()}
        fixed_tools.append(tool)
    return fixed_tools


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}

    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    
    # Fix invalid arguments in tools
    if tools:
        tools = _fix_tool_schemas(tools)


    if tools and tool_choice is None:
        tool_choice = "auto"
    
    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
    except (RateLimitError, APIError, BadRequestError) as e:
        if "Too many tokens" in str(e):
            logger.warning(f"[Bedrock] Token overflow, sleeping 15s...")
        elif "Rate limit exceeded" in str(e):
            logger.warning(f"[Bedrock] Rate limit exceeded, sleeping 15s...")
        else:
            logger.error(f"[Bedrock] Non-retryable error: {e}")
            # raise
        response = ModelResponse(
            id="error-" + str(int(time.time())),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choices(
                    finish_reason="error",
                    index=0,
                    message=ChatCompletionAssistantMessage(
                        role="assistant",
                        content=f"Error",
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(
                completion_tokens=0,
                prompt_tokens=len(str(litellm_messages)),
                total_tokens=0,
            ),
        )
    except Exception as e:
        logger.error(e)
        response = ModelResponse(
            id="error-" + str(int(time.time())),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choices(
                    finish_reason="error",
                    index=0,
                    message=ChatCompletionAssistantMessage(
                        role="assistant",
                        content=f"Error",
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(
                completion_tokens=0,
                prompt_tokens=len(str(litellm_messages)),
                total_tokens=0,
            ),
        )
        # raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=safe_json_loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
