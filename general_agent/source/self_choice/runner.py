"""Self-Choice Evaluation Runner"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from ..host import BenchmarkHost
from ..llm_api import LiteLLMAPI
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


@dataclass
class SelfChoiceResult:
    """Self-Choice evaluation result"""
    task_id: str
    model_judgment: str  # "Correct" or "Wrong"
    raw_response: str    # Full LLM response
    prompt_tokens: int
    output_tokens: int
    total_tokens: int


class SelfChoiceRunner:
    """
    Self-Choice evaluation runner
    
    Loads trajectory, uses MCP to get tools info,
    and has the LLM judge whether the Agent's answer is correct.
    """
    
    def __init__(
        self,
        host: BenchmarkHost,
        llm_model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        compress_tools: bool = False,  # Whether to compress tool definitions to reduce token count
        minimal_tools: bool = False,   # Minimal mode - only keep name, description, param names
    ):
        """
        Initialize runner
        
        Args:
            host: BenchmarkHost instance (for obtaining tools schema)
            llm_model: LLM model for judgment
            temperature: LLM temperature
            max_tokens: Maximum output token count
            compress_tools: Whether to compress tool definitions
            minimal_tools: Minimal mode (most token-efficient)
        """
        self.host = host
        self.compress_tools = compress_tools
        self.minimal_tools = minimal_tools
        self.llm = LiteLLMAPI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def _format_tools_schema(self, tools_schema: list[dict]) -> str:
        """Convert tools schema to JSON string as-is"""
        return json.dumps(tools_schema, ensure_ascii=False, indent=2)
    
    def _format_trajectory(
        self,
        messages: list[dict],
        skip_task_messages: bool = True,
        compress: bool = False,
        max_tool_output_len: int = 500,
    ) -> str:
        """
        Convert trajectory to JSON string
        
        Args:
            messages: Message list
            skip_task_messages: Whether to skip the first system and first user message
                              (these are already included as task_description in the prompt)
            compress: Whether to compress (truncate long tool outputs)
            max_tool_output_len: Maximum output length in compression mode
        """
        # Skip the first system and first user message
        if skip_task_messages:
            filtered_messages = []
            skipped_system = False
            skipped_user = False
            
            for msg in messages:
                role = msg.get("role", "")
                
                if role == "system" and not skipped_system:
                    skipped_system = True
                    continue
                
                if role == "user" and not skipped_user:
                    skipped_user = True
                    continue
                
                filtered_messages.append(msg)
        else:
            filtered_messages = messages
        
        if not compress:
            return json.dumps(filtered_messages, ensure_ascii=False, indent=2)
        
        # Compression mode: truncate tool outputs
        compressed = []
        for msg in filtered_messages:
            new_msg = msg.copy()
            content = new_msg.get("content", "")
            if isinstance(content, str) and len(content) > max_tool_output_len:
                new_msg["content"] = content[:max_tool_output_len] + "... [truncated]"
            compressed.append(new_msg)
        
        return json.dumps(compressed, ensure_ascii=False, indent=2)
    
    def _extract_task_description(self, trace_data: dict, max_len: int = 0) -> str:
        """
        Extract task description from trajectory messages
        
        Rule: Concatenate content of first role=system and first role=user messages
        
        Args:
            trace_data: Trace data
            max_len: Maximum character count, 0 means no limit
        """
        messages = trace_data.get("trace", {}).get("messages", [])
        
        system_content = ""
        user_content = ""
        
        for msg in messages:
            if msg.get("role") == "system" and not system_content:
                system_content = msg.get("content", "")
            elif msg.get("role") == "user" and not user_content:
                user_content = msg.get("content", "")
            
            # Stop once both are found
            if system_content and user_content:
                break
        
        # Concatenate
        if system_content and user_content:
            result = f"{system_content}\n\n{user_content}"
        elif user_content:
            result = user_content
        elif system_content:
            result = system_content
        else:
            result = ""
        
        # Truncate (if needed)
        if max_len > 0 and len(result) > max_len:
            result = result[:max_len] + "\n\n... [task description truncated]"
        
        return result
    
    async def evaluate_trajectory(
        self,
        trajectory_file: Path,
        required_servers: list[str] = None,
        distraction_count: int = -1,  # -1 means all
        eval_file: Path = None,  # For extracting questions from evaluation file (mathhay special handling)
        benchmark: str = None,  # Benchmark name, for special handling
    ) -> SelfChoiceResult:
        """
        Evaluate a single trajectory
        
        Args:
            trajectory_file: Trajectory JSON file path
            required_servers: Required servers (for filtering tools)
            distraction_count: Number of distraction tools, -1 means all
            eval_file: Evaluation result file (for mathhay question extraction)
            benchmark: Benchmark name
            
        Returns:
            SelfChoiceResult: Evaluation result
        """
        # Load trajectory
        with open(trajectory_file) as f:
            trace_data = json.load(f)
        
        task_id = trace_data.get("task_id", trajectory_file.stem)
        
        # Get tools description (optional compression/minimal mode)
        if self.minimal_tools:
            # Use plain text format (most token-efficient)
            if required_servers:
                tools_desc = self.host.get_filtered_tools_text(
                    required_client_names=required_servers,
                    distraction_count=distraction_count,  # -1 = all
                    max_description_len=50,
                )
            else:
                tools_desc = self.host.get_tools_text(max_description_len=50)
        else:
            # Use JSON format
            if required_servers:
                tools_schema = self.host.get_filtered_tools_schema(
                    required_client_names=required_servers,
                    distraction_count=distraction_count,  # -1 = all
                    compress=self.compress_tools,
                    minimal=False,
                )
            else:
                tools_schema = self.host.get_tools_schema(
                    compress=self.compress_tools,
                    minimal=False,
                )
            tools_desc = self._format_tools_schema(tools_schema)
        
        # Build prompt (simplified structure)
        # 1. Tools description - loaded from MCP (handled above)
        
        # 2. Trajectory - from trace, skip first system and user message
        #    because these are already included as task_description in the prompt
        #    Also compress trajectory in minimal mode
        compress_trajectory = self.minimal_tools
        trajectory = self._format_trajectory(
            trace_data.get("trace", {}).get("messages", []),
            skip_task_messages=True,
            compress=compress_trajectory,
            max_tool_output_len=500,
        )
        
        # 3. Task description - use different strategies based on benchmark
        #    For mathhay: extract question from eval_file (avoid truncating huge haystack)
        #    For other benchmarks: extract from messages (truncate in minimal mode)
        if benchmark == "mathhay" and eval_file and Path(eval_file).exists():
            # Extract question and answer from evaluation file
            with open(eval_file) as f:
                eval_data = json.load(f)
            question = eval_data.get("question", "")
            predicted_answer = eval_data.get("predicted_answer", "")
            task_desc = f"Question: {question}\n\nPredicted Answer: {predicted_answer}"
        else:
            # Default: extract from messages
            max_task_desc_len = 8000 if self.minimal_tools else 0  # ~2000 tokens
            task_desc = self._extract_task_description(trace_data, max_len=max_task_desc_len)
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            tools_description=tools_desc,
            trajectory=trajectory,
            task_description=task_desc,
        )
        
        # Call LLM (no tools, single turn)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        response = await self.llm.generate(messages, tools=None)
        
        # Extract judgment
        raw_response = response.content or ""
        judgment = self._extract_judgment(raw_response)
        
        return SelfChoiceResult(
            task_id=task_id,
            model_judgment=judgment,
            raw_response=raw_response,
            prompt_tokens=response.prompt_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        )
    
    def _extract_judgment(self, response: str) -> str:
        """Extract judgment from LLM response"""
        # Match <judgment>...</judgment>, case-insensitive
        pattern = r"(?i)<judgment>\s*(Correct|Wrong)\s*</judgment>"
        match = re.search(pattern, response)
        
        if match:
            return match.group(1).capitalize()
        
        # Fallback: check if response contains Correct or Wrong
        if "Correct" in response:
            return "Correct"
        if "Wrong" in response:
            return "Wrong"
        
        return "Unknown"
