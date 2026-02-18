"""
Universal Agent - General-purpose LLM Agent

Runs inside the Host, with access to all tools from all benchmarks.
Not bound to any specific benchmark; a true All-in-One Agent.

Sequential Scaling Support:
- EXTEND: When budget is sufficient but the model wants to stop, inject a continue prompt
- STOP: When budget is insufficient but the model wants to continue, inject a stop prompt
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, Literal

from loguru import logger as loguru_logger

from .host import BenchmarkHost
from .scaling import ScalingConfig, ScalingCheckpoint, CheckpointStore
from .config import (
    get_task_timeout,
    get_max_execution_rounds,
    get_content_truncate_length,
    get_error_truncate_length,
    get_error_display_prefix,
)

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """LLM client protocol"""
    
    async def generate(
        self, 
        messages: list[dict], 
        tools: list[dict] | None = None
    ) -> "LLMResponse": 
        ...


@dataclass
class ToolCall:
    """Tool call"""
    name: str
    arguments: dict[str, Any]
    id: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = f"call_{uuid.uuid4().hex[:8]}"


@dataclass  
class LLMResponse:
    """LLM response"""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    # Token statistics (MCP-bench format)
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Message:
    """Conversation message"""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response
    name: Optional[str] = None  # For tool response
    
    def to_dict(self) -> dict:
        """Convert to OpenAI format"""
        d = {"role": self.role}
        
        if self.content is not None:
            d["content"] = self.content
        
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments
                    }
                }
                for tc in self.tool_calls
            ]
            if self.content is None:
                d["content"] = None
        
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        
        if self.name:
            d["name"] = self.name
            
        return d


@dataclass
class AgentStep:
    """Agent execution step"""
    step: int
    timestamp: str
    message_type: str  # "llm_response", "tool_call", "tool_result", "user_message"
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None
    tool_result: Optional[str] = None
    tool_error: bool = False


@dataclass
class AgentRound:
    """Agent execution round (MCP-bench format)"""
    round_number: int
    reasoning: str  # LLM's reasoning/thinking process
    should_continue: bool
    prompt_tokens: int
    output_tokens: int
    round_total_tokens: int
    cumulative_total_tokens: int
    incremental_tokens: int
    tools_executed: int
    planned_tools: list[dict] = field(default_factory=list)  # [{"tool": str, "parameters": dict}] Tools planned by LLM for execution
    executions: list[dict] = field(default_factory=list)  # [{"tool": str, "success": bool, "parameters": dict, "result": str, "planned_layer": None}]


@dataclass
class AgentTrace:
    """Agent execution trace"""
    task_id: str
    messages: list[Message] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    rounds: list[AgentRound] = field(default_factory=list)  # New: rounds in MCP format
    final_response: Optional[str] = None
    total_steps: int = 0
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: float = 0.0
    # Token statistics
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary"""
        return {
            "task_id": self.task_id,
            "messages": [m.to_dict() for m in self.messages],
            "steps": [
                {
                    "step": s.step,
                    "timestamp": s.timestamp,
                    "message_type": s.message_type,
                    "content": s.content,
                    "tool_name": s.tool_name,
                    "tool_arguments": s.tool_arguments,
                    "tool_result": s.tool_result,
                    "tool_error": s.tool_error,
                }
                for s in self.steps
            ],
            "rounds": [
                {
                    "round_number": r.round_number,
                    "reasoning": r.reasoning,
                    "should_continue": r.should_continue,
                    "prompt_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "round_total_tokens": r.round_total_tokens,
                    "cumulative_total_tokens": r.cumulative_total_tokens,
                    "incremental_tokens": r.incremental_tokens,
                    "tools_executed": r.tools_executed,
                    "executions": r.executions,
                }
                for r in self.rounds
            ],
            "final_response": self.final_response,
            "total_steps": self.total_steps,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }


class UniversalAgent:
    """
    Universal Agent - Runs inside the Host
    
    Features:
    - Access to all tools from all benchmarks
    - Not bound to any specific benchmark
    - Supports multi-turn conversation (interacting with User Simulator)
    """
    
    # MCP-bench style Planning System Prompt
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI agent that can solve a wide variety of problems, including searching the web for information, writing and running code, performing calculations and logical reasoning, and interacting with external services. You can choose to use tools when helpful, or solve problems through your own reasoning.

## Tool Selection
- CAREFULLY read tool names and descriptions before selecting
- Choose tools that are DIRECTLY relevant to the current task
- AVOID REDUNDANT CALLS: Don't repeat successful tools unless specifically needed
- If no tools are needed, solve the problem through reasoning alone

## Execution Strategy
- Analyze the task to understand what information or actions are needed
- Decide whether to use tools, reason independently, or combine both approaches
- If using tools, identify which are most relevant based on their names and descriptions
- BUILD ON PREVIOUS RESULTS: Use information from previous tool calls
- If a tool returns an error, try an alternative approach or tool

## Response Guidelines
- Follow any policies (<policy>) or constraints provided in the task
- When you have gathered sufficient information, provide a clear final answer
- If a task cannot be completed with available tools, try solving it through reasoning
- Do not guess or make up information - only use data from tool results or verified reasoning
"""

    def __init__(
        self,
        host: BenchmarkHost,
        llm_client: LLMClient,
        system_prompt: Optional[str] = None,
        max_steps: int = None,
        task_timeout: int = None,
        compress_tools: bool = False,
        tool_description_max_len: int = 75,
        # Sequential Scaling parameters
        target_budget: Optional[int] = None,
        scaling_config: Optional[ScalingConfig] = None,
        force_max_steps: bool = False,
    ):
        """
        Initialize the Agent
        
        Args:
            host: BenchmarkHost instance (provides all tools)
            llm_client: LLM client
            system_prompt: System prompt
            max_steps: Maximum steps (default from config: max_execution_rounds=800)
            task_timeout: Task timeout in seconds (default from config: task_timeout=5000)
            compress_tools: Whether to compress tool definitions (saves ~30-50% tool tokens)
            tool_description_max_len: Max length of tool description when compressed (chars, default 75)
            target_budget: Token budget for scaling mode (None = normal mode)
            scaling_config: Configuration for EXTEND/STOP prompts and thresholds
            force_max_steps: If True, enforce max_steps even in scaling mode
        """
        self.host = host
        self.llm = llm_client
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps if max_steps is not None else get_max_execution_rounds()
        self.task_timeout = task_timeout if task_timeout is not None else get_task_timeout()
        
        # Tool compression config
        self.compress_tools = compress_tools
        self.tool_description_max_len = tool_description_max_len
        
        # Content truncation config
        self.content_truncate_length = get_content_truncate_length()
        self.error_truncate_length = get_error_truncate_length()
        self.error_display_prefix = get_error_display_prefix()
        
        # Tool filtering config
        self._active_tools_schema: Optional[list[dict]] = None
        self._excluded_clients: list[str] | None = None  # Client names to completely exclude
        
        # Sequential Scaling configuration
        self.target_budget = target_budget  # None = normal mode (no budget limit)
        self.scaling_config = scaling_config or ScalingConfig()
        self.force_max_steps = force_max_steps  # If True, enforce max_steps even in scaling mode
        self.extend_rounds: list[int] = []  # Track rounds where EXTEND was triggered
        self.stop_rounds: list[int] = []    # Track rounds where STOP was triggered
        
        # Accumulated information tracking (mcp-bench compatible)
        self.accumulated_information: str = ""
        self.accumulated_information_uncompressed: str = ""
    
    def set_excluded_clients(self, excluded_client_names: list[str] | None):
        """
        Set the client names to completely exclude
        
        Used for swebench/terminalbench mutual exclusion: exclude one's tools when running the other
        
        Args:
            excluded_client_names: List of client names to exclude
        """
        self._excluded_clients = excluded_client_names
        # Clear cached active tools to force recalculation
        self._active_tools_schema = None
        if excluded_client_names:
            logger.info(f"Set excluded clients: {excluded_client_names}")
    
    def set_active_tools(
        self,
        required_client_names: list[str],
        distraction_count: int = 10,
        seed: int | None = None,
        excluded_client_names: list[str] | None = None,
    ) -> int:
        """
        Set the active tools for the current task (required + distraction)
        
        Args:
            required_client_names: Required client names (task-related servers)
            distraction_count: Number of distraction servers
            seed: Random seed (for reproducibility)
            excluded_client_names: Client names to completely exclude (e.g., exclude terminalbench when running swebench)
            
        Returns:
            Total number of active tools
        """
        self._active_tools_schema = self.host.get_filtered_tools_schema(
            required_client_names=required_client_names,
            distraction_count=distraction_count,
            seed=seed,
            compress=self.compress_tools,
            max_description_len=self.tool_description_max_len,
            excluded_client_names=excluded_client_names,
        )
        logger.info(f"Set {len(self._active_tools_schema)} active tools (compress={self.compress_tools})")
        return len(self._active_tools_schema)
    
    def clear_active_tools(self):
        """Clear active tool filters, restore to using all tools"""
        self._active_tools_schema = None
    
    def get_tools_schema(self) -> list[dict]:
        """Get the currently active tools schema (returns filtered if filter is set; otherwise returns all)"""
        if self._active_tools_schema is not None:
            return self._active_tools_schema
        return self.host.get_tools_schema(
            compress=self.compress_tools,
            max_description_len=self.tool_description_max_len,
            excluded_client_names=self._excluded_clients,
        )
    
    def _detect_answer_tag(self, content: str) -> tuple[bool, Optional[str]]:
        """
        Detect whether the LLM response content contains an <answer>...</answer> tag.
        
        Args:
            content: LLM response content
            
        Returns:
            tuple: (whether answer tag was detected, extracted answer content)
        """
        if not content:
            return False, None
        
        if "<answer>" in content and "</answer>" in content:
            try:
                # Extract content between <answer> and </answer>
                answer = content.split("<answer>")[1].split("</answer>")[0].strip()
                return True, answer
            except (IndexError, ValueError):
                # Parsing failed, but tag exists
                return True, content
        
        return False, None

    def reset_scaling_state(self):
        """Reset scaling state for a new task (call at start of each task)"""
        self.extend_rounds = []
        self.stop_rounds = []
    
    def _decide_scaling_action(
        self, 
        response: "LLMResponse", 
        trace: "AgentTrace"
    ) -> Literal["EXTEND", "STOP", "NONE"]:
        """
        Decide whether to inject EXTEND/STOP prompt based on budget and model behavior.
        
        Logic:
        - STOP: budget exceeded → inject STOP prompt, let model generate one more round
        - EXTEND: budget not exceeded + no tool_calls → inject EXTEND prompt to continue
        - NONE: budget not exceeded + has tool_calls → let it continue naturally
        
        Args:
            response: Current LLM response
            trace: Current agent trace (for token statistics)
            
        Returns:
            "EXTEND", "STOP", or "NONE"
        """
        # Normal mode: no budget limit, no scaling
        if self.target_budget is None:
            return "NONE"
        
        # Use current round's prompt_tokens as context window size
        current_context_size = response.prompt_tokens
        
        # Check if budget is exceeded
        budget_exceeded = current_context_size > self.target_budget
        
        # If budget exceeded, always trigger STOP (regardless of tool_calls)
        if budget_exceeded:
            return "STOP"
        
        # Budget not exceeded: check if model wants to stop early
        has_tool_calls = bool(response.tool_calls)
        if not has_tool_calls:
            # Model wants to stop, but budget not exceeded yet
            # → Encourage to continue exploring
            return "EXTEND"
        
        # Budget not exceeded + has tool_calls → let it continue naturally
        return "NONE"

    async def run(
        self,
        task_id: str,
        instruction: str,
        policy: Optional[str] = None,
        use_synthesize: bool = True,
        benchmark: Optional[str] = None,
    ) -> AgentTrace:
        """
        Run a single task (without User Simulator)
        
        Args:
            task_id: Task ID
            instruction: User instruction
            policy: Optional policy (appended to system prompt)
            use_synthesize: Whether to use LLM to synthesize an answer at the end (needed for mcp-bench, not for search benchmark)
            benchmark: Benchmark type (for controlling specific behavior, e.g., explicit termination for swebench)
            
        Returns:
            AgentTrace: Execution trace
        """
        try:
            return await asyncio.wait_for(
                self._run_impl(task_id, instruction, policy, use_synthesize, benchmark),
                timeout=self.task_timeout
            )
        except asyncio.TimeoutError:
            trace = AgentTrace(
                task_id=task_id,
                start_time=datetime.now().isoformat(),
                error=f"Task timeout after {self.task_timeout} seconds",
            )
            trace.end_time = datetime.now().isoformat()
            return trace
    
    async def _run_impl(
        self,
        task_id: str,
        instruction: str,
        policy: Optional[str] = None,
        use_synthesize: bool = True,
        benchmark: Optional[str] = None,
    ) -> AgentTrace:
        """
        Actual task execution implementation (internal method, with timeout protection)
        
        Supports Sequential Scaling:
        - In Scaling mode (target_budget is set): ignores max_steps, uses token budget
        - In Normal mode (target_budget is None): uses max_steps as before
        
        Args:
            benchmark: Benchmark type (for controlling specific behavior, e.g., explicit termination for swebench)
        """
        start_time = datetime.now()
        trace = AgentTrace(task_id=task_id, start_time=start_time.isoformat())
        
        # Reset accumulated information
        self.reset_accumulated_information()
        
        # Reset scaling state for new task
        self.reset_scaling_state()
        
        # Save instruction for later synthesize
        self._current_instruction = instruction
        
        # Build system prompt
        full_system_prompt = self.system_prompt
        if policy:
            full_system_prompt = f"{self.system_prompt}\n\n## Policy\n\n{policy}"

        
        # Initialize messages
        messages = [
            Message(role="system", content=full_system_prompt),
            Message(role="user", content=instruction),
        ]
        trace.messages = messages.copy()
        
        # Get tools
        tools_schema = self.get_tools_schema()
        
        # Determine max iterations: infinite in Scaling mode, max_steps in Normal mode
        # If force_max_steps is True, always use max_steps even in scaling mode
        is_scaling_mode = self.target_budget is not None
        if self.force_max_steps:
            max_iterations = self.max_steps
        else:
            max_iterations = 100000 if is_scaling_mode else self.max_steps  # Use large number instead of inf for range
        
        try:
            for step in range(max_iterations):
                timestamp = datetime.now().isoformat()
                
                # Call LLM
                messages_dict = [m.to_dict() for m in messages]
                response = await self.llm.generate(messages_dict, tools=tools_schema)
                
                # Update trace token statistics
                trace.total_prompt_tokens += response.prompt_tokens
                trace.total_output_tokens += response.output_tokens
                trace.total_tokens += response.total_tokens
                
                # Extract planned_tools from tool_calls
                planned_tools = [
                    {"tool": tc.name, "parameters": tc.arguments}
                    for tc in response.tool_calls
                ] if response.tool_calls else []
                
                # Create Round record (MCP format)
                current_round = AgentRound(
                    round_number=step + 1,
                    reasoning=response.content or "",
                    should_continue=bool(response.tool_calls),
                    prompt_tokens=response.prompt_tokens,
                    output_tokens=response.output_tokens,
                    round_total_tokens=response.total_tokens,
                    cumulative_total_tokens=trace.total_tokens,
                    incremental_tokens=response.total_tokens,
                    tools_executed=len(response.tool_calls),
                    planned_tools=planned_tools,
                    executions=[],
                )
                
                # === Scaling Decision (only in Scaling mode) ===
                scaling_action = self._decide_scaling_action(response, trace) if is_scaling_mode else "NONE"
                
                # Debug: log every round's token status in scaling mode
                if is_scaling_mode:
                    loguru_logger.debug(
                        f"[Scaling] round={step+1} prompt_tokens={response.prompt_tokens} "
                        f"budget={self.target_budget} has_tool_calls={bool(response.tool_calls)} "
                        f"action={scaling_action}"
                    )
                
                if scaling_action == "EXTEND":
                    # Model wants to stop (no tool_calls), but budget is sufficient
                    # → Inject EXTEND prompt to encourage continuation
                    self.extend_rounds.append(step + 1)
                    loguru_logger.info(
                        f"[EXTEND] round={step+1} prompt_tokens={response.prompt_tokens} "
                        f"budget={self.target_budget} remaining={self.target_budget - response.prompt_tokens}"
                    )
                    
                    # Add assistant response (no tool_calls)
                    current_round.should_continue = True  # Mark as continuing due to EXTEND
                    trace.rounds.append(current_round)
                    
                    assistant_msg = Message(role="assistant", content=response.content)
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    # Inject EXTEND prompt
                    extend_msg = Message(role="user", content=self.scaling_config.extend_prompt)
                    messages.append(extend_msg)
                    trace.messages.append(extend_msg)
                    
                    trace.steps.append(AgentStep(
                        step=step,
                        timestamp=timestamp,
                        message_type="llm_response",
                        content=f"[EXTEND triggered] {response.content}",
                    ))
                    
                    continue  # Next iteration
                
                elif scaling_action == "STOP":
                    # Budget exceeded → inject STOP prompt, let model generate one more round
                    # If that round has tool_calls, execute them then stop
                    # If no tool_calls, just stop
                    self.stop_rounds.append(step + 1)
                    loguru_logger.info(
                        f"[STOP] round={step+1} prompt_tokens={response.prompt_tokens} "
                        f"budget={self.target_budget}"
                    )
                    
                    # Add assistant message (may or may not have tool_calls)
                    assistant_msg = Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls if response.tool_calls else None,
                    )
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    # Execute tool calls if any
                    if response.tool_calls:
                        for tool_call in response.tool_calls:
                            trace.steps.append(AgentStep(
                                step=step,
                                timestamp=timestamp,
                                message_type="tool_call",
                                tool_name=tool_call.name,
                                tool_arguments=tool_call.arguments,
                            ))
                            
                            try:
                                result, _ = await self.host.route_tool_call(
                                    tool_call.name,
                                    tool_call.arguments
                                )
                                result_text = self._extract_result_text(result, tool_call.name)
                                tool_error = False
                                success = True
                            except Exception as e:
                                result_text = f"Error: {str(e)}"
                                tool_error = True
                                success = False
                            
                            trace.steps.append(AgentStep(
                                step=step,
                                timestamp=datetime.now().isoformat(),
                                message_type="tool_result",
                                tool_name=tool_call.name,
                                tool_result=result_text,
                                tool_error=tool_error,
                            ))
                            
                            current_round.executions.append({
                                "tool": tool_call.name,
                                "success": success,
                                "parameters": tool_call.arguments,
                                "result": result_text[:500],
                                "planned_layer": None,
                            })
                            
                            tool_msg = Message(
                                role="tool",
                                content=result_text,
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                            messages.append(tool_msg)
                            trace.messages.append(tool_msg)
                    
                    trace.rounds.append(current_round)
                    
                    # Inject STOP prompt
                    stop_msg = Message(role="user", content=self.scaling_config.stop_prompt)
                    messages.append(stop_msg)
                    trace.messages.append(stop_msg)
                    
                    # Get final response
                    final_response = await self.llm.generate(
                        [m.to_dict() for m in messages], 
                        tools=tools_schema,
                    )
                    
                    # Update token statistics
                    trace.total_prompt_tokens += final_response.prompt_tokens
                    trace.total_output_tokens += final_response.output_tokens
                    trace.total_tokens += final_response.total_tokens
                    
                    final_msg = Message(role="assistant", content=final_response.content)
                    messages.append(final_msg)
                    trace.messages.append(final_msg)
                    
                    trace.steps.append(AgentStep(
                        step=step + 1,
                        timestamp=datetime.now().isoformat(),
                        message_type="llm_response",
                        content=f"[STOP final] {final_response.content}",
                    ))
                    
                    # If STOP response still has tool_calls, execute them then stop
                    if final_response.tool_calls:
                        for tool_call in final_response.tool_calls:
                            try:
                                result, _ = await self.host.route_tool_call(
                                    tool_call.name,
                                    tool_call.arguments
                                )
                                result_text = self._extract_result_text(result, tool_call.name)
                            except Exception as e:
                                result_text = f"Error: {str(e)}"
                            
                            tool_msg = Message(
                                role="tool",
                                content=result_text,
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                            messages.append(tool_msg)
                            trace.messages.append(tool_msg)
                    
                    trace.final_response = final_response.content
                    trace.total_steps = step + 2  # Current step + final response step
                    break  # Force terminate
                
                # === Normal flow (no scaling intervention) ===
                if response.tool_calls:
                    # Has tool calls
                    assistant_msg = Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        trace.steps.append(AgentStep(
                            step=step,
                            timestamp=timestamp,
                            message_type="tool_call",
                            tool_name=tool_call.name,
                            tool_arguments=tool_call.arguments,
                        ))
                        
                        # Call tool
                        try:
                            result, _ = await self.host.route_tool_call(
                                tool_call.name,
                                tool_call.arguments
                            )
                            result_text = self._extract_result_text(result, tool_call.name)
                            tool_error = False
                            success = True
                        except Exception as e:
                            result_text = f"Error: {str(e)}"
                            tool_error = True
                            success = False
                        
                        # Record result
                        trace.steps.append(AgentStep(
                            step=step,
                            timestamp=datetime.now().isoformat(),
                            message_type="tool_result",
                            tool_name=tool_call.name,
                            tool_result=result_text,
                            tool_error=tool_error,
                        ))
                        
                        # Add to round's executions (MCP format)
                        current_round.executions.append({
                            "tool": tool_call.name,
                            "success": success,
                            "parameters": tool_call.arguments,
                            "result": result_text[:500],  # Truncate result
                            "planned_layer": None,
                        })
                        
                        # Add tool response to messages
                        tool_msg = Message(
                            role="tool",
                            content=result_text,
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                        messages.append(tool_msg)
                        trace.messages.append(tool_msg)
                    
                    # === SWEBench AgentFinishAction mechanism ===
                    # Only when benchmark == "swebench", terminate immediately after calling swebench_finish
                    # To align with the original OpenHands behavior
                    if benchmark == "swebench":
                        swebench_finish_called = any(tc.name == "swebench_finish" for tc in response.tool_calls)
                        
                        if swebench_finish_called:
                            current_round.should_continue = False
                            trace.rounds.append(current_round)
                            
                            # Get the return content of the swebench_finish tool as final_response
                            finish_message = None
                            for tc in response.tool_calls:
                                if tc.name == "swebench_finish":
                                    for msg in reversed(messages):
                                        if msg.role == "tool" and msg.name == tc.name:
                                            try:
                                                result = json.loads(msg.content)
                                                finish_message = result.get("message", "")
                                            except:
                                                finish_message = msg.content
                                            break
                                    break
                            
                            trace.final_response = finish_message or response.content or "Task completed via swebench_finish"
                            trace.total_steps = step + 1
                            logger.info(f"[Agent] swebench_finish called, terminating immediately (OpenHands alignment)")
                            break  # Terminate immediately, no more LLM calls
                    
                    # === Search Domain <answer> tag detection mechanism ===
                    # Only when benchmark == "search", detect <answer> tag and terminate immediately
                    # To align with the original deepresearch_llm_modeling behavior
                    if benchmark == "search":
                        has_answer, extracted_answer = self._detect_answer_tag(response.content)
                        
                        if has_answer:
                            current_round.should_continue = False
                            trace.rounds.append(current_round)
                            
                            # Use extracted answer as final_response
                            trace.final_response = extracted_answer or response.content
                            trace.total_steps = step + 1
                            logger.info(f"[Agent] Search <answer> tag detected, terminating immediately (deepresearch alignment)")
                            break  # Terminate immediately, no more LLM calls
                    
                    # Update accumulated_information (after each round)
                    round_info = self._build_round_summary(current_round)
                    self._update_accumulated_information(step + 1, round_info)
                    
                    # Add round to trace
                    trace.rounds.append(current_round)
                
                else:
                    # No tool calls, this is the final reply
                    current_round.should_continue = False
                    trace.rounds.append(current_round)
                    
                    assistant_msg = Message(role="assistant", content=response.content)
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    trace.steps.append(AgentStep(
                        step=step,
                        timestamp=timestamp,
                        message_type="llm_response",
                        content=response.content,
                    ))
                    
                    # If synthesize is enabled and there is accumulated information, use synthesize to generate the final answer
                    if use_synthesize and self.accumulated_information and len(trace.rounds) > 1:
                        try:
                            synthesized = await self._synthesize_final_solution(
                                self._current_instruction,
                                len(trace.steps)
                            )
                            trace.final_response = synthesized
                        except Exception as synth_error:
                            logger.warning(f"Failed to synthesize final solution: {synth_error}")
                            trace.final_response = response.content
                    else:
                        trace.final_response = response.content
                    
                    trace.total_steps = step + 1
                    break
            
            else:
                # Reached max steps - also attempt synthesize (if enabled)
                trace.error = f"Reached max steps ({self.max_steps})"
                trace.total_steps = self.max_steps
                if use_synthesize and self.accumulated_information:
                    try:
                        synthesized = await self._synthesize_final_solution(
                            self._current_instruction,
                            len(trace.steps)
                        )
                        trace.final_response = synthesized
                    except Exception as synth_error:
                        logger.warning(f"Failed to synthesize final solution: {synth_error}")
                
        except Exception as e:
            trace.error = str(e)
            logger.exception(f"Error running task {task_id}")
        
        # Record end time and duration
        end_time = datetime.now()
        trace.end_time = end_time.isoformat()
        trace.duration = (end_time - start_time).total_seconds()
        
        return trace
    
    async def run_with_custom_tools(
        self,
        task_id: str,
        instruction: str,
        tools_schema: list[dict],
        tool_handler,  # async callable: (tool_name: str, args: dict) -> str
        policy: Optional[str] = None,
    ) -> AgentTrace:
        """
        Run a task using custom tools (without MCP Host)
        
        Used for scenarios requiring dynamic Docker containers, such as SWE-Bench.
        
        Args:
            task_id: Task ID
            instruction: User instruction
            tools_schema: List of tool schemas (OpenAI format)
            tool_handler: Async tool handler function, takes (tool_name, args) and returns result string
            policy: Optional policy
            
        Returns:
            AgentTrace: Execution trace
        """
        try:
            return await asyncio.wait_for(
                self._run_with_custom_tools_impl(
                    task_id, instruction, tools_schema, tool_handler, policy
                ),
                timeout=self.task_timeout
            )
        except asyncio.TimeoutError:
            trace = AgentTrace(
                task_id=task_id,
                start_time=datetime.now().isoformat(),
                error=f"Task timeout after {self.task_timeout} seconds",
            )
            trace.end_time = datetime.now().isoformat()
            return trace
    
    async def _run_with_custom_tools_impl(
        self,
        task_id: str,
        instruction: str,
        tools_schema: list[dict],
        tool_handler,
        policy: Optional[str] = None,
    ) -> AgentTrace:
        """
        Implementation for running a task with custom tools
        """
        start_time = datetime.now()
        trace = AgentTrace(task_id=task_id, start_time=start_time.isoformat())
        
        # Build system prompt
        full_system_prompt = self.system_prompt
        if policy:
            full_system_prompt = f"{self.system_prompt}\n\n## Policy\n\n{policy}"
        
        # Initialize messages
        messages = [
            Message(role="system", content=full_system_prompt),
            Message(role="user", content=instruction),
        ]
        trace.messages = messages.copy()
        
        # Convert tool format to OpenAI function format
        formatted_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                }
            }
            for tool in tools_schema
        ]
        
        try:
            for step in range(self.max_steps):
                timestamp = datetime.now().isoformat()
                
                # Call LLM
                messages_dict = [m.to_dict() for m in messages]
                response = await self.llm.generate(messages_dict, tools=formatted_tools)
                
                # Update trace token statistics
                trace.total_prompt_tokens += response.prompt_tokens
                trace.total_output_tokens += response.output_tokens
                trace.total_tokens += response.total_tokens
                
                # Create Round record
                planned_tools = [
                    {"tool": tc.name, "parameters": tc.arguments}
                    for tc in response.tool_calls
                ] if response.tool_calls else []
                
                current_round = AgentRound(
                    round_number=step + 1,
                    reasoning=response.content or "",
                    should_continue=bool(response.tool_calls),
                    prompt_tokens=response.prompt_tokens,
                    output_tokens=response.output_tokens,
                    round_total_tokens=response.total_tokens,
                    cumulative_total_tokens=trace.total_tokens,
                    incremental_tokens=response.total_tokens,
                    tools_executed=len(response.tool_calls),
                    planned_tools=planned_tools,
                    executions=[],
                )
                
                if response.tool_calls:
                    # Has tool calls
                    assistant_msg = Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        trace.steps.append(AgentStep(
                            step=step,
                            timestamp=timestamp,
                            message_type="tool_call",
                            tool_name=tool_call.name,
                            tool_arguments=tool_call.arguments,
                        ))
                        
                        # Call custom tool handler
                        try:
                            if asyncio.iscoroutinefunction(tool_handler):
                                result_text = await tool_handler(tool_call.name, tool_call.arguments)
                            else:
                                result_text = tool_handler(tool_call.name, tool_call.arguments)
                            tool_error = False
                            success = True
                        except Exception as e:
                            result_text = f"Error: {str(e)}"
                            tool_error = True
                            success = False
                        
                        # Truncate overly long results
                        if len(result_text) > 50000:
                            result_text = result_text[:50000] + "\n<response clipped>"
                        
                        # Record result
                        trace.steps.append(AgentStep(
                            step=step,
                            timestamp=datetime.now().isoformat(),
                            message_type="tool_result",
                            tool_name=tool_call.name,
                            tool_result=result_text,
                            tool_error=tool_error,
                        ))
                        
                        # Add to round's executions
                        current_round.executions.append({
                            "tool": tool_call.name,
                            "success": success,
                            "parameters": tool_call.arguments,
                            "result": result_text[:500],
                            "planned_layer": None,
                        })
                        
                        # Add tool response to messages
                        tool_msg = Message(
                            role="tool",
                            content=result_text,
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                        messages.append(tool_msg)
                        trace.messages.append(tool_msg)
                        
                        # Check if this is a finish tool
                        if "finish" in tool_call.name.lower():
                            trace.final_response = response.content or result_text
                            trace.total_steps = step + 1
                            trace.rounds.append(current_round)
                            
                            end_time = datetime.now()
                            trace.end_time = end_time.isoformat()
                            trace.duration = (end_time - start_time).total_seconds()
                            return trace
                    
                    # Add round to trace
                    trace.rounds.append(current_round)
                
                else:
                    # No tool calls, this is the final reply
                    current_round.should_continue = False
                    trace.rounds.append(current_round)
                    
                    assistant_msg = Message(role="assistant", content=response.content)
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    trace.steps.append(AgentStep(
                        step=step,
                        timestamp=timestamp,
                        message_type="llm_response",
                        content=response.content,
                    ))
                    
                    trace.final_response = response.content
                    trace.total_steps = step + 1
                    break
            
            else:
                # Reached max steps
                trace.error = f"Reached max steps ({self.max_steps})"
                trace.total_steps = self.max_steps
                
        except Exception as e:
            trace.error = str(e)
            logger.exception(f"Error running task {task_id}")
        
        # Record end time and duration
        end_time = datetime.now()
        trace.end_time = end_time.isoformat()
        trace.duration = (end_time - start_time).total_seconds()
        
        return trace
    
    async def run_from_checkpoint(
        self,
        checkpoint: "ScalingCheckpoint",
        target_budget: int,
        policy: Optional[str] = None,
        use_synthesize: bool = True,
        benchmark: Optional[str] = None,
    ) -> AgentTrace:
        """
        Resume execution from checkpoint (Sequential Scaling core method)
        
        Key features:
        1. Token statistics accumulate from checkpoint (not starting from 0)
        2. Message history has STOP_PROMPT cleaned (via CheckpointStore.clean_stop_prompts())
        3. Continues execution until reaching the new target_budget
        
        Args:
            checkpoint: Cleaned ScalingCheckpoint (STOP_PROMPT removed)
            target_budget: New token budget (must be greater than checkpoint's budget)
            policy: Optional policy
            use_synthesize: Whether to use LLM to synthesize an answer at the end
            benchmark: Benchmark type (for controlling specific behavior)
            
        Returns:
            AgentTrace: Complete execution trace (prefix + new execution)
        """
        try:
            return await asyncio.wait_for(
                self._run_from_checkpoint_impl(
                    checkpoint, target_budget, policy, use_synthesize, benchmark
                ),
                timeout=self.task_timeout
            )
        except asyncio.TimeoutError:
            trace = AgentTrace(
                task_id=checkpoint.task_id,
                start_time=datetime.now().isoformat(),
                error=f"Task timeout after {self.task_timeout} seconds (from checkpoint)",
            )
            trace.end_time = datetime.now().isoformat()
            return trace
    
    async def _run_from_checkpoint_impl(
        self,
        checkpoint: "ScalingCheckpoint",
        target_budget: int,
        policy: Optional[str] = None,
        use_synthesize: bool = True,
        benchmark: Optional[str] = None,
    ) -> AgentTrace:
        """
        Implementation for resuming execution from checkpoint (internal method)
        
        Core logic for Sequential Scaling:
        - Restore messages (already cleaned)
        - Restore token statistics
        - Continue execution with the new target_budget
        """
        start_time = datetime.now()
        
        # 1. Restore token statistics (key step)
        trace = AgentTrace(task_id=checkpoint.task_id, start_time=start_time.isoformat())
        trace.total_tokens = checkpoint.cumulative_tokens
        trace.total_prompt_tokens = checkpoint.prompt_tokens
        trace.total_output_tokens = checkpoint.output_tokens
        trace.total_steps = checkpoint.total_steps
        
        # 2. Restore messages (STOP_PROMPT already cleaned)
        # Need to properly handle tool_calls deserialization
        def restore_message(m: dict) -> Message:
            """Restore Message from dict, properly handling tool_calls"""
            tool_calls = None
            if m.get("tool_calls"):
                tool_calls = []
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict):
                        # Restore from OpenAI format
                        func = tc.get("function", {})
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        tool_calls.append(ToolCall(
                            name=func.get("name", ""),
                            arguments=args,
                            id=tc.get("id", "")
                        ))
                    else:
                        tool_calls.append(tc)
            
            return Message(
                role=m.get("role", "user"),
                content=m.get("content"),
                tool_calls=tool_calls,
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name")
            )
        
        messages = [restore_message(m) if isinstance(m, dict) else m for m in checkpoint.messages]
        trace.messages = messages.copy()
        
        # 3. Restore rounds records (provide defaults for missing fields)
        if checkpoint.rounds:
            restored_rounds = []
            for r in checkpoint.rounds:
                if isinstance(r, dict):
                    # Provide defaults for AgentRound required fields
                    round_data = {
                        "round_number": r.get("round_number", 0),
                        "reasoning": r.get("reasoning", ""),
                        "should_continue": r.get("should_continue", True),
                        "prompt_tokens": r.get("prompt_tokens", 0),
                        "output_tokens": r.get("output_tokens", 0),
                        "round_total_tokens": r.get("round_total_tokens", 0),
                        "cumulative_total_tokens": r.get("cumulative_total_tokens", 0),
                        "incremental_tokens": r.get("incremental_tokens", 0),
                        "tools_executed": r.get("tools_executed", 0),
                        "planned_tools": r.get("planned_tools", []),
                        "executions": r.get("executions", []),
                    }
                    restored_rounds.append(AgentRound(**round_data))
                else:
                    restored_rounds.append(r)
            trace.rounds = restored_rounds
        
        # 4. Set new target_budget
        old_budget = self.target_budget
        self.target_budget = target_budget
        
        # Reset scaling state for continuation
        self.reset_scaling_state()
        
        # Save instruction for later synthesize (extract from first user message)
        self._current_instruction = ""
        for msg in messages:
            if (isinstance(msg, Message) and msg.role == "user") or \
               (isinstance(msg, dict) and msg.get("role") == "user"):
                content = msg.content if isinstance(msg, Message) else msg.get("content", "")
                if content and not content.startswith("Before finalizing") and \
                   "CRITICAL" not in content:  # Exclude EXTEND/STOP prompts
                    self._current_instruction = content
                    break
        
        loguru_logger.info(
            f"[CHECKPOINT RESUME] task_id={checkpoint.task_id} "
            f"from_budget={checkpoint.budget_level} to_budget={target_budget} "
            f"restored_tokens={checkpoint.cumulative_tokens} restored_steps={checkpoint.total_steps}"
        )
        
        # 5. Get tools
        tools_schema = self.get_tools_schema()
        
        # 6. Continue execution (using Scaling mode logic)
        max_iterations = 100000  # Use large number for Scaling mode
        
        try:
            for step in range(max_iterations):
                timestamp = datetime.now().isoformat()
                
                # Call LLM
                messages_dict = [
                    m.to_dict() if isinstance(m, Message) else m 
                    for m in messages
                ]
                response = await self.llm.generate(messages_dict, tools=tools_schema)
                
                # Token statistics continue accumulating
                trace.total_prompt_tokens += response.prompt_tokens
                trace.total_output_tokens += response.output_tokens
                trace.total_tokens += response.total_tokens
                
                # Extract planned_tools from tool_calls
                planned_tools = [
                    {"tool": tc.name, "parameters": tc.arguments}
                    for tc in response.tool_calls
                ] if response.tool_calls else []
                
                # Create Round record
                current_round = AgentRound(
                    round_number=trace.total_steps + step + 1,
                    reasoning=response.content or "",
                    should_continue=bool(response.tool_calls),
                    prompt_tokens=response.prompt_tokens,
                    output_tokens=response.output_tokens,
                    round_total_tokens=response.total_tokens,
                    cumulative_total_tokens=trace.total_tokens,
                    incremental_tokens=response.total_tokens,
                    tools_executed=len(response.tool_calls) if response.tool_calls else 0,
                    planned_tools=planned_tools,
                    executions=[],
                )
                
                # Scaling decision
                scaling_action = self._decide_scaling_action(response, trace)
                
                loguru_logger.debug(
                    f"[CHECKPOINT] round={trace.total_steps + step + 1} "
                    f"prompt_tokens={response.prompt_tokens} budget={target_budget} "
                    f"has_tool_calls={bool(response.tool_calls)} action={scaling_action}"
                )
                
                if scaling_action == "EXTEND":
                    # Inject EXTEND prompt
                    self.extend_rounds.append(trace.total_steps + step + 1)
                    loguru_logger.info(
                        f"[EXTEND] round={trace.total_steps + step + 1} "
                        f"prompt_tokens={response.prompt_tokens} budget={target_budget}"
                    )
                    
                    current_round.should_continue = True
                    trace.rounds.append(current_round)
                    
                    assistant_msg = Message(role="assistant", content=response.content)
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    extend_msg = Message(role="user", content=self.scaling_config.extend_prompt)
                    messages.append(extend_msg)
                    trace.messages.append(extend_msg)
                    
                    trace.steps.append(AgentStep(
                        step=trace.total_steps + step,
                        timestamp=timestamp,
                        message_type="llm_response",
                        content=f"[EXTEND triggered] {response.content}",
                    ))
                    
                    continue
                
                elif scaling_action == "STOP":
                    # Inject STOP prompt
                    self.stop_rounds.append(trace.total_steps + step + 1)
                    loguru_logger.info(
                        f"[STOP] round={trace.total_steps + step + 1} "
                        f"prompt_tokens={response.prompt_tokens} budget={target_budget}"
                    )
                    
                    # Handle tool_calls
                    assistant_msg = Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls if response.tool_calls else None,
                    )
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    if response.tool_calls:
                        for tool_call in response.tool_calls:
                            try:
                                result, _ = await self.host.route_tool_call(
                                    tool_call.name, tool_call.arguments
                                )
                                result_text = self._extract_result_text(result, tool_call.name)
                            except Exception as e:
                                result_text = f"Error: {str(e)}"
                            
                            current_round.executions.append({
                                "tool": tool_call.name,
                                "success": "Error" not in result_text,
                                "parameters": tool_call.arguments,
                                "result": result_text[:500],
                                "planned_layer": None,
                            })
                            
                            tool_msg = Message(
                                role="tool",
                                content=result_text,
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                            messages.append(tool_msg)
                            trace.messages.append(tool_msg)
                    
                    trace.rounds.append(current_round)
                    
                    # Inject STOP prompt
                    stop_msg = Message(role="user", content=self.scaling_config.stop_prompt)
                    messages.append(stop_msg)
                    trace.messages.append(stop_msg)
                    
                    # Get final response
                    final_response = await self.llm.generate(
                        [m.to_dict() if isinstance(m, Message) else m for m in messages],
                        tools=tools_schema,
                    )
                    
                    trace.total_prompt_tokens += final_response.prompt_tokens
                    trace.total_output_tokens += final_response.output_tokens
                    trace.total_tokens += final_response.total_tokens
                    
                    final_msg = Message(role="assistant", content=final_response.content)
                    messages.append(final_msg)
                    trace.messages.append(final_msg)
                    
                    trace.final_response = final_response.content
                    trace.total_steps = checkpoint.total_steps + step + 2
                    break
                
                # Normal flow
                if response.tool_calls:
                    assistant_msg = Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    for tool_call in response.tool_calls:
                        trace.steps.append(AgentStep(
                            step=trace.total_steps + step,
                            timestamp=timestamp,
                            message_type="tool_call",
                            tool_name=tool_call.name,
                            tool_arguments=tool_call.arguments,
                        ))
                        
                        try:
                            result, _ = await self.host.route_tool_call(
                                tool_call.name, tool_call.arguments
                            )
                            result_text = self._extract_result_text(result, tool_call.name)
                            tool_error = False
                            success = True
                        except Exception as e:
                            result_text = f"Error: {str(e)}"
                            tool_error = True
                            success = False
                        
                        trace.steps.append(AgentStep(
                            step=trace.total_steps + step,
                            timestamp=datetime.now().isoformat(),
                            message_type="tool_result",
                            tool_name=tool_call.name,
                            tool_result=result_text,
                            tool_error=tool_error,
                        ))
                        
                        current_round.executions.append({
                            "tool": tool_call.name,
                            "success": success,
                            "parameters": tool_call.arguments,
                            "result": result_text[:500],
                            "planned_layer": None,
                        })
                        
                        tool_msg = Message(
                            role="tool",
                            content=result_text,
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                        messages.append(tool_msg)
                        trace.messages.append(tool_msg)
                    
                    # === SWEBench AgentFinishAction mechanism (when resuming from checkpoint) ===
                    if benchmark == "swebench":
                        swebench_finish_called = any(tc.name == "swebench_finish" for tc in response.tool_calls)
                        
                        if swebench_finish_called:
                            current_round.should_continue = False
                            trace.rounds.append(current_round)
                            
                            finish_message = None
                            for tc in response.tool_calls:
                                if tc.name == "swebench_finish":
                                    for msg in reversed(messages):
                                        if msg.role == "tool" and msg.name == tc.name:
                                            try:
                                                result = json.loads(msg.content)
                                                finish_message = result.get("message", "")
                                            except:
                                                finish_message = msg.content
                                            break
                                    break
                            
                            trace.final_response = finish_message or response.content or "Task completed via swebench_finish"
                            trace.total_steps = checkpoint.total_steps + step + 1
                            logger.info(f"[Agent] swebench_finish called (from checkpoint), terminating immediately")
                            break
                    
                    # === Search Domain <answer> tag detection mechanism (when resuming from checkpoint) ===
                    if benchmark == "search":
                        has_answer, extracted_answer = self._detect_answer_tag(response.content)
                        
                        if has_answer:
                            current_round.should_continue = False
                            trace.rounds.append(current_round)
                            
                            trace.final_response = extracted_answer or response.content
                            trace.total_steps = checkpoint.total_steps + step + 1
                            logger.info(f"[Agent] Search <answer> tag detected (from checkpoint), terminating immediately")
                            break
                    
                    trace.rounds.append(current_round)
                
                else:
                    # No tool_calls, final reply
                    current_round.should_continue = False
                    trace.rounds.append(current_round)
                    
                    assistant_msg = Message(role="assistant", content=response.content)
                    messages.append(assistant_msg)
                    trace.messages.append(assistant_msg)
                    
                    trace.steps.append(AgentStep(
                        step=trace.total_steps + step,
                        timestamp=timestamp,
                        message_type="llm_response",
                        content=response.content,
                    ))
                    
                    if use_synthesize and self.accumulated_information and len(trace.rounds) > 1:
                        try:
                            synthesized = await self._synthesize_final_solution(
                                self._current_instruction,
                                len(trace.steps)
                            )
                            trace.final_response = synthesized
                        except Exception as synth_error:
                            logger.warning(f"Failed to synthesize: {synth_error}")
                            trace.final_response = response.content
                    else:
                        trace.final_response = response.content
                    
                    trace.total_steps = checkpoint.total_steps + step + 1
                    break
            
            else:
                trace.error = "Reached max iterations (from checkpoint)"
                trace.total_steps = checkpoint.total_steps + max_iterations
                
        except Exception as e:
            trace.error = str(e)
            logger.exception(f"Error resuming task {checkpoint.task_id} from checkpoint")
        
        finally:
            # Restore original budget
            self.target_budget = old_budget
        
        end_time = datetime.now()
        trace.end_time = end_time.isoformat()
        trace.duration = (end_time - start_time).total_seconds()
        
        return trace

    async def run_with_user_simulator(
        self,
        task_id: str,
        user_simulator,  # UserSimulator adapter, implements respond(agent_msg) -> user_response
        policy: Optional[str] = None,
        first_agent_message: str = "Hi! How can I help you today?",
    ) -> AgentTrace:
        """
        Run a multi-turn conversation task (interacting with User Simulator)
        
        The flow is exactly consistent with the tau2-bench Orchestrator:
        1. Agent sends greeting "Hi! How can I help you today?"
        2. User Simulator replies based on scenario
        3. Agent processes user request (may call tools)
        4. Agent replies to user
        5. Repeat 2-4 until User sends STOP signal or max_steps is reached
        
        Note: Only uses max_steps as global limit (consistent with tau2-bench),
        no separate max_turns limit.
            
        Args:
            task_id: Task ID
            user_simulator: User simulator adapter
            policy: Optional policy
            first_agent_message: Agent's first greeting message
            
        Returns:
            AgentTrace: Execution trace
        """
        try:
            return await asyncio.wait_for(
                self._run_with_user_simulator_impl(
                    task_id, user_simulator, policy, first_agent_message
                ),
                timeout=self.task_timeout
            )
        except asyncio.TimeoutError:
            trace = AgentTrace(
                task_id=task_id,
                start_time=datetime.now().isoformat(),
                error=f"Task timeout after {self.task_timeout} seconds",
            )
            trace.end_time = datetime.now().isoformat()
            return trace
    
    async def _run_with_user_simulator_impl(
        self,
        task_id: str,
        user_simulator,
        policy: Optional[str] = None,
        first_agent_message: str = "Hi! How can I help you today?",
    ) -> AgentTrace:
        """
        Actual implementation for running multi-turn conversation task (internal method, with timeout protection)
        
        Supports Sequential Scaling for tau2-bench:
        - In Scaling mode (target_budget is set): ignores max_steps, uses token budget
        - In Normal mode (target_budget is None): uses max_steps as before
        - EXTEND/STOP prompts use role="user" to match User Simulator format
        """
        start_time = datetime.now()
        trace = AgentTrace(task_id=task_id, start_time=start_time.isoformat())
        
        # Reset scaling state for new task
        self.reset_scaling_state()
        
        # Determine mode
        is_scaling_mode = self.target_budget is not None
        max_iterations = 100000 if is_scaling_mode else self.max_steps
        
        loguru_logger.info(f"[AGENT] Starting user simulation for task: {task_id}")
        loguru_logger.info(f"[AGENT] Max steps: {self.max_steps}, Scaling mode: {is_scaling_mode}")
        if is_scaling_mode:
            loguru_logger.info(f"[AGENT] Token budget: {self.target_budget}")
        
        # Build system prompt (using Universal System Prompt + Policy)
        if policy:
            full_system_prompt = f"""{self.system_prompt}

## Policy

{policy}"""
        else:
            full_system_prompt = self.system_prompt
        
        # Initialize message list (only system prompt)
        messages = [
            Message(role="system", content=full_system_prompt),
        ]
        trace.messages = [Message(role="system", content=full_system_prompt)]
        
        # Get tools
        tools_schema = self.get_tools_schema()
        step_count = 0
        
        try:
            # === Step 0: Agent sends greeting ===
            agent_response = first_agent_message
            assistant_msg = Message(role="assistant", content=agent_response)
            messages.append(assistant_msg)
            trace.messages.append(assistant_msg)
            
            loguru_logger.info(f"[AGENT] Initial greeting: {agent_response[:100]}...")
            
            trace.steps.append(AgentStep(
                step=step_count,
                timestamp=datetime.now().isoformat(),
                message_type="llm_response",
                content=agent_response,
            ))
            
            # === Conversation loop ===
            while step_count < max_iterations:
                # === User Simulator reply ===
                # Prefer respond_async (supports MCP user tool calls)
                if hasattr(user_simulator, 'respond_async'):
                    user_response = await user_simulator.respond_async(agent_response)
                elif asyncio.iscoroutinefunction(user_simulator.respond):
                    user_response = await user_simulator.respond(agent_response)
                else:
                    user_response = user_simulator.respond(agent_response)
                
                # Record user message
                loguru_logger.info(f"[AGENT] User response (step {step_count}): {str(user_response)[:150]}...")
                trace.steps.append(AgentStep(
                    step=step_count,
                    timestamp=datetime.now().isoformat(),
                    message_type="user_message",
                    content=user_response,
                ))
                step_count += 1
                
                # Check if finished
                is_stop = False
                if user_response is None:
                    is_stop = True
                elif hasattr(user_simulator, 'is_stop_signal'):
                    is_stop = user_simulator.is_stop_signal(user_response)
                elif "[STOP]" in str(user_response):
                    is_stop = True
                
                if is_stop:
                    trace.final_response = agent_response
                    trace.total_steps = step_count
                    # Record end time and duration
                    end_time = datetime.now()
                    trace.end_time = end_time.isoformat()
                    trace.duration = (end_time - start_time).total_seconds()
                    return trace
                
                # Add user message to conversation (including tool_calls executed by User Simulator)
                user_tool_calls = None
                if hasattr(user_simulator, 'get_last_tool_calls'):
                    user_tool_calls = user_simulator.get_last_tool_calls() or None
                    if user_tool_calls:
                        loguru_logger.info(f"[AGENT] User tool_calls: {len(user_tool_calls)} tools")
                        for tc in user_tool_calls:
                            loguru_logger.info(f"[AGENT]   - {tc.name}({tc.arguments})")
                user_msg = Message(role="user", content=user_response, tool_calls=user_tool_calls)
                messages.append(user_msg)
                trace.messages.append(user_msg)
                
                # Add corresponding ToolMessages for User tool_calls (tau2 evaluator needs these to replay state)
                # Note: Only added to trace, not to messages
                # Because Claude/Anthropic models don't support user-initiated tool_calls
                # user_response in messages already contains a summary of tool execution results
                if hasattr(user_simulator, 'get_last_tool_results'):
                    user_tool_results = user_simulator.get_last_tool_results()
                    if user_tool_results:
                        for tr in user_tool_results:
                            tool_msg = Message(
                                role="tool",
                                content=tr["result"],
                                tool_call_id=tr["id"],
                                name=tr["name"],
                            )
                            # Only add to trace for evaluation, not sent to LLM
                            trace.messages.append(tool_msg)
                
                # === Agent reply loop (may include multiple tool calls) ===
                while step_count < max_iterations:
                    timestamp = datetime.now().isoformat()
                    
                    messages_dict = [m.to_dict() for m in messages]
                    
                    # Call LLM
                    response = await self.llm.generate(messages_dict, tools=tools_schema)
                    
                    # Update token statistics for scaling decisions
                    trace.total_prompt_tokens += response.prompt_tokens
                    trace.total_output_tokens += response.output_tokens
                    trace.total_tokens += response.total_tokens
                    
                    # Create round for tracking (needed for scaling decision)
                    current_round = AgentRound(
                        round_number=step_count + 1,
                        reasoning=response.content or "",
                        should_continue=bool(response.tool_calls),
                        prompt_tokens=response.prompt_tokens,
                        output_tokens=response.output_tokens,
                        round_total_tokens=response.total_tokens,
                        cumulative_total_tokens=trace.total_tokens,
                        incremental_tokens=response.total_tokens,
                        tools_executed=len(response.tool_calls) if response.tool_calls else 0,
                        planned_tools=[],
                        executions=[],
                    )
                    trace.rounds.append(current_round)
                    
                    # === Scaling Decision (only in Scaling mode) ===
                    scaling_action = self._decide_scaling_action(response, trace) if is_scaling_mode else "NONE"
                    
                    if scaling_action == "EXTEND":
                        # Model wants to stop (no tool_calls), but budget is sufficient
                        # → Inject EXTEND prompt as "user" message
                        self.extend_rounds.append(step_count + 1)
                        loguru_logger.info(
                            f"[EXTEND] step={step_count+1} tokens={trace.total_tokens} "
                            f"budget={self.target_budget} remaining={self.target_budget - trace.total_tokens}"
                        )
                        
                        # Add assistant response (no tool_calls)
                        assistant_msg = Message(role="assistant", content=response.content)
                        messages.append(assistant_msg)
                        trace.messages.append(assistant_msg)
                        
                        trace.steps.append(AgentStep(
                            step=step_count,
                            timestamp=timestamp,
                            message_type="llm_response",
                            content=f"[EXTEND triggered] {response.content}",
                        ))
                        
                        # Inject EXTEND prompt as user message (matching User Simulator format)
                        extend_msg = Message(role="user", content=self.scaling_config.extend_prompt)
                        messages.append(extend_msg)
                        trace.messages.append(extend_msg)
                        
                        step_count += 1
                        continue  # Continue inner loop for Agent to respond
                    
                    elif scaling_action == "STOP":
                        # Budget exceeded → inject STOP prompt, let model generate one more round
                        # If that round has tool_calls, execute them then stop
                        # If no tool_calls, just stop
                        self.stop_rounds.append(step_count + 1)
                        loguru_logger.info(
                            f"[STOP] step={step_count+1} prompt_tokens={response.prompt_tokens} "
                            f"budget={self.target_budget}"
                        )
                        
                        # Add assistant message (may or may not have tool_calls)
                        assistant_msg = Message(
                            role="assistant",
                            content=response.content,
                            tool_calls=response.tool_calls if response.tool_calls else None,
                        )
                        messages.append(assistant_msg)
                        trace.messages.append(assistant_msg)
                        
                        # Execute tool calls if any
                        if response.tool_calls:
                            for tool_call in response.tool_calls:
                                loguru_logger.info(f"[STOP] Executing tool: {tool_call.name}")
                                try:
                                    result, _ = await self.host.route_tool_call(
                                        tool_call.name,
                                        tool_call.arguments
                                    )
                                    result_text = self._extract_result_text(result, tool_call.name)
                                except Exception as e:
                                    result_text = f"Error: {str(e)}"
                                
                                tool_msg = Message(
                                    role="tool",
                                    content=result_text,
                                    tool_call_id=tool_call.id,
                                    name=tool_call.name,
                                )
                                messages.append(tool_msg)
                                trace.messages.append(tool_msg)
                        
                        # Inject STOP prompt as user message
                        stop_msg = Message(role="user", content=self.scaling_config.stop_prompt)
                        messages.append(stop_msg)
                        trace.messages.append(stop_msg)
                        
                        # Get final response
                        final_response = await self.llm.generate(
                            [m.to_dict() for m in messages],
                            tools=tools_schema,
                        )
                        
                        # Update token statistics
                        trace.total_prompt_tokens += final_response.prompt_tokens
                        trace.total_output_tokens += final_response.output_tokens
                        trace.total_tokens += final_response.total_tokens
                        
                        final_msg = Message(role="assistant", content=final_response.content)
                        messages.append(final_msg)
                        trace.messages.append(final_msg)
                        
                        trace.steps.append(AgentStep(
                            step=step_count + 1,
                            timestamp=datetime.now().isoformat(),
                            message_type="llm_response",
                            content=f"[STOP final] {final_response.content}",
                        ))
                        
                        # If STOP response still has tool_calls, execute them then stop
                        if final_response.tool_calls:
                            for tool_call in final_response.tool_calls:
                                try:
                                    result, _ = await self.host.route_tool_call(
                                        tool_call.name,
                                        tool_call.arguments
                                    )
                                    result_text = self._extract_result_text(result, tool_call.name)
                                except Exception as e:
                                    result_text = f"Error: {str(e)}"
                                
                                tool_msg = Message(
                                    role="tool",
                                    content=result_text,
                                    tool_call_id=tool_call.id,
                                    name=tool_call.name,
                                )
                                messages.append(tool_msg)
                                trace.messages.append(tool_msg)
                        
                        trace.final_response = final_response.content
                        trace.total_steps = step_count + 2
                        
                        # Exit completely (skip User Simulator)
                        end_time = datetime.now()
                        trace.end_time = end_time.isoformat()
                        trace.duration = (end_time - start_time).total_seconds()
                        return trace
                    
                    # === Normal flow ===
                    if response.tool_calls:
                        # Handle tool calls
                        assistant_msg = Message(
                            role="assistant",
                            content=response.content,
                            tool_calls=response.tool_calls,
                        )
                        messages.append(assistant_msg)
                        trace.messages.append(assistant_msg)
                        
                        for tool_call in response.tool_calls:
                            loguru_logger.info(f"[AGENT] Assistant tool_call: {tool_call.name}({tool_call.arguments})")
                            trace.steps.append(AgentStep(
                                step=step_count,
                                timestamp=timestamp,
                                message_type="tool_call",
                                tool_name=tool_call.name,
                                tool_arguments=tool_call.arguments,
                            ))
                            
                            try:
                                result, _ = await self.host.route_tool_call(
                                    tool_call.name,
                                    tool_call.arguments
                                )
                                # Get raw result (for trace) and truncated result (for LLM)
                                raw_result_text = self._extract_result_text(result, tool_call.name, skip_truncate=True)
                                result_text = self._extract_result_text(result, tool_call.name)
                                tool_error = False
                                loguru_logger.info(f"[AGENT] Tool result: {result_text[:200]}...")
                            except Exception as e:
                                result_text = f"Error: {str(e)}"
                                raw_result_text = result_text
                                tool_error = True
                                loguru_logger.warning(f"[AGENT] Tool error: {result_text}")
                            
                            # Save full result to trace (for evaluation)
                            trace.steps.append(AgentStep(
                                step=step_count,
                                timestamp=datetime.now().isoformat(),
                                message_type="tool_result",
                                tool_name=tool_call.name,
                                tool_result=raw_result_text,  # Use full untruncated result
                                tool_error=tool_error,
                            ))
                            
                            # Save full result to trace.messages (for evaluation parsing)
                            trace_tool_msg = Message(
                                role="tool",
                                content=raw_result_text,  # Untruncated
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                            trace.messages.append(trace_tool_msg)
                            
                            # Send truncated result to LLM
                            llm_tool_msg = Message(
                                role="tool",
                                content=result_text,  # Truncated version
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                            messages.append(llm_tool_msg)
                            step_count += 1
                    
                    else:
                        # Agent sends text response to user
                        agent_response = response.content
                        assistant_msg = Message(role="assistant", content=agent_response)
                        messages.append(assistant_msg)
                        trace.messages.append(assistant_msg)
                        
                        trace.steps.append(AgentStep(
                            step=step_count,
                            timestamp=timestamp,
                            message_type="llm_response",
                            content=agent_response,
                        ))
                        step_count += 1
                        
                        # Break out of tool call loop, proceed to next User reply
                        break
                
                else:
                    # Inner while loop ended normally (step_count >= max_iterations), reached max steps
                    if is_scaling_mode:
                        trace.error = f"Reached max iterations (scaling mode)"
                    else:
                        trace.error = f"Reached max steps ({self.max_steps})"
                    break
            
            # Outer while loop ended normally
            if trace.error is None and step_count >= max_iterations:
                if is_scaling_mode:
                    trace.error = f"Reached max iterations (scaling mode)"
                else:
                    trace.error = f"Reached max steps ({self.max_steps})"
            
            trace.total_steps = step_count
            
        except Exception as e:
            trace.error = str(e)
            trace.total_steps = step_count
            logger.exception(f"Error running task {task_id}")
        
        # Record end time and duration
        end_time = datetime.now()
        trace.end_time = end_time.isoformat()
        trace.duration = (end_time - start_time).total_seconds()
        
        return trace
    
    async def run_from_checkpoint_with_user_simulator(
        self,
        checkpoint: "ScalingCheckpoint",
        target_budget: int,
        user_simulator,
        policy: Optional[str] = None,
        first_agent_message: str = "Hi! How can I help you today?",
    ) -> AgentTrace:
        """
        Resume execution from checkpoint (with User Simulator, for tau2-bench)
        
        Sequential Scaling tau2-bench version:
        1. Restore messages (STOP_PROMPT already cleaned)
        2. Restore token statistics
        3. Restore User Simulator state (via replay_message_history)
        4. Continue execution until reaching new target_budget or User sends STOP
        
        Args:
            checkpoint: Cleaned ScalingCheckpoint
            target_budget: New token budget
            user_simulator: User simulator adapter
            policy: Optional policy
            first_agent_message: For logging only, will not be resent
            
        Returns:
            AgentTrace: Contains complete execution trace
        """
        try:
            return await asyncio.wait_for(
                self._run_from_checkpoint_with_user_simulator_impl(
                    checkpoint, target_budget, user_simulator, policy
                ),
                timeout=self.task_timeout
            )
        except asyncio.TimeoutError:
            trace = AgentTrace(
                task_id=checkpoint.task_id,
                start_time=datetime.now().isoformat(),
                error=f"Task timeout after {self.task_timeout} seconds (from checkpoint with user simulator)",
            )
            trace.end_time = datetime.now().isoformat()
            return trace
    
    async def _run_from_checkpoint_with_user_simulator_impl(
        self,
        checkpoint: "ScalingCheckpoint",
        target_budget: int,
        user_simulator,
        policy: Optional[str] = None,
    ) -> AgentTrace:
        """
        Implementation of resuming User Simulator execution from checkpoint
        
        Key steps:
        1. Restore Agent state (messages, tokens)
        2. Call user_simulator.replay_message_history() to restore User state
        3. Continue normal User Simulator interaction loop
        """
        start_time = datetime.now()
        
        # 1. Restore token statistics
        trace = AgentTrace(task_id=checkpoint.task_id, start_time=start_time.isoformat())
        trace.total_tokens = checkpoint.cumulative_tokens
        trace.total_prompt_tokens = checkpoint.prompt_tokens
        trace.total_output_tokens = checkpoint.output_tokens
        trace.total_steps = checkpoint.total_steps
        
        # 2. Restore messages (need to properly handle tool_calls deserialization)
        def restore_message(m: dict) -> Message:
            """Restore Message from dict, properly handling tool_calls"""
            tool_calls = None
            if m.get("tool_calls"):
                tool_calls = []
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict):
                        # Restore from OpenAI format
                        func = tc.get("function", {})
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        tool_calls.append(ToolCall(
                            name=func.get("name", ""),
                            arguments=args,
                            id=tc.get("id", "")
                        ))
                    else:
                        tool_calls.append(tc)
            
            return Message(
                role=m.get("role", "user"),
                content=m.get("content"),
                tool_calls=tool_calls,
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name")
            )
        
        messages = [restore_message(m) if isinstance(m, dict) else m for m in checkpoint.messages]
        trace.messages = messages.copy()
        
        # 3. Restore rounds (provide defaults for missing fields)
        if checkpoint.rounds:
            restored_rounds = []
            for r in checkpoint.rounds:
                if isinstance(r, dict):
                    # Provide defaults for AgentRound required fields
                    round_data = {
                        "round_number": r.get("round_number", 0),
                        "reasoning": r.get("reasoning", ""),
                        "should_continue": r.get("should_continue", True),
                        "prompt_tokens": r.get("prompt_tokens", 0),
                        "output_tokens": r.get("output_tokens", 0),
                        "round_total_tokens": r.get("round_total_tokens", 0),
                        "cumulative_total_tokens": r.get("cumulative_total_tokens", 0),
                        "incremental_tokens": r.get("incremental_tokens", 0),
                        "tools_executed": r.get("tools_executed", 0),
                        "planned_tools": r.get("planned_tools", []),
                        "executions": r.get("executions", []),
                    }
                    restored_rounds.append(AgentRound(**round_data))
                else:
                    restored_rounds.append(r)
            trace.rounds = restored_rounds
        
        # 4. Set new target_budget
        old_budget = self.target_budget
        self.target_budget = target_budget
        is_scaling_mode = True
        
        self.reset_scaling_state()
        
        loguru_logger.info(
            f"[CHECKPOINT RESUME WITH USER SIM] task_id={checkpoint.task_id} "
            f"from_budget={checkpoint.budget_level} to_budget={target_budget}"
        )
        
        # 5. Restore User Simulator state
        if hasattr(user_simulator, 'replay_message_history'):
            try:
                # Convert messages to dict format
                messages_dict = [
                    m.to_dict() if isinstance(m, Message) else m 
                    for m in messages
                ]
                await user_simulator.replay_message_history(messages_dict)
                loguru_logger.info("[CHECKPOINT] User Simulator state restored via replay_message_history")
            except Exception as e:
                loguru_logger.warning(f"[CHECKPOINT] Failed to restore User Simulator state: {e}")
        
        # 6. Get last assistant message as starting point
        agent_response = ""
        for msg in reversed(messages):
            if (isinstance(msg, Message) and msg.role == "assistant") or \
               (isinstance(msg, dict) and msg.get("role") == "assistant"):
                content = msg.content if isinstance(msg, Message) else msg.get("content", "")
                if content:
                    agent_response = content
                    break
        
        # 7. Get tools
        tools_schema = self.get_tools_schema()
        step_count = checkpoint.total_steps
        max_iterations = 100000
        
        try:
            # === Conversation loop ===
            while step_count < max_iterations:
                # User Simulator response
                if hasattr(user_simulator, 'respond_async'):
                    user_response = await user_simulator.respond_async(agent_response)
                elif asyncio.iscoroutinefunction(user_simulator.respond):
                    user_response = await user_simulator.respond(agent_response)
                else:
                    user_response = user_simulator.respond(agent_response)
                
                loguru_logger.info(f"[CHECKPOINT] User response (step {step_count}): {str(user_response)[:150]}...")
                trace.steps.append(AgentStep(
                    step=step_count,
                    timestamp=datetime.now().isoformat(),
                    message_type="user_message",
                    content=user_response,
                ))
                step_count += 1
                
                # Check if finished
                is_stop = False
                if user_response is None:
                    is_stop = True
                elif hasattr(user_simulator, 'is_stop_signal'):
                    is_stop = user_simulator.is_stop_signal(user_response)
                elif "[STOP]" in str(user_response):
                    is_stop = True
                
                if is_stop:
                    trace.final_response = agent_response
                    trace.total_steps = step_count
                    end_time = datetime.now()
                    trace.end_time = end_time.isoformat()
                    trace.duration = (end_time - start_time).total_seconds()
                    self.target_budget = old_budget
                    return trace
                
                # Add user message
                user_tool_calls = None
                if hasattr(user_simulator, 'get_last_tool_calls'):
                    user_tool_calls = user_simulator.get_last_tool_calls() or None
                
                user_msg = Message(role="user", content=user_response, tool_calls=user_tool_calls)
                messages.append(user_msg)
                trace.messages.append(user_msg)
                
                # Add User tool_results to trace
                if hasattr(user_simulator, 'get_last_tool_results'):
                    user_tool_results = user_simulator.get_last_tool_results()
                    if user_tool_results:
                        for tr in user_tool_results:
                            tool_msg = Message(
                                role="tool",
                                content=tr["result"],
                                tool_call_id=tr["id"],
                                name=tr["name"],
                            )
                            trace.messages.append(tool_msg)
                
                # === Agent response loop ===
                while step_count < max_iterations:
                    timestamp = datetime.now().isoformat()
                    messages_dict = [
                        m.to_dict() if isinstance(m, Message) else m 
                        for m in messages
                    ]
                    
                    response = await self.llm.generate(messages_dict, tools=tools_schema)
                    
                    trace.total_prompt_tokens += response.prompt_tokens
                    trace.total_output_tokens += response.output_tokens
                    trace.total_tokens += response.total_tokens
                    
                    current_round = AgentRound(
                        round_number=step_count + 1,
                        reasoning=response.content or "",
                        should_continue=bool(response.tool_calls),
                        prompt_tokens=response.prompt_tokens,
                        output_tokens=response.output_tokens,
                        round_total_tokens=response.total_tokens,
                        cumulative_total_tokens=trace.total_tokens,
                        incremental_tokens=response.total_tokens,
                        tools_executed=len(response.tool_calls) if response.tool_calls else 0,
                        planned_tools=[],
                        executions=[],
                    )
                    trace.rounds.append(current_round)
                    
                    # Scaling decision
                    scaling_action = self._decide_scaling_action(response, trace)
                    
                    if scaling_action == "EXTEND":
                        self.extend_rounds.append(step_count + 1)
                        loguru_logger.info(f"[EXTEND] step={step_count+1} prompt_tokens={response.prompt_tokens}")
                        
                        assistant_msg = Message(role="assistant", content=response.content)
                        messages.append(assistant_msg)
                        trace.messages.append(assistant_msg)
                        
                        extend_msg = Message(role="user", content=self.scaling_config.extend_prompt)
                        messages.append(extend_msg)
                        trace.messages.append(extend_msg)
                        
                        step_count += 1
                        continue
                    
                    elif scaling_action == "STOP":
                        self.stop_rounds.append(step_count + 1)
                        loguru_logger.info(f"[STOP] step={step_count+1} prompt_tokens={response.prompt_tokens}")
                        
                        assistant_msg = Message(
                            role="assistant",
                            content=response.content,
                            tool_calls=response.tool_calls if response.tool_calls else None,
                        )
                        messages.append(assistant_msg)
                        trace.messages.append(assistant_msg)
                        
                        if response.tool_calls:
                            for tool_call in response.tool_calls:
                                try:
                                    result, _ = await self.host.route_tool_call(
                                        tool_call.name, tool_call.arguments
                                    )
                                    result_text = self._extract_result_text(result, tool_call.name)
                                except Exception as e:
                                    result_text = f"Error: {str(e)}"
                                
                                tool_msg = Message(
                                    role="tool",
                                    content=result_text,
                                    tool_call_id=tool_call.id,
                                    name=tool_call.name,
                                )
                                messages.append(tool_msg)
                                trace.messages.append(tool_msg)
                        
                        stop_msg = Message(role="user", content=self.scaling_config.stop_prompt)
                        messages.append(stop_msg)
                        trace.messages.append(stop_msg)
                        
                        final_response = await self.llm.generate(
                            [m.to_dict() if isinstance(m, Message) else m for m in messages],
                            tools=tools_schema,
                        )
                        
                        trace.total_prompt_tokens += final_response.prompt_tokens
                        trace.total_output_tokens += final_response.output_tokens
                        trace.total_tokens += final_response.total_tokens
                        
                        final_msg = Message(role="assistant", content=final_response.content)
                        messages.append(final_msg)
                        trace.messages.append(final_msg)
                        
                        trace.final_response = final_response.content
                        trace.total_steps = step_count + 2
                        
                        end_time = datetime.now()
                        trace.end_time = end_time.isoformat()
                        trace.duration = (end_time - start_time).total_seconds()
                        self.target_budget = old_budget
                        return trace
                    
                    # Normal flow
                    if response.tool_calls:
                        assistant_msg = Message(
                            role="assistant",
                            content=response.content,
                            tool_calls=response.tool_calls,
                        )
                        messages.append(assistant_msg)
                        trace.messages.append(assistant_msg)
                        
                        for tool_call in response.tool_calls:
                            try:
                                result, _ = await self.host.route_tool_call(
                                    tool_call.name, tool_call.arguments
                                )
                                raw_result = self._extract_result_text(result, tool_call.name, skip_truncate=True)
                                result_text = self._extract_result_text(result, tool_call.name)
                            except Exception as e:
                                result_text = raw_result = f"Error: {str(e)}"
                            
                            trace.steps.append(AgentStep(
                                step=step_count,
                                timestamp=datetime.now().isoformat(),
                                message_type="tool_result",
                                tool_name=tool_call.name,
                                tool_result=raw_result,
                            ))
                            
                            trace_tool_msg = Message(
                                role="tool",
                                content=raw_result,
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                            trace.messages.append(trace_tool_msg)
                            
                            llm_tool_msg = Message(
                                role="tool",
                                content=result_text,
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                            messages.append(llm_tool_msg)
                            step_count += 1
                    
                    else:
                        agent_response = response.content
                        assistant_msg = Message(role="assistant", content=agent_response)
                        messages.append(assistant_msg)
                        trace.messages.append(assistant_msg)
                        
                        trace.steps.append(AgentStep(
                            step=step_count,
                            timestamp=timestamp,
                            message_type="llm_response",
                            content=agent_response,
                        ))
                        step_count += 1
                        break
                
                else:
                    trace.error = "Reached max iterations (from checkpoint with user simulator)"
                    break
            
            if trace.error is None and step_count >= max_iterations:
                trace.error = "Reached max iterations (from checkpoint with user simulator)"
            
            trace.total_steps = step_count
            
        except Exception as e:
            trace.error = str(e)
            trace.total_steps = step_count
            logger.exception(f"Error resuming task {checkpoint.task_id} from checkpoint with user simulator")
        
        finally:
            self.target_budget = old_budget
        
        end_time = datetime.now()
        trace.end_time = end_time.isoformat()
        trace.duration = (end_time - start_time).total_seconds()
        
        return trace
    
    def _extract_result_text(self, result: Any, tool_name: str = None, skip_truncate: bool = False) -> str:
        """Extract text from MCP result
        
        Args:
            result: MCP call result
            tool_name: Tool name, used to clean FastMCP-added error prefix
            skip_truncate: Whether to skip truncation (for saving full result to trace)
        """
        text = None
        is_error = False
        
        if hasattr(result, 'content') and result.content:
            content = result.content[0]
            if hasattr(content, 'text'):
                text = content.text
        elif isinstance(result, str):
            text = result
        else:
            try:
                text = json.dumps(result, default=str, ensure_ascii=False)
            except:
                text = str(result)
        
        # Clean FastMCP-added error prefix "Error executing tool {tool_name}: "
        # Convert to tau2 native format "Error: {message}"
        # mcp-bench tool name separator (Bedrock compatible)
        MCPBENCH_SEPARATOR = "__"
        if text and tool_name:
            # Extract "think" from "BioMCP__think" for matching error prefix
            if MCPBENCH_SEPARATOR in tool_name:
                base_tool_name = tool_name.split(MCPBENCH_SEPARATOR, 1)[1]
            else:
                base_tool_name = tool_name
            prefix = f"Error executing tool {base_tool_name}: "
            if text.startswith(prefix):
                text = text[len(prefix):]
                is_error = True
        
        # Check if it's an error message
        if text and (text.startswith("Error:") or text.startswith("error:")):
            is_error = True
        
        # Apply content truncation (mcp-bench configuration)
        # If skip_truncate=True, skip truncation (for saving full result to trace)
        if not skip_truncate:
            text = self._truncate_content(text or "", is_error=is_error)
        
        return text or ""
    
    def _truncate_content(self, content: str, is_error: bool = False) -> str:
        """
        Truncate content based on configuration
        
        Args:
            content: Original content
            is_error: Whether it's an error message
            
        Returns:
            Truncated content
        """
        if not content:
            return content
        
        if is_error:
            max_len = self.error_truncate_length
            if len(content) > max_len:
                # For errors, show prefix portion
                prefix_len = self.error_display_prefix
                return content[:prefix_len] + f"... [truncated, {len(content)} chars total]"
        else:
            max_len = self.content_truncate_length
            if len(content) > max_len:
                return content[:max_len] + f"... [truncated, {len(content)} chars total]"
        
        return content

    async def compress_accumulated_information(self, target_tokens: int = 3000) -> bool:
        """
        Use LLM to compress accumulated_information to reduce token usage
        
        Args:
            target_tokens: Target token count
            
        Returns:
            bool: Whether compression was successful
        """
        if not self.accumulated_information:
            logger.info("No accumulated information to compress")
            return False
        
        original_length = len(self.accumulated_information)
        original_tokens = original_length // 4
        
        if original_tokens <= target_tokens:
            logger.info(f"Accumulated information already within target ({original_tokens} <= {target_tokens} tokens)")
            return False
        
        logger.info(f"Starting LLM-based compression of accumulated_information: {original_tokens} tokens -> target {target_tokens} tokens")
        
        system_prompt = "You are an expert information summarizer. Your task is to compress execution history while preserving all critical information and findings."
        
        user_prompt = f"""Please compress the following execution history to approximately {target_tokens} tokens while preserving:
1. All key findings and results
2. Important tool execution outcomes
3. Critical information discovered
4. Task progress and context

EXECUTION HISTORY TO COMPRESS:
{self.accumulated_information}

COMPRESSED EXECUTION HISTORY:"""

        try:
            # Use LLM for compression
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt[:30000]},  # Limit input length
            ]
            response = await self.llm.generate(messages, tools=None)
            compressed_content = response.content or ""
            
            compressed_length = len(compressed_content)
            compressed_tokens = compressed_length // 4
            
            if compressed_tokens < original_tokens:
                self.accumulated_information = compressed_content.strip()
                logger.info(f"LLM compression successful: {original_tokens} -> {compressed_tokens} tokens ({((original_tokens - compressed_tokens) / original_tokens * 100):.1f}% reduction)")
                return True
            else:
                logger.warning("LLM compression did not reduce token count, falling back to rule-based compression")
                return self._fallback_rule_based_compression(target_tokens, original_tokens)
                
        except Exception as llm_error:
            logger.warning(f"LLM compression failed: {llm_error}, falling back to rule-based compression")
            return self._fallback_rule_based_compression(target_tokens, original_tokens)
    
    def _fallback_rule_based_compression(self, target_tokens: int, original_tokens: int) -> bool:
        """Rule-based compression fallback method"""
        try:
            rounds = self.accumulated_information.split("--- Summary of Round ")
            
            if len(rounds) <= 1:
                # Simple truncation compression
                target_chars = target_tokens * 4
                compressed_info = f"[Early execution history compressed for token limit]\n\n{self.accumulated_information[-target_chars:]}"
                self.accumulated_information = compressed_info
                
                new_length = len(self.accumulated_information)
                new_tokens = new_length // 4
                logger.info(f"Rule-based simple compression completed: {original_tokens} -> {new_tokens} tokens")
                return True
            
            # Smart round-based compression
            first_part = rounds[0]
            keep_recent_rounds = 2
            recent_rounds = rounds[-keep_recent_rounds:] if len(rounds) > keep_recent_rounds else rounds[1:]
            
            middle_rounds_count = len(rounds) - 1 - len(recent_rounds)
            if middle_rounds_count > 0:
                compressed_middle = f"[Rounds 2-{middle_rounds_count+1} compressed: Multiple tools executed successfully, information gathered and accumulated]"
            else:
                compressed_middle = ""
            
            compressed_parts = [first_part.strip()]
            if compressed_middle:
                compressed_parts.append(compressed_middle)
            
            for round_content in recent_rounds:
                if round_content.strip():
                    compressed_parts.append("--- Summary of Round " + round_content)
            
            self.accumulated_information = "\n\n".join(compressed_parts)
            
            new_length = len(self.accumulated_information)
            new_tokens = new_length // 4
            
            # If still too long, compress further
            if new_tokens > target_tokens:
                target_chars = target_tokens * 4
                if len(self.accumulated_information) > target_chars:
                    keep_start = target_chars // 2
                    keep_end = target_chars // 2
                    start_part = self.accumulated_information[:keep_start]
                    end_part = self.accumulated_information[-keep_end:]
                    self.accumulated_information = f"{start_part}\n\n[Middle content compressed for token limit]\n\n{end_part}"
                    
                    new_length = len(self.accumulated_information)
                    new_tokens = new_length // 4
            
            logger.info(f"Rule-based compression completed: {original_tokens} -> {new_tokens} tokens ({((original_tokens - new_tokens) / original_tokens * 100):.1f}% reduction)")
            return True
        except Exception as e:
            logger.warning(f"Rule-based compression failed: {e}")
            return False
    
    def _update_accumulated_information(self, round_num: int, round_info: str):
        """
        Update accumulated_information (uncompressed and possibly compressed versions)
        
        Args:
            round_num: Round number
            round_info: Summary of this round
        """
        round_summary = f"--- Summary of Round {round_num} ---\n{round_info}\n"
        
        # Update uncompressed version (always keep full history)
        self.accumulated_information_uncompressed += round_summary
        
        # Update possibly compressed version
        self.accumulated_information += round_summary
    
    def reset_accumulated_information(self):
        """Reset accumulated information (called at task start)"""
        self.accumulated_information = ""
        self.accumulated_information_uncompressed = ""
    
    def _build_round_summary(self, round_info: AgentRound) -> str:
        """
        Build summary for a single round
        
        Args:
            round_info: AgentRound object
            
        Returns:
            Summary string for this round
        """
        lines = []
        lines.append(f"Reasoning: {round_info.reasoning[:500] if round_info.reasoning else 'N/A'}")
        lines.append(f"Tools Executed: {round_info.tools_executed}")
        
        for exec_info in round_info.executions:
            tool_name = exec_info.get("tool", "unknown")
            success = "✓" if exec_info.get("success", False) else "✗"
            result = exec_info.get("result", "")[:200]
            lines.append(f"  - {tool_name} [{success}]: {result}")
        
        return "\n".join(lines)
    
    async def _synthesize_final_solution(self, task: str, total_executions: int) -> str:
        """
        Use LLM to synthesize all execution results into a final answer
        
        Args:
            task: Original task description
            total_executions: Total number of executions
            
        Returns:
            Synthesized final answer
        """
        logger.info("Synthesizing final solution from all rounds...")
        
        user_prompt = f"""You are an expert solution synthesizer for multi-tool AI agent execution.
ORIGINAL TASK: "{task}"
A multi-round execution process has completed with {total_executions} total tool calls.

ACCUMULATED INFORMATION AND RESULTS:
{self.accumulated_information}

Based on the original task and all the information gathered, provide a final, comprehensive, and well-structured answer that directly addresses the user's request.
Synthesize the key findings and present them in a clear, organized manner."""

        system_prompt = "You are an expert solution synthesizer. Combine information from tool executions into a cohesive, high-quality answer."

        max_retries = 3
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt[:30000]},  # Limit input length
                ]
                response = await self.llm.generate(messages, tools=None)
                solution = response.content or ""
                
                if solution:
                    return solution
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"ERROR in final solution attempt {attempt + 1}/{max_retries}: {e}")
                
                # Try compression and retry
                if "token" in error_msg.lower() and attempt < max_retries - 1:
                    logger.info("Token limit error in final solution, attempting compression...")
                    if await self.compress_accumulated_information():
                        # Rebuild prompt
                        user_prompt = f"""You are an expert solution synthesizer for multi-tool AI agent execution.
ORIGINAL TASK: "{task}"
A multi-round execution process has completed with {total_executions} total tool calls.

ACCUMULATED INFORMATION AND RESULTS:
{self.accumulated_information}

Based on the original task and all the information gathered, provide a final, comprehensive, and well-structured answer that directly addresses the user's request.
Synthesize the key findings and present them in a clear, organized manner."""
                        continue
                
                if attempt == max_retries - 1:
                    raise e
        
        return "Error: Failed to synthesize final solution"
