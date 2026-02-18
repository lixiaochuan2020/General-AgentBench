"""Bump Sort Evaluation Runner"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from ..host import BenchmarkHost
from ..llm_api import LiteLLMAPI
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


@dataclass
class ComparisonResult:
    """Single comparison result"""
    winner: int                  # 1 or 2
    trajectory_1_id: str         # Trajectory 1 identifier (pass_N)
    trajectory_2_id: str         # Trajectory 2 identifier (pass_N)
    raw_response: str            # Full LLM response
    prompt_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class BumpSortResult:
    """Bump Sort final result"""
    task_id: str
    final_winner_pass: str       # Final winning pass (e.g., "pass_3")
    final_winner_score: float    # Ground truth score of the winning trajectory
    num_comparisons: int         # Number of comparisons
    comparison_history: list     # History of all comparisons
    best_at_k: dict              # best@k evaluation results: {"k=1": 0/1, "k=2": 0/1, ...}
    skipped: bool = False        # Whether skipped (all passes have score 0)
    skip_reason: str = None


class BumpSortRunner:
    """
    Bump Sort Evaluation Runner
    
    Finds the optimal trajectory through pairwise comparison (bump sorting).
    """
    
    def __init__(
        self,
        host: BenchmarkHost,
        llm_model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        required_servers: list[str] = None,
        distraction_count: int = -1,  # -1 means load all distraction tools
        compress_tools: bool = False,  # Whether to compress tool definitions to reduce token count
        minimal_tools: bool = False,   # Minimal mode - only keep name, description, param names
    ):
        self.host = host
        self.required_servers = required_servers
        self.distraction_count = distraction_count
        self.compress_tools = compress_tools
        self.minimal_tools = minimal_tools
        self.llm = LiteLLMAPI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def _format_tools_schema(self, tools_schema: list[dict]) -> str:
        """Convert tools schema to JSON string"""
        return json.dumps(tools_schema, ensure_ascii=False, indent=2)
    
    def _format_trajectory(
        self,
        messages: list[dict],
        compress: bool = False,
        max_tool_output_len: int = 500,
        skip_task_messages: bool = True,
    ) -> str:
        """
        Convert trajectory messages to string
        
        Args:
            messages: List of messages
            compress: Whether to compress (truncate tool output, use compact format)
            max_tool_output_len: Max length of tool output in compress mode
            skip_task_messages: Whether to skip the first system and first user message
                              (these are already included as task_description in the prompt)
        """
        # If skipping task messages, filter first
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
            
            messages = filtered_messages
        
        if not compress:
            return json.dumps(messages, ensure_ascii=False, indent=2)
        
        # Compress mode: use more compact text format
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                lines.append(f"[USER] {content[:500]}..." if len(str(content)) > 500 else f"[USER] {content}")
            
            elif role == "assistant":
                # Handle tool_calls
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "")
                        if isinstance(args, str) and len(args) > 200:
                            args = args[:200] + "..."
                        lines.append(f"[CALL] {name}({args})")
                elif content:
                    lines.append(f"[ASSISTANT] {content[:300]}..." if len(str(content)) > 300 else f"[ASSISTANT] {content}")
            
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "")[:10]
                if isinstance(content, str):
                    if len(content) > max_tool_output_len:
                        content = content[:max_tool_output_len] + f"... [truncated, total {len(content)} chars]"
                    lines.append(f"[TOOL:{tool_id}] {content}")
                else:
                    content_str = json.dumps(content, ensure_ascii=False)
                    if len(content_str) > max_tool_output_len:
                        content_str = content_str[:max_tool_output_len] + f"... [truncated]"
                    lines.append(f"[TOOL:{tool_id}] {content_str}")
        
        return "\n".join(lines)
    
    def _extract_task_description(self, trace_data: dict, max_len: int = 0) -> str:
        """
        Extract task description from trajectory messages
        
        Rule: Concatenate the first role=system content + first role=user content
        
        Args:
            trace_data: Trace data
            max_len: Max character count, 0 means no limit
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
            result = "Task description not found"
        
        # Truncate (if needed)
        if max_len > 0 and len(result) > max_len:
            result = result[:max_len] + "\n\n... [task description truncated]"
        
        return result
    
    def _extract_final_answer(self, trace_data: dict) -> str:
        """Extract final answer from trajectory"""
        trace = trace_data.get("trace", {})
        
        # Prefer the final_response field
        if trace.get("final_response"):
            return trace["final_response"]
        
        # Otherwise use the last assistant message
        messages = trace.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        
        return "Final answer not found"
    
    def _extract_ranking(self, response: str) -> int:
        """Extract ranking from LLM response"""
        # Match <ranking>1</ranking> or <ranking>2</ranking>
        pattern = r"<ranking>\s*([12])\s*</ranking>"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # If no tag found, try extracting from the end
        last_lines = response.strip().split('\n')[-3:]
        for line in reversed(last_lines):
            if '1' in line and '2' not in line:
                return 1
            elif '2' in line and '1' not in line:
                return 2
        
        logger.warning("Could not extract ranking, defaulting to 1")
        return 1
    
    async def compare_two_trajectories(
        self,
        trace_data_1: dict,
        trace_data_2: dict,
        pass_id_1: str,
        pass_id_2: str,
    ) -> ComparisonResult:
        """
        Compare two trajectories and return the winner
        """
        # Get tools description (includes all distraction tools, optionally compressed/minimal/text mode)
        if self.minimal_tools:
            # Use plain text format (most token-efficient)
            if self.required_servers:
                tools_desc = self.host.get_filtered_tools_text(
                    required_client_names=self.required_servers,
                    distraction_count=self.distraction_count,  # -1 = all
                    max_description_len=50,
                )
            else:
                tools_desc = self.host.get_tools_text(max_description_len=50)
        else:
            # Use JSON format
            if self.required_servers:
                tools_schema = self.host.get_filtered_tools_schema(
                    required_client_names=self.required_servers,
                    distraction_count=self.distraction_count,  # -1 = all
                    compress=self.compress_tools,
                    minimal=False,
                )
            else:
                tools_schema = self.host.get_tools_schema(
                    compress=self.compress_tools,
                    minimal=False,
                )
            tools_desc = self._format_tools_schema(tools_schema)
        
        # Extract info (truncate task_description in minimal mode)
        max_task_desc_len = 8000 if self.minimal_tools else 0  # ~2000 tokens
        task_desc = self._extract_task_description(trace_data_1, max_len=max_task_desc_len)
        
        # Also compress trajectory in minimal mode
        compress_trajectory = self.minimal_tools
        trajectory_1 = self._format_trajectory(
            trace_data_1.get("trace", {}).get("messages", []),
            compress=compress_trajectory,
            max_tool_output_len=500,
            skip_task_messages=True,  # Skip the first system and user message
        )
        trajectory_2 = self._format_trajectory(
            trace_data_2.get("trace", {}).get("messages", []),
            compress=compress_trajectory,
            max_tool_output_len=500,
            skip_task_messages=True,  # Skip the first system and user message
        )
        
        # Build prompt (no longer includes final_answer)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            tools_description=tools_desc,
            task_description=task_desc,
            trajectory_1=trajectory_1,
            trajectory_2=trajectory_2,
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        response = await self.llm.generate(messages, tools=None)
        raw_response = response.content or ""
        
        # Extract ranking
        winner = self._extract_ranking(raw_response)
        
        return ComparisonResult(
            winner=winner,
            trajectory_1_id=pass_id_1,
            trajectory_2_id=pass_id_2,
            raw_response=raw_response,
            prompt_tokens=response.prompt_tokens if response else 0,
            output_tokens=response.output_tokens if response else 0,
            total_tokens=response.total_tokens if response else 0,
        )
    
    async def bump_sort(
        self,
        task_id: str,
        passes_data: list[tuple[str, dict, dict]],  # [(pass_id, trace_data, eval_data), ...]
        benchmark: str,
    ) -> BumpSortResult:
        """
        Execute Bump Sort to find the optimal trajectory
        
        Args:
            task_id: Task ID
            passes_data: [(pass_id, trace_data, eval_data), ...] list
            benchmark: Benchmark type ("search" or "tau2bench")
            
        Returns:
            BumpSortResult: Final result
        """
        # Pre-compute raw scores for all passes
        all_scores = {}
        for pass_id, _, eval_data in passes_data:
            score = self._get_score(eval_data, benchmark)
            all_scores[pass_id] = score
        
        # Initialize best@k dictionary
        best_at_k = {}
        num_passes = len(passes_data)
        
        # If all scores are 0, skip comparison, all best@k are 0
        if all(score == 0.0 for score in all_scores.values()):
            for k in range(1, num_passes + 1):
                best_at_k[f"k={k}"] = 0.0
            return BumpSortResult(
                task_id=task_id,
                final_winner_pass=passes_data[0][0],  # Return the first pass
                final_winner_score=0.0,
                num_comparisons=0,
                comparison_history=[],
                best_at_k=best_at_k,
                skipped=True,
                skip_reason="All passes have score 0.0, skipping comparison",
            )
        
        # Helper function: compute best@k score
        def get_best_at_k_score(original_score: float) -> float:
            if benchmark == "mcpbench":
                # mcpbench: use raw score directly as best@k
                return original_score
            else:
                # Other benchmarks: raw score == 1.0 -> best@k = 1.0, otherwise 0.0
                return 1.0 if original_score == 1.0 else 0.0
        
        # Execute Bump Sort
        comparison_history = []
        current_winner_idx = 0  # Start from the first one
        
        # step0: best@1 = raw score of pass1
        first_pass_id = passes_data[0][0]
        best_at_k["k=1"] = get_best_at_k_score(all_scores[first_pass_id])
        logger.info(f"  best@1: {first_pass_id} original_score={all_scores[first_pass_id]} -> best@1={best_at_k['k=1']}")
        
        for i in range(1, len(passes_data)):
            pass_id_1, trace_data_1, _ = passes_data[current_winner_idx]
            pass_id_2, trace_data_2, _ = passes_data[i]
            
            logger.info(f"  Comparing {pass_id_1} vs {pass_id_2}...")
            
            result = await self.compare_two_trajectories(
                trace_data_1, trace_data_2, pass_id_1, pass_id_2
            )
            
            # Determine winner
            if result.winner == 2:
                current_winner_idx = i
                winner_pass = pass_id_2
            else:
                winner_pass = pass_id_1
            
            comparison_history.append({
                "round": i,
                "trajectory_1": pass_id_1,
                "trajectory_2": pass_id_2,
                "winner": result.winner,
                "winner_pass": winner_pass,
                "total_tokens": result.total_tokens,
            })
            
            # Compute best@(i+1): raw score of the winner
            k = i + 1
            best_at_k[f"k={k}"] = get_best_at_k_score(all_scores[winner_pass])
            logger.info(f"    Winner: {winner_pass}, original_score={all_scores[winner_pass]} -> best@{k}={best_at_k[f'k={k}']}")
        
        # Get the final winner's score
        final_pass_id, _, final_eval_data = passes_data[current_winner_idx]
        final_score = self._get_score(final_eval_data, benchmark)
        
        return BumpSortResult(
            task_id=task_id,
            final_winner_pass=final_pass_id,
            final_winner_score=final_score,
            num_comparisons=len(comparison_history),
            comparison_history=comparison_history,
            best_at_k=best_at_k,
            skipped=False,
        )
    
    def _get_score(self, eval_data: dict, benchmark: str) -> float:
        """Get evaluation score based on benchmark type"""
        if benchmark == "tau2bench":
            # tau2bench: reward_info.reward
            if "reward_info" in eval_data:
                return eval_data.get("reward_info", {}).get("reward", 0.0)
            return eval_data.get("reward", 0.0)
        elif benchmark in ("swebench", "terminalbench"):
            # swebench, terminalbench: reward field
            return float(eval_data.get("reward", 0))
        elif benchmark == "mcpbench":
            # mcpbench: needs to be computed from evaluation components
            # S_overall = (S_rule + S_llm) / 2
            # S_rule = avg(valid_tool_name_rate, input_schema_compliance, execution_success_rate)
            # S_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
            evaluation = eval_data.get("evaluation", {})
            
            # Rule-based score (0-1), handle None values
            valid_tool_name_rate = evaluation.get("valid_tool_name_rate") or 0.0
            input_schema_compliance = evaluation.get("input_schema_compliance") or 0.0
            execution_success_rate = evaluation.get("execution_success_rate") or 0.0
            s_rule = (valid_tool_name_rate + input_schema_compliance + execution_success_rate) / 3.0
            
            # LLM-judge score (normalized to 0-1), handle None values
            task_fulfillment = evaluation.get("task_fulfillment") or 0.0
            grounding = evaluation.get("grounding") or 0.0
            tool_appropriateness = evaluation.get("tool_appropriateness") or 0.0
            parameter_accuracy = evaluation.get("parameter_accuracy") or 0.0
            dependency_awareness = evaluation.get("dependency_awareness") or 0.0
            s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5.0 / 10.0
            
            s_overall = (s_rule + s_llm) / 2.0
            return s_overall
        else:
            # search, mathhay: score field
            return float(eval_data.get("score", 0))
