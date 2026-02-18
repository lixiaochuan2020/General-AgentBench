"""
ScalingController - Sequential Scaling Coordinator

Coordinates the execution of multiple budget levels, automatically reusing prefix checkpoints.

Usage example:
    controller = ScalingController(
        budget_levels=[8, 16, 32, 64, 128],
        checkpoint_dir="./checkpoints",
    )
    
    # Run up to 32K, automatically reusing the 16K or 8K checkpoint
    result = await controller.run_sequential(
        agent=agent,
        task_id="task_001",
        benchmark="tau2",
        target_budget=32,
        instruction="...",
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING

from loguru import logger as loguru_logger

from .checkpoint import ScalingCheckpoint, CheckpointStore
from .config import DeterministicConfig

if TYPE_CHECKING:
    from ..agent import UniversalAgent, AgentTrace

logger = logging.getLogger(__name__)


@dataclass
class SequentialRunResult:
    """Sequential Scaling run result"""
    
    task_id: str
    benchmark: str
    target_budget: int
    
    # Execution result
    trace: "AgentTrace"
    checkpoint: ScalingCheckpoint
    
    # Prefix reuse information
    used_prefix: bool = False
    prefix_budget: Optional[int] = None
    prefix_tokens: int = 0
    
    # Token statistics (final values)
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    
    def to_dict(self) -> dict:
        """Convert to a serializable dictionary"""
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "target_budget": self.target_budget,
            "used_prefix": self.used_prefix,
            "prefix_budget": self.prefix_budget,
            "prefix_tokens": self.prefix_tokens,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "trace": self.trace.to_dict() if self.trace else None,
        }


class ScalingController:
    """
    Sequential Scaling Coordinator
    
    Coordinates the execution of multiple budget levels. Core features:
    1. Find reusable prefix checkpoints
    2. Clean STOP_PROMPT (remove forced stop prompts)
    3. Resume execution from checkpoint
    4. Save new checkpoints
    
    Usage:
        # Method 1: Automatic prefix reuse
        controller = ScalingController(...)
        result = await controller.run_sequential(agent, task_id, 32, ...)
        
        # Method 2: Manual control
        prefix = controller.find_best_prefix(task_id, benchmark, 32)
        if prefix:
            cleaned = controller.checkpoint_store.clean_stop_prompts(prefix)
            trace = await agent.run_from_checkpoint(cleaned, 32000, ...)
    """
    
    # Default budget levels (K tokens)
    DEFAULT_BUDGET_LEVELS = [8, 16, 32, 64, 128]
    
    def __init__(
        self,
        budget_levels: list[int] = None,
        checkpoint_dir: str = "./checkpoints",
        deterministic_config: DeterministicConfig = None,
    ):
        """
        Initialize ScalingController
        
        Args:
            budget_levels: List of budget levels (K tokens), e.g. [8, 16, 32, 64, 128]
            checkpoint_dir: Checkpoint storage directory
            deterministic_config: Deterministic configuration (seed, temperature, etc.)
        """
        self.budget_levels = budget_levels or self.DEFAULT_BUDGET_LEVELS
        self.checkpoint_store = CheckpointStore(checkpoint_dir)
        self.deterministic_config = deterministic_config or DeterministicConfig()
        
        # Sort budget levels (ensure ascending order)
        self.budget_levels = sorted(self.budget_levels)
        
        loguru_logger.info(
            f"[ScalingController] Initialized with budget_levels={self.budget_levels} "
            f"checkpoint_dir={checkpoint_dir}"
        )
    
    def find_best_prefix(
        self,
        benchmark: str,
        task_id: str,
        target_budget: int,
    ) -> Optional[ScalingCheckpoint]:
        """
        Find the largest available prefix checkpoint
        
        Rules:
        1. Only search for budget levels smaller than target_budget
        2. Skip checkpoints with is_complete=True (task completed naturally)
        3. Return the largest available checkpoint
        
        Args:
            benchmark: Benchmark name
            task_id: Task ID
            target_budget: Target budget (K tokens)
            
        Returns:
            The largest available prefix checkpoint, or None if not found
        """
        return self.checkpoint_store.find_best_prefix_checkpoint(
            benchmark=benchmark,
            task_id=task_id,
            target_budget=target_budget,
            budget_levels=self.budget_levels,
        )
    
    async def run_sequential(
        self,
        agent: "UniversalAgent",
        task_id: str,
        benchmark: str,
        target_budget: int,
        instruction: str = "",
        policy: str = None,
        user_simulator: Any = None,
        task_metadata: dict = None,
        **run_kwargs,
    ) -> SequentialRunResult:
        """
        Run up to the specified budget, automatically reusing prefix checkpoints
        
        Core flow:
        1. Find a reusable prefix checkpoint
        2. If found: clean STOP_PROMPT, resume from checkpoint
        3. If not found: start execution from scratch
        4. Save new checkpoint
        
        Args:
            agent: UniversalAgent instance
            task_id: Task ID
            benchmark: Benchmark name (tau2, mcp, search, swebench)
            target_budget: Target budget (K tokens)
            instruction: Task instruction (used when starting from scratch)
            policy: Policy string (optional)
            user_simulator: User Simulator adapter (used for tau2-bench)
            task_metadata: Task metadata (saved to checkpoint)
            **run_kwargs: Additional arguments passed to agent.run() or agent.run_from_checkpoint()
            
        Returns:
            SequentialRunResult: Contains trace, checkpoint, and statistics
        """
        loguru_logger.info(
            f"[ScalingController] run_sequential task_id={task_id} "
            f"benchmark={benchmark} target_budget={target_budget}K"
        )
        
        # 1. Find a reusable prefix checkpoint
        prefix_checkpoint = self.find_best_prefix(benchmark, task_id, target_budget)
        
        used_prefix = False
        prefix_budget = None
        prefix_tokens = 0
        trace = None
        
        if prefix_checkpoint:
            # 2. Clean STOP_PROMPT
            cleaned_checkpoint = self.checkpoint_store.clean_stop_prompts(prefix_checkpoint)
            
            loguru_logger.info(
                f"[ScalingController] Found prefix checkpoint: budget={prefix_checkpoint.budget_level} "
                f"tokens={prefix_checkpoint.cumulative_tokens} messages={len(prefix_checkpoint.messages)}"
            )
            
            used_prefix = True
            prefix_budget = prefix_checkpoint.budget_level // 1000  # Convert to K
            prefix_tokens = prefix_checkpoint.cumulative_tokens
            
            # 3. Set agent's target_budget
            target_budget_tokens = target_budget * 1000
            
            # 4. Resume execution from checkpoint
            if user_simulator is not None:
                # tau2-bench: with User Simulator
                trace = await agent.run_from_checkpoint_with_user_simulator(
                    checkpoint=cleaned_checkpoint,
                    target_budget=target_budget_tokens,
                    user_simulator=user_simulator,
                    policy=policy,
                    **run_kwargs,
                )
            else:
                # mcp-bench / search: without User Simulator
                trace = await agent.run_from_checkpoint(
                    checkpoint=cleaned_checkpoint,
                    target_budget=target_budget_tokens,
                    policy=policy,
                    **run_kwargs,
                )
        else:
            loguru_logger.info(
                f"[ScalingController] No prefix checkpoint found, starting fresh"
            )
            
            # 5. Start execution from scratch
            target_budget_tokens = target_budget * 1000
            agent.target_budget = target_budget_tokens
            
            if user_simulator is not None:
                # tau2-bench: with User Simulator
                trace = await agent.run_with_user_simulator(
                    task_id=task_id,
                    user_simulator=user_simulator,
                    policy=policy,
                    **run_kwargs,
                )
            else:
                # mcp-bench / search: without User Simulator
                trace = await agent.run(
                    task_id=task_id,
                    instruction=instruction,
                    policy=policy,
                    **run_kwargs,
                )
        
        # 6. Create and save new checkpoint
        checkpoint = self._create_checkpoint(
            task_id=task_id,
            benchmark=benchmark,
            budget_level=target_budget * 1000,
            trace=trace,
            task_metadata=task_metadata,
            agent=agent,
        )
        self.checkpoint_store.save(checkpoint)
        
        loguru_logger.info(
            f"[ScalingController] Completed: total_tokens={trace.total_tokens} "
            f"used_prefix={used_prefix} prefix_budget={prefix_budget}K"
        )
        
        # 7. Build result
        result = SequentialRunResult(
            task_id=task_id,
            benchmark=benchmark,
            target_budget=target_budget,
            trace=trace,
            checkpoint=checkpoint,
            used_prefix=used_prefix,
            prefix_budget=prefix_budget,
            prefix_tokens=prefix_tokens,
            total_tokens=trace.total_tokens,
            total_prompt_tokens=trace.total_prompt_tokens,
            total_output_tokens=trace.total_output_tokens,
        )
        
        return result
    
    def _create_checkpoint(
        self,
        task_id: str,
        benchmark: str,
        budget_level: int,
        trace: "AgentTrace",
        task_metadata: dict = None,
        agent: "UniversalAgent" = None,
    ) -> ScalingCheckpoint:
        """
        Create a ScalingCheckpoint from an AgentTrace
        
        Args:
            task_id: Task ID
            benchmark: Benchmark name
            budget_level: Budget level (tokens, not K)
            trace: AgentTrace object
            task_metadata: Task metadata
            agent: UniversalAgent (used to retrieve stop_rounds)
            
        Returns:
            ScalingCheckpoint object
        """
        # Get stop_prompt_indices
        stop_prompt_indices = []
        if agent is not None and hasattr(agent, 'stop_rounds'):
            stop_prompt_indices = agent.stop_rounds.copy()
        
        # Determine if the task completed naturally (was not forcefully stopped by STOP)
        is_complete = (
            trace.error is None and 
            len(stop_prompt_indices) == 0 and
            trace.final_response is not None
        )
        
        # Get message list
        messages = []
        if hasattr(trace, 'messages'):
            messages = [
                m.to_dict() if hasattr(m, 'to_dict') else m 
                for m in trace.messages
            ]
        
        # Get rounds list
        rounds = []
        if hasattr(trace, 'rounds'):
            rounds = [
                r if isinstance(r, dict) else {
                    "round_number": r.round_number,
                    "reasoning": r.reasoning,
                    "prompt_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "round_total_tokens": r.round_total_tokens,
                    "cumulative_total_tokens": r.cumulative_total_tokens,
                    "tools_executed": r.tools_executed,
                }
                for r in trace.rounds
            ]
        
        checkpoint = ScalingCheckpoint(
            task_id=task_id,
            benchmark=benchmark,
            budget_level=budget_level,
            messages=messages,
            rounds=rounds,
            cumulative_tokens=trace.total_tokens,
            total_prompt_tokens=trace.total_prompt_tokens,
            total_output_tokens=trace.total_output_tokens,
            total_steps=trace.total_steps,
            stop_prompt_indices=stop_prompt_indices,
            is_complete=is_complete,
            temperature=self.deterministic_config.temperature,
            task_metadata=task_metadata or {},
        )
        
        return checkpoint
    
    async def run_all_budgets(
        self,
        agent: "UniversalAgent",
        task_id: str,
        benchmark: str,
        instruction: str = "",
        policy: str = None,
        user_simulator: Any = None,
        task_metadata: dict = None,
        **run_kwargs,
    ) -> list[SequentialRunResult]:
        """
        Run all budget levels, each level automatically reusing the previous level's checkpoint
        
        This is the complete Sequential Scaling flow, executing all budget levels in order.
        
        Args:
            agent: UniversalAgent instance
            task_id: Task ID
            benchmark: Benchmark name
            instruction: Task instruction
            policy: Policy string
            user_simulator: User Simulator adapter
            task_metadata: Task metadata
            **run_kwargs: Additional arguments
            
        Returns:
            List of run results for all budget levels
        """
        results = []
        
        for target_budget in self.budget_levels:
            loguru_logger.info(
                f"[ScalingController] Running budget level {target_budget}K "
                f"({self.budget_levels.index(target_budget) + 1}/{len(self.budget_levels)})"
            )
            
            result = await self.run_sequential(
                agent=agent,
                task_id=task_id,
                benchmark=benchmark,
                target_budget=target_budget,
                instruction=instruction,
                policy=policy,
                user_simulator=user_simulator,
                task_metadata=task_metadata,
                **run_kwargs,
            )
            
            results.append(result)
            
            # If the task completed naturally, subsequent budgets may not need to continue
            if result.checkpoint.is_complete:
                loguru_logger.info(
                    f"[ScalingController] Task completed naturally at {target_budget}K, "
                    f"remaining budgets will reuse this checkpoint"
                )
        
        return results
