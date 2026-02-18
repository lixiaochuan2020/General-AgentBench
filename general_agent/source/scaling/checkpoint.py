"""
Scaling Checkpoint Storage

Provides ScalingCheckpoint dataclass and CheckpointStore for persisting scaling state.
This enables prefix consistency verification and analysis across budget levels.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ScalingCheckpoint:
    """
    Checkpoint for Sequential Scaling runs
    
    Stores the state of an agent run at a specific budget level.
    Used for:
    - Verifying prefix consistency across budget levels
    - Analyzing EXTEND/STOP trigger patterns
    - Debugging and reproducibility
    
    Attributes:
        task_id: Unique identifier for the task
        benchmark: Benchmark name (tau2, mcpbench, search, mathhay, swebench, terminalbench)
        budget_level: Token budget used for this run (e.g., 8000, 16000, 32000)
        messages: Complete message history as list of dicts
        cumulative_tokens: Total tokens consumed at end of run
        extend_rounds: List of round numbers where EXTEND was triggered
        stop_rounds: List of round numbers where STOP was triggered
        total_prompt_tokens: Total prompt (input) tokens consumed
        total_output_tokens: Total output tokens generated
        rounds: List of round information dicts (for analysis)
        final_response: The final response content
        seed: The seed used for this run
    """
    task_id: str
    benchmark: str
    budget_level: int
    messages: list[dict] = field(default_factory=list)
    cumulative_tokens: int = 0
    extend_rounds: list[int] = field(default_factory=list)
    stop_rounds: list[int] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    rounds: list[dict] = field(default_factory=list)
    final_response: Optional[str] = None
    seed: int = 42
    
    # Additional fields for Sequential Scaling
    stop_prompt_indices: list[int] = field(default_factory=list)
    # Marks which messages contain STOP_PROMPT, used for cleaning during restoration
    # Example: [15] means messages[15] contains STOP_PROMPT
    
    is_complete: bool = False
    # Whether the task completed naturally (vs being truncated by budget)
    
    temperature: float = 0.7
    # Temperature parameter used at runtime
    
    total_steps: int = 0
    # Total number of steps executed
    
    task_metadata: dict = field(default_factory=dict)
    # Task metadata (domain, initial state, etc.)
    
    # Property aliases for compatibility with both naming conventions
    @property
    def prompt_tokens(self) -> int:
        """Alias for total_prompt_tokens"""
        return self.total_prompt_tokens
    
    @property
    def output_tokens(self) -> int:
        """Alias for total_output_tokens"""
        return self.total_output_tokens
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ScalingCheckpoint":
        """Create from dictionary"""
        return cls(**data)


class CheckpointStore:
    """
    File-based Checkpoint Storage Manager
    
    Manages saving and loading of ScalingCheckpoint objects.
    
    Storage structure:
        {checkpoint_dir}/
        ├── {benchmark}/
        │   ├── {task_id}/
        │   │   ├── budget_8000.json
        │   │   ├── budget_16000.json
        │   │   └── budget_32000.json
    
    Usage:
        store = CheckpointStore("checkpoints")
        store.save(checkpoint)
        checkpoint = store.load("tau2", "task_001", 8000)
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize CheckpointStore
        
        Args:
            checkpoint_dir: Root directory for checkpoint storage
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_checkpoint_path(self, benchmark: str, task_id: str, budget_level: int) -> Path:
        """
        Get the file path for a specific checkpoint
        
        Args:
            benchmark: Benchmark name
            task_id: Task identifier (sanitized for filesystem)
            budget_level: Token budget
            
        Returns:
            Path to the checkpoint JSON file
        """
        # Sanitize task_id for filesystem (replace problematic characters)
        safe_task_id = task_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        
        path = self.checkpoint_dir / benchmark / safe_task_id
        path.mkdir(parents=True, exist_ok=True)
        return path / f"budget_{budget_level}.json"
    
    def save(self, checkpoint: ScalingCheckpoint) -> Path:
        """
        Save a checkpoint to disk
        
        Args:
            checkpoint: ScalingCheckpoint to save
            
        Returns:
            Path to the saved checkpoint file
        """
        path = self._get_checkpoint_path(
            checkpoint.benchmark,
            checkpoint.task_id,
            checkpoint.budget_level,
        )
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
        
        return path
    
    def load(
        self, 
        benchmark: str, 
        task_id: str, 
        budget_level: int
    ) -> Optional[ScalingCheckpoint]:
        """
        Load a checkpoint from disk
        
        Args:
            benchmark: Benchmark name
            task_id: Task identifier
            budget_level: Token budget
            
        Returns:
            ScalingCheckpoint if found, None otherwise
        """
        path = self._get_checkpoint_path(benchmark, task_id, budget_level)
        
        if not path.exists():
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return ScalingCheckpoint.from_dict(data)
    
    def exists(self, benchmark: str, task_id: str, budget_level: int) -> bool:
        """
        Check if a checkpoint exists
        
        Args:
            benchmark: Benchmark name
            task_id: Task identifier
            budget_level: Token budget
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        path = self._get_checkpoint_path(benchmark, task_id, budget_level)
        return path.exists()
    
    def list_checkpoints(
        self, 
        benchmark: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> list[ScalingCheckpoint]:
        """
        List all checkpoints, optionally filtered by benchmark and/or task_id
        
        Args:
            benchmark: Filter by benchmark name (optional)
            task_id: Filter by task_id (optional)
            
        Returns:
            List of ScalingCheckpoint objects
        """
        checkpoints = []
        
        # Determine search paths
        if benchmark:
            benchmark_dirs = [self.checkpoint_dir / benchmark]
        else:
            benchmark_dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir()]
        
        for benchmark_dir in benchmark_dirs:
            if not benchmark_dir.exists():
                continue
            
            if task_id:
                safe_task_id = task_id.replace("/", "_").replace("\\", "_").replace(":", "_")
                task_dirs = [benchmark_dir / safe_task_id]
            else:
                task_dirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]
            
            for task_dir in task_dirs:
                if not task_dir.exists():
                    continue
                
                for checkpoint_file in task_dir.glob("budget_*.json"):
                    try:
                        with open(checkpoint_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        checkpoints.append(ScalingCheckpoint.from_dict(data))
                    except Exception:
                        # Skip invalid checkpoint files
                        pass
        
        return checkpoints
    
    def verify_prefix_consistency(
        self,
        benchmark: str,
        task_id: str,
        smaller_budget: int,
        larger_budget: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Verify that the smaller budget's messages are a prefix of the larger budget's messages
        
        Args:
            benchmark: Benchmark name
            task_id: Task identifier
            smaller_budget: Smaller token budget (e.g., 8000)
            larger_budget: Larger token budget (e.g., 16000)
            
        Returns:
            Tuple of (is_consistent, error_message)
            - (True, None) if consistent
            - (False, error_msg) if inconsistent
        """
        small_cp = self.load(benchmark, task_id, smaller_budget)
        large_cp = self.load(benchmark, task_id, larger_budget)
        
        if small_cp is None:
            return False, f"Checkpoint for budget {smaller_budget} not found"
        if large_cp is None:
            return False, f"Checkpoint for budget {larger_budget} not found"
        
        # Check that small messages are prefix of large messages
        small_msgs = small_cp.messages
        large_msgs = large_cp.messages
        
        if len(small_msgs) > len(large_msgs):
            return False, f"Smaller budget has more messages ({len(small_msgs)}) than larger ({len(large_msgs)})"
        
        # Compare each message in the prefix
        for i, (small_msg, large_msg) in enumerate(zip(small_msgs, large_msgs)):
            if small_msg != large_msg:
                return False, f"Message mismatch at index {i}: {small_msg} != {large_msg}"
        
        return True, None

    def clean_stop_prompts(self, checkpoint: ScalingCheckpoint) -> ScalingCheckpoint:
        """
        Remove STOP_PROMPT to restore a clean prefix
        
        Used when restoring from a lower-budget checkpoint in Sequential Scaling,
        cleaning out STOP_PROMPT so the agent can continue execution.
        
        Note: Only cleans STOP_PROMPT, not EXTEND_PROMPT.
        Reason: EXTEND trigger conditions are unrelated to the budget cap (triggers as long as budget is sufficient),
                so 8K and 16K make the same decision at the same position. Keeping EXTEND does not affect prefix consistency.
                Only STOP needs cleaning, because 8K was forced to stop at a certain position, but 16K should continue.
        
        Args:
            checkpoint: The checkpoint to clean
            
        Returns:
            Cleaned checkpoint (original is not modified; returns a new copy)
        """
        import copy
        
        # Create a deep copy to avoid modifying the original checkpoint
        cleaned_cp = copy.deepcopy(checkpoint)
        
        cleaned_messages = []
        removed_count = 0
        
        for i, msg in enumerate(cleaned_cp.messages):
            # Auto-detect whether it contains STOP_PROMPT (without relying on stop_prompt_indices)
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            has_stop_prompt = self._contains_stop_prompt(content)
            
            if has_stop_prompt or i in cleaned_cp.stop_prompt_indices:
                # This message contains STOP_PROMPT, clean it
                cleaned_msg = self._strip_stop_prompt(msg)
                # If the message is empty or only whitespace after cleaning, skip it
                cleaned_content = cleaned_msg.get("content", "") if isinstance(cleaned_msg, dict) else getattr(cleaned_msg, "content", "")
                if cleaned_content and cleaned_content.strip():
                    cleaned_messages.append(cleaned_msg)
                else:
                    removed_count += 1
            else:
                # Clean message, keep as-is
                cleaned_messages.append(msg)
        
        # Also remove the assistant final answer after STOP_PROMPT (if any)
        # Because it was generated under STOP_PROMPT coercion, may be low quality, and the agent should regenerate it
        if removed_count > 0 and len(cleaned_messages) > 0:
            last_msg = cleaned_messages[-1]
            last_role = last_msg.get("role", "") if isinstance(last_msg, dict) else getattr(last_msg, "role", "")
            if last_role == "assistant":
                # Check if this is a final answer triggered by STOP_PROMPT (usually has no tool_calls)
                last_tc = last_msg.get("tool_calls") if isinstance(last_msg, dict) else getattr(last_msg, "tool_calls", None)
                if not last_tc:
                    # Remove this forcibly generated final answer
                    cleaned_messages.pop()
                    removed_count += 1
        
        cleaned_cp.messages = cleaned_messages
        cleaned_cp.stop_prompt_indices = []
        return cleaned_cp
    
    def _contains_stop_prompt(self, content: str) -> bool:
        """Detect whether message content contains STOP_PROMPT"""
        if not content:
            return False
        
        stop_markers = [
            "**CRITICAL: You MUST provide your final answer immediately",
            "CRITICAL: You MUST provide your final answer immediately",
            "Do NOT perform any more tool calling or reasoning",
            "Return the final answer under the required format NOW",
        ]
        
        return any(marker in content for marker in stop_markers)
    
    def _strip_stop_prompt(self, message: dict) -> dict:
        """
        Remove STOP_PROMPT from the message (keep EXTEND_PROMPT)
        
        Args:
            message: Message dict {"role": "...", "content": "...", ...}
            
        Returns:
            Cleaned message dict
        """
        import copy
        cleaned_msg = copy.deepcopy(message)
        
        content = cleaned_msg.get("content", "")
        if not content:
            return cleaned_msg
        
        # STOP_PROMPT signature texts (ordered by priority, longer matches first)
        stop_markers = [
            "**CRITICAL: You MUST provide your final answer immediately",
            "CRITICAL: You MUST provide your final answer immediately",
            "**CRITICAL: You MUST provide your final answer",
            "CRITICAL: You MUST provide your final answer",
            " CRITICAL: Do NOT perform any more tool calling",  # With space prefix
            "CRITICAL: Do NOT perform any more tool calling",
            "**CRITICAL:",  # More lenient match
            " CRITICAL:",   # More lenient match with space prefix
        ]
        
        for marker in stop_markers:
            if marker in content:
                idx = content.find(marker)
                content = content[:idx].strip()
                break
        
        cleaned_msg["content"] = content
        return cleaned_msg
    
    def find_best_prefix_checkpoint(
        self,
        benchmark: str,
        task_id: str,
        target_budget: int,
        budget_levels: list[int] = None,
    ) -> Optional[ScalingCheckpoint]:
        """
        Find the best prefix checkpoint for Sequential Scaling restoration (chained loading)
        
        Dynamically scans existing checkpoint files to find the largest checkpoint smaller than target_budget.
        This enables chained loading: 96k←80k, 112k←96k, 128k←112k
        
        Args:
            benchmark: Benchmark name
            task_id: Task ID
            target_budget: Target budget (tokens, e.g. 80000)
            budget_levels: Available budget levels list (deprecated, now auto-scanned)
            
        Returns:
            Largest available prefix checkpoint, or None if not found
        """
        # Dynamically scan existing checkpoint files
        safe_task_id = task_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        task_dir = self.checkpoint_dir / benchmark / safe_task_id
        
        if not task_dir.exists():
            return None
        
        # Scan all budget_*.json files and extract budget values
        available_budgets = []
        for checkpoint_file in task_dir.glob("budget_*.json"):
            try:
                # Extract budget from filename: budget_64000.json -> 64000
                budget_str = checkpoint_file.stem.replace("budget_", "")
                budget_value = int(budget_str)
                available_budgets.append(budget_value)
            except ValueError:
                continue
        
        if not available_budgets:
            return None
        
        # Search from largest to smallest for checkpoints smaller than target_budget (chained loading)
        for budget in sorted(available_budgets, reverse=True):
            if budget < target_budget:
                cp = self.load(benchmark, task_id, budget)
                if cp is not None and not cp.is_complete:
                    # Found the largest incomplete checkpoint, resume from here
                    return cp
        
        return None
