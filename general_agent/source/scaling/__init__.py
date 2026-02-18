"""
Sequential Scaling Module

Provides token budget-based scaling with EXTEND/STOP mechanisms for Universal Agent.

Key components:
- ScalingConfig: Configuration for EXTEND/STOP prompts and thresholds
- DeterministicConfig: Configuration for LLM determinism (seed, temperature)
- ScalingCheckpoint: Checkpoint data structure for saving/loading state
- CheckpointStore: File-based checkpoint storage manager
- ScalingController: Orchestrates multi-budget execution with checkpoint reuse
- SequentialRunResult: Result of a sequential scaling run
"""

from .config import ScalingConfig, DeterministicConfig
from .checkpoint import ScalingCheckpoint, CheckpointStore
from .controller import ScalingController, SequentialRunResult

__all__ = [
    "ScalingConfig",
    "DeterministicConfig",
    "ScalingCheckpoint", 
    "CheckpointStore",
    "ScalingController",
    "SequentialRunResult",
]
