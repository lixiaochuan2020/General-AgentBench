"""
Scaling Configuration

Defines ScalingConfig dataclass with EXTEND/STOP prompts and threshold parameters.
Defines DeterministicConfig for controlling LLM execution determinism.
"""

from dataclasses import dataclass


@dataclass
class DeterministicConfig:
    """
    LLM Execution Determinism Configuration
    
    Used to ensure prefix consistency in Sequential Scaling.
    
    Key insight: Due to the checkpoint mechanism, temperature > 0 can also guarantee prefix consistency:
    - 8K run: Generates trajectory A, saves to checkpoint
    - 16K run: Directly loads checkpoint (trajectory A), only executes the incremental part
    - Result: Prefix is identical (same data)
    
    Therefore temperature=0.7 is better: maintains model diversity, avoids degenerate output.
    
    Attributes:
        seed: Random seed for LLM API calls
        temperature: Sampling temperature, can be > 0 because checkpoint guarantees consistency
        top_p: Top-p sampling parameter
    """
    seed: int = 42
    temperature: float = 0.7
    top_p: float = 1.0
    
    def to_llm_kwargs(self) -> dict:
        """Convert to LLM API call parameters"""
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass
class ScalingConfig:
    """
    Sequential Scaling Configuration
    
    Controls EXTEND/STOP behavior when running under token budget mode.
    
    Attributes:
        extend_prompt: Message to inject when model stops but budget remains.
                       Encourages the model to continue exploring.
        stop_prompt: Message to inject when budget is insufficient but model wants to continue.
                     Forces the model to provide a final answer immediately.
        stop_buffer_rounds: Number of average rounds to reserve as buffer before triggering STOP.
                           Higher value = more conservative (earlier STOP).
                           Default 1.0 means reserve tokens for approximately 1 more round.
        seed: Random seed for LLM calls to ensure prefix consistency across budget levels.
              The same seed ensures 16K run's first 8K matches standalone 8K run.
    """
    
    extend_prompt: str = (
        "Before finalizing your answer, take additional time to verify your reasoning, "
        "consider alternative approaches, and search for any missing information that "
        "could strengthen your response."
    )
    
    stop_prompt: str = (
        "**CRITICAL: You MUST provide your final answer immediately. "
        "Do NOT perform any more tool calling or reasoning. "
        "Return the final answer under the required format NOW.**"
    )
    
    stop_buffer_rounds: float = 1.0  # Reserve tokens for approximately 1 round
    
    seed: int = 42  # Fixed seed for reproducibility and prefix consistency
    
    def __post_init__(self):
        """Validate configuration values"""
        if self.stop_buffer_rounds < 0:
            raise ValueError(f"stop_buffer_rounds must be >= 0, got {self.stop_buffer_rounds}")
        if not self.extend_prompt:
            raise ValueError("extend_prompt cannot be empty")
        if not self.stop_prompt:
            raise ValueError("stop_prompt cannot be empty")
