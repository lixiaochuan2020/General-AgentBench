"""Self-Choice Evaluation Module

Tests the model's ability to autonomously judge the correctness of a trajectory.
"""

from .runner import SelfChoiceRunner, SelfChoiceResult
from .evaluator import SelfChoiceEvaluator, SelfChoiceEvalResult
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

__all__ = [
    "SelfChoiceRunner",
    "SelfChoiceResult",
    "SelfChoiceEvaluator",
    "SelfChoiceEvalResult",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
]
