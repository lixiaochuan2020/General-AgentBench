"""Evaluator module - directly calls original benchmark evaluation code

SWEBenchEvaluator: 
    - Uses OpenHands evaluation flow
    - Calls swebench.harness.grading.get_eval_report

TerminalBenchEvaluator:
    - Uses Terminal-Bench evaluation flow
    - Calls terminal_bench/parsers/pytest_parser.py
"""

from .swebench_evaluator import SWEBenchEvaluator, SWEBenchEvalResult
from .terminalbench_evaluator import TerminalBenchEvaluator, TerminalBenchEvalResult

__all__ = [
    'SWEBenchEvaluator', 
    'SWEBenchEvalResult',
    'TerminalBenchEvaluator', 
    'TerminalBenchEvalResult',
]
