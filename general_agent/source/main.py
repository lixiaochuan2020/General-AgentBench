"""
MCP-Benchmark Module Entry Point

Provides basic package entry point and component exports.
For the actual execution entry point, use run.py.
For server configuration, refer to DEFAULT_SERVERS and get_all_servers() in run.py.
"""

# Export core components
from .host import BenchmarkHost
from .agent import UniversalAgent, AgentTrace, ToolCall, LLMResponse
from .native_evaluators import (
    Tau2Evaluator,
    NativeEvalResult,
    NativeEvaluatorRegistry,
    evaluate_with_native,
)
from .llm_api import OpenAIAPI, LiteLLMAPI

__all__ = [
    # Host
    "BenchmarkHost",
    # Agent
    "UniversalAgent",
    "AgentTrace",
    "ToolCall",
    "LLMResponse",
    # Evaluators
    "Tau2Evaluator",
    "NativeEvalResult",
    "NativeEvaluatorRegistry",
    "evaluate_with_native",
    # LLM API
    "OpenAIAPI",
    "LiteLLMAPI",
]
