"""MCP-Benchmark: All-in-One Multi-Task Benchmark Framework

Architecture:
- BenchmarkHost: MCP Client manager, connects to multiple MCP Servers
- UniversalAgent: Universal Agent, sees all tools from all benchmarks
- Native Evaluators: Uses each benchmark's native evaluator

Usage:
    from source import BenchmarkHost, UniversalAgent, Tau2Evaluator
"""

__version__ = "0.1.0"

from .host import BenchmarkHost, MCPClient
from .agent import UniversalAgent, AgentTrace, LLMResponse, ToolCall
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
    "MCPClient",
    # Agent
    "UniversalAgent",
    "AgentTrace",
    "LLMResponse",
    "ToolCall",
    # Evaluators
    "Tau2Evaluator",
    "NativeEvalResult",
    "NativeEvaluatorRegistry",
    "evaluate_with_native",
    # LLM API
    "OpenAIAPI",
    "LiteLLMAPI",
]