"""Model route registry.

This file records the LiteLLM model route strings used in experiments.
- A single alias can map to multiple routes, meaning all listed routes were used
  across different runs/stages.

Keep secrets (API keys) in environment variables / .env, not in this file.
"""

from __future__ import annotations

from typing import Dict, List


MODELS: Dict[str, List[str]] = {
    "Qwen3-235B": ["bedrock/qwen.qwen3-235b-a22b-2507-v1:0"],
    "Qwen3-Next": ["huggingface/Qwen/Qwen3-Next-80B-A3B-Thinking:together"],
    "OpenAI-oss-120B": ["huggingface/openai/gpt-oss-120b:novita"],
    "Gemini-2.5-Flash": ["gemini/gemini-2.5-flash"],
    "Gemini-2.5-Pro": ["gemini/gemini-2.5-pro"],
    "Claude-Haiku-4.5": ["bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"],
    "Claude-Sonnet-4.5": [
        "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
    "DeepSeek-R1": [
        "huggingface/deepseek-ai/DeepSeek-R1-0528:novita",
        "huggingface/deepseek-ai/DeepSeek-R1-0528:together",
    ],
    "DeepSeek-V3.2": [
        "huggingface/deepseek-ai/DeepSeek-V3.2:novita",
        "huggingface/deepseek-ai/DeepSeek-V3.2:fireworks-ai",
        "bedrock/converse/deepseek.v3.2",
    ],
    "GPT-5": ["openai/gpt-5"],
}
