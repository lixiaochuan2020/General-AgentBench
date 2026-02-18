"""LLM API base class"""

from abc import ABC, abstractmethod
from typing import Any
from ..agent import LLMResponse


class BaseLLMAPI(ABC):
    """LLM API base class"""
    
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> LLMResponse: 
        """Generate a response"""
        pass
