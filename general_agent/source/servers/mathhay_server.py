"""
MathHay Server

MathHay is a long-context math reasoning benchmark that does not require any external tools.
All information is provided in the prompt, and the model needs to:
1. Find relevant documents among a large number of distractor documents
2. Extract numerical information from the documents
3. Perform multi-step math reasoning

This Server exists only for framework consistency; it does not actually provide any tools.
The model gets all information directly from the prompt and answers the question.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP


class MathHayServer:
    """
    MathHay Benchmark Server
    
    This is an "empty" server because MathHay does not require any tool calls.
    All information needed for math reasoning is provided in the long context.
    
    The internal tools provided are only for state management:
    - reset_state: Reset server state
    - get_answer: Get current answer
    - set_answer: Set answer
    """
    
    def __init__(self):
        """Initialize MathHay Server"""
        self.name = "mathhay"
        self.mcp = FastMCP(self.name)
        
        # Task state
        self.current_answer = None
        self.current_reasoning = None
        self.task_completed = False
        
        # Register tools
        self._register_tools()
    
    def reset_state(self) -> None:
        """Reset server state (called before each task)"""
        self.current_answer = None
        self.current_reasoning = None
        self.task_completed = False
    
    def get_answer(self) -> Optional[float]:
        """Get the current submitted answer"""
        return self.current_answer
    
    def get_reasoning(self) -> Optional[str]:
        """Get the current reasoning process"""
        return self.current_reasoning
    
    def set_answer(self, answer: float, reasoning: str = "") -> None:
        """Set the answer and reasoning process"""
        self.current_answer = answer
        self.current_reasoning = reasoning
        self.task_completed = True
    
    def _register_tools(self) -> None:
        """Register MCP tools"""
        
        @self.mcp.tool(description="Reset MathHay server state. Call this before starting a new task.")
        def reset_state() -> str:
            """Reset server state"""
            self.reset_state()
            return "MathHay server state reset successfully."
        
        @self.mcp.tool(description="Get the current answer for the MathHay task.")
        def get_answer() -> str:
            """Get current answer"""
            if self.current_answer is not None:
                return json.dumps({
                    "answer": self.current_answer,
                    "reasoning": self.current_reasoning,
                    "completed": self.task_completed
                })
            return json.dumps({"answer": None, "completed": False})
        
        @self.mcp.tool(description="Submit the final answer for the MathHay task.")
        def submit_answer(answer: float, reasoning: str = "") -> str:
            """
            Submit the final answer
            
            Args:
                answer: Numerical answer
                reasoning: Reasoning process (optional)
            
            Returns:
                Confirmation message
            """
            self.set_answer(answer, reasoning)
            return f"Answer submitted: {answer}"
    
    def run(self) -> None:
        """Run MCP server"""
        self.mcp.run()


# Create global instance
server = MathHayServer()


def main():
    """Entry point"""
    server.run()


if __name__ == "__main__":
    main()
