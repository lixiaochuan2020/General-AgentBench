"""
TerminalBench MCP Server - Simulated tools for TerminalBench tasks.

This server provides placeholder/mock implementations of the tools used by
the Terminus agent for TerminalBench. These are not functional implementations
but provide the correct tool schemas so the Universal Agent knows what tools
are available for TerminalBench tasks.

Actual TerminalBench evaluation would require tmux sessions and the full runtime.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Create the MCP server
server = FastMCP(
    name="TerminalBench Tools",
)

# ============================================================================
# Tool: send_keystrokes
# ============================================================================

@server.tool()
def send_keystrokes(
    keystrokes: str,
    duration: float = 1.0
) -> Dict[str, Any]:
    """Send keystrokes to the terminal and wait for the result.
    
    This is the primary interaction method for terminal-based tasks.
    The text inside "keystrokes" will be used completely verbatim as keystrokes.
    
    Command Guidelines:
    - Most bash commands should end with a newline (\\n) to cause them to execute
    - For special key sequences, use tmux-style escape sequences:
      - C-c for Ctrl+C
      - C-d for Ctrl+D
      - C-z for Ctrl+Z
    
    Duration Guidelines:
    - Immediate tasks (cd, ls, echo, cat): set duration to 0.1 seconds
    - Normal commands (gcc, find, rustc): set duration to 1.0 seconds
    - Slow commands (make, python scripts, wget): set appropriate longer duration
    - Never wait longer than 60 seconds; prefer to poll for intermediate results
    
    To wait without action, use: {"keystrokes": "", "duration": 10.0}
    
    Args:
        keystrokes: The exact keystrokes to send to the terminal. 
                    Include \\n at the end for command execution.
        duration: Seconds to wait for the command to complete before returning.
                  Default is 1.0 seconds.
    
    Returns:
        The terminal output after waiting for the specified duration.
    """
    logger.info(f"[TerminalBench Mock] send_keystrokes: {repr(keystrokes)}, duration={duration}")
    
    # Mock response - actual implementation would require tmux session
    return {
        "status": "mock",
        "message": "This is a mock TerminalBench tool. Actual execution requires tmux session.",
        "keystrokes": keystrokes,
        "duration": duration,
        "terminal_output": f"[Mock] Would send keystrokes: {repr(keystrokes)} and wait {duration}s"
    }


# ============================================================================
# Tool: get_terminal_state
# ============================================================================

@server.tool()
def get_terminal_state() -> Dict[str, Any]:
    """Get the current state of the terminal.
    
    Returns the visible content of the terminal screen.
    Use this to check the current state before deciding on the next action.
    
    Returns:
        The current terminal screen content.
    """
    logger.info("[TerminalBench Mock] get_terminal_state called")
    
    return {
        "status": "mock",
        "message": "This is a mock TerminalBench tool. Actual execution requires tmux session.",
        "terminal_state": "[Mock] Terminal state would be shown here",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Tool: wait_for_output
# ============================================================================

@server.tool()
def wait_for_output(
    duration: float = 5.0,
    pattern: Optional[str] = None
) -> Dict[str, Any]:
    """Wait for terminal output without sending any keystrokes.
    
    Use this when waiting for a long-running command to complete or
    waiting for specific output to appear.
    
    Args:
        duration: Number of seconds to wait before returning. Default is 5.0 seconds.
        pattern: Optional pattern to watch for. Returns early if pattern is matched.
    
    Returns:
        The terminal output after waiting.
    """
    logger.info(f"[TerminalBench Mock] wait_for_output: duration={duration}, pattern={pattern}")
    
    return {
        "status": "mock",
        "message": "This is a mock TerminalBench tool. Actual execution requires tmux session.",
        "duration": duration,
        "pattern": pattern,
        "terminal_output": f"[Mock] Would wait {duration}s for output"
    }


# ============================================================================
# Tool: mark_task_complete
# ============================================================================

@server.tool()
def mark_task_complete(confirmed: bool = False) -> Dict[str, Any]:
    """Mark the current task as complete.
    
    This will trigger the solution to be graded and you won't be able to
    make any further corrections.
    
    The first call will ask for confirmation. Call again with confirmed=True
    to actually mark the task as complete.
    
    Args:
        confirmed: Set to True to confirm task completion. 
                   First call without confirmation will return a warning.
    
    Returns:
        Confirmation status or warning message.
    """
    logger.info(f"[TerminalBench Mock] mark_task_complete: confirmed={confirmed}")
    
    if not confirmed:
        return {
            "status": "pending_confirmation",
            "message": "Are you sure you want to mark the task as complete? "
                       "This will trigger your solution to be graded and you won't be able to "
                       "make any further corrections. Call again with confirmed=True to proceed."
        }
    
    return {
        "status": "complete",
        "message": "Task marked as complete.",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Tool: analyze_task
# ============================================================================

@server.tool()
def analyze_task(analysis: str, plan: str) -> Dict[str, Any]:
    """Log your analysis of the current state and plan for next steps.
    
    Use this to structure your thinking before executing commands.
    This is similar to the analysis and plan fields in the Terminus agent.
    
    Args:
        analysis: Your analysis of the current state based on terminal output.
                  What do you see? What has been accomplished? What still needs to be done?
        plan: Your plan for the next steps. What commands will you run and why?
    
    Returns:
        Confirmation that the analysis was logged.
    """
    logger.info(f"[TerminalBench] analyze_task - Analysis: {analysis[:100]}... Plan: {plan[:100]}...")
    
    return {
        "status": "logged",
        "analysis": analysis,
        "plan": plan,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    print(f"[{datetime.now()}] TerminalBench MCP Server starting on stdio...")
    server.run(transport="stdio")
    print(f"[{datetime.now()}] TerminalBench MCP Server finished.")
