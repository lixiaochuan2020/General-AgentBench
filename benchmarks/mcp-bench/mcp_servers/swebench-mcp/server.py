"""
SWEBench MCP Server - Simulated tools for SWE-Bench coding tasks.

This server provides placeholder/mock implementations of the tools used by
OpenHands CodeAct Agent for SWE-Bench. These are not functional implementations
but provide the correct tool schemas so the Universal Agent knows what tools
are available for SWE-Bench tasks.

Actual SWE-Bench evaluation would require Docker and the OpenHands runtime.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Create the MCP server
server = FastMCP(
    name="SWEBench Tools",
)

# ============================================================================
# Tool: execute_bash
# ============================================================================

@server.tool()
def execute_bash(
    command: str,
    is_input: str = "false",
    timeout: Optional[float] = None,
    security_risk: str = "low"
) -> Dict[str, Any]:
    """Execute a bash command in the terminal within a persistent shell session.
    
    ### Command Execution
    * One command at a time: You can only execute one bash command at a time. 
      If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
    * Persistent session: Commands execute in a persistent shell session where 
      environment variables, virtual environments, and working directory persist between commands.
    * Soft timeout: Commands have a soft timeout of 10 seconds.
    
    ### Long-running Commands
    * For commands that may run indefinitely, run them in the background and redirect output 
      to a file, e.g. `python3 app.py > server.log 2>&1 &`.
    * If a bash command returns exit code `-1`, this means the process is not yet finished.
      By setting `is_input` to `true`, you can send additional input or control commands.
    
    ### Output Handling
    * Output truncation: If the output exceeds a maximum length, it will be truncated.
    
    Args:
        command: The bash command to execute. Can be empty string to view additional logs 
                 when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt.
        is_input: If "true", the command is an input to the running process. Default is "false".
        timeout: Optional hard timeout in seconds for the command execution.
        security_risk: The security risk level of the command ("low", "medium", "high").
    
    Returns:
        The command output and exit code.
    """
    logger.info(f"[SWEBench Mock] execute_bash called: {command[:100]}...")
    
    # Mock response - actual implementation would require Docker runtime
    return {
        "status": "mock",
        "message": "This is a mock SWEBench tool. Actual execution requires Docker runtime.",
        "command": command,
        "exit_code": 0,
        "output": f"[Mock] Would execute: {command}"
    }


# ============================================================================
# Tool: str_replace_editor
# ============================================================================

@server.tool()
def str_replace_editor(
    command: str,
    path: str,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: Optional[List[int]] = None,
    security_risk: str = "low"
) -> Dict[str, Any]:
    """Custom editing tool for viewing, creating and editing files in plain-text format.
    
    Commands:
    * `view`: View a file or directory. If `path` is a file, displays with line numbers.
              If `path` is a directory, lists non-hidden files up to 2 levels deep.
              Use `view_range` to specify a line range (e.g., [1, 100]).
    * `create`: Create a new file with content. Cannot be used if file already exists.
              Use `file_text` parameter for the content.
    * `str_replace`: Replace text in a file. Provide `old_str` (exact match) and `new_str`.
              The `old_str` must match EXACTLY including whitespace and indentation.
    * `insert`: Insert text at a specific line. Use `insert_line` and `new_str`.
    * `undo_edit`: Revert the last edit made to the file at `path`.
    
    CRITICAL REQUIREMENTS:
    1. EXACT MATCHING: The `old_str` must match EXACTLY one or more consecutive lines.
    2. UNIQUENESS: Include sufficient context (3-5 lines) before and after the change point.
    3. Always use absolute file paths (starting with /).
    
    Args:
        command: The command to run ('view', 'create', 'str_replace', 'insert', 'undo_edit').
        path: Absolute path to the file or directory.
        file_text: Content for the new file (used with 'create').
        old_str: The exact text to replace (used with 'str_replace').
        new_str: The replacement text (used with 'str_replace' and 'insert').
        insert_line: Line number for insertion (used with 'insert').
        view_range: Line range to view, e.g., [1, 100] (used with 'view').
        security_risk: The security risk level ("low", "medium", "high").
    
    Returns:
        The result of the editor operation.
    """
    logger.info(f"[SWEBench Mock] str_replace_editor called: {command} on {path}")
    
    return {
        "status": "mock",
        "message": "This is a mock SWEBench tool. Actual execution requires Docker runtime.",
        "command": command,
        "path": path,
        "result": f"[Mock] Would execute {command} on {path}"
    }


# ============================================================================
# Tool: execute_ipython_cell
# ============================================================================

@server.tool()
def execute_ipython_cell(
    code: str,
    security_risk: str = "low"
) -> Dict[str, Any]:
    """Run a cell of Python code in an IPython environment.
    
    * Define variables and import packages before using them.
    * Variables defined in IPython are not available outside (e.g., in terminal).
    * Supports magic commands like %pip.
    
    Args:
        code: The Python code to execute. Supports magic commands.
        security_risk: The security risk level ("low", "medium", "high").
    
    Returns:
        The output of the code execution.
    """
    logger.info(f"[SWEBench Mock] execute_ipython_cell called: {code[:100]}...")
    
    return {
        "status": "mock",
        "message": "This is a mock SWEBench tool. Actual execution requires Docker runtime.",
        "code": code,
        "output": f"[Mock] Would execute Python code:\n{code}"
    }


# ============================================================================
# Tool: browser
# ============================================================================

@server.tool()
def browser(
    code: str,
    security_risk: str = "low"
) -> Dict[str, Any]:
    """Interact with the browser using Python code.
    
    Use it ONLY when you need to interact with a webpage.
    
    Available functions:
    - goto(url: str): Navigate to a URL
    - go_back(): Navigate back in history
    - go_forward(): Navigate forward in history
    - click(bid: str): Click on an element by bid
    - fill(bid: str, value: str): Fill an input field
    - select_option(bid: str, options: list): Select dropdown options
    - scroll(delta_x: float, delta_y: float): Scroll the page
    - hover(bid: str): Hover over an element
    - press(key: str): Press a key
    
    Multiple actions can be provided at once but will be executed sequentially.
    
    Args:
        code: Python code using browser functions. Example: goto('http://example.com')
        security_risk: The security risk level ("low", "medium", "high").
    
    Returns:
        The browser interaction result and page state.
    """
    logger.info(f"[SWEBench Mock] browser called: {code[:100]}...")
    
    return {
        "status": "mock",
        "message": "This is a mock SWEBench tool. Actual execution requires BrowserGym runtime.",
        "code": code,
        "result": f"[Mock] Would execute browser action: {code}"
    }


# ============================================================================
# Tool: think
# ============================================================================

@server.tool()
def think(thought: str) -> Dict[str, Any]:
    """Use this tool to think about something.
    
    It will not obtain new information or make any changes to the repository,
    but just log the thought. Use it when complex reasoning or brainstorming is needed.
    
    Common use cases:
    1. Exploring a repository and discovering the source of a bug
    2. After receiving test results, brainstorm ways to fix failing tests
    3. When planning a complex refactoring
    4. When designing a new feature
    5. When debugging a complex issue
    
    Args:
        thought: The thought to log.
    
    Returns:
        Confirmation that the thought was logged.
    """
    logger.info(f"[SWEBench] think: {thought[:200]}...")
    
    return {
        "status": "logged",
        "thought": thought,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Tool: finish
# ============================================================================

@server.tool()
def finish(message: str) -> Dict[str, Any]:
    """Signals the completion of the current task or conversation.
    
    Use this tool when:
    - You have successfully completed the user's requested task
    - You cannot proceed further due to technical limitations or missing information
    
    The message should include:
    - A clear summary of actions taken and their results
    - Any next steps for the user
    - Explanation if you're unable to complete the task
    
    Args:
        message: Final message to send to the user.
    
    Returns:
        Confirmation of task completion.
    """
    logger.info(f"[SWEBench] finish: {message[:200]}...")
    
    return {
        "status": "finished",
        "message": message,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    print(f"[{datetime.now()}] SWEBench MCP Server starting on stdio...")
    server.run(transport="stdio")
    print(f"[{datetime.now()}] SWEBench MCP Server finished.")
