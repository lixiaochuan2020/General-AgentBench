"""
SWEBench MCP Server

MCP Server fully consistent with OpenHands SWE-Bench tools.
Provides 3 core tools:
1. execute_bash - Execute bash commands
2. str_replace_editor - File view/create/edit
3. finish - Complete task

Note: Does not include think tool (removed per user request)

Reference: OpenHands/openhands/agenthub/codeact_agent/tools/
"""

import json
import sys
import logging
import os
import subprocess
from typing import Any, Optional, Dict, List
from pathlib import Path
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Set up logger
logger = logging.getLogger(__name__)


class SWEBenchServer:
    """
    SWEBench MCP Server
    
    Provides a tool interface fully consistent with OpenHands.
    Core tools: execute_bash, str_replace_editor, finish
    """
    
    def __init__(self):
        """Initialize SWEBench Server"""
        self.name = "swebench"
        self.mcp = FastMCP(self.name)
        
        # Working directory (default: /workspace)
        self.workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
        
        # File edit history (for undo_edit)
        self.edit_history: Dict[str, List[str]] = {}
        
        # Task state
        self.task_finished = False
        self.task_result = None
        
        # Initialize tools
        self._register_tools()
    
    def _register_tools(self) -> None:
        """
        Register MCP tools
        
        Fully consistent with OpenHands, only 3 core tools:
        1. execute_bash - Execute bash commands
        2. str_replace_editor - File view/create/edit
        3. finish - Complete task
        """
        
        server_ref = self
        
        # ============================================================================
        # Tool 1: execute_bash
        # Fully consistent with OpenHands bash.py
        # ============================================================================
        
        @self.mcp.tool(name="swebench_execute_bash")
        def execute_bash(
            command: str,
            is_input: str = "false",
            timeout: Optional[float] = None
        ) -> str:
            """
            Execute a bash command in the terminal.
            
            * Long running commands: For commands that may run indefinitely, it should 
              be run in the background and the output should be redirected to a file, 
              e.g. command = `python3 app.py > server.log 2>&1 &`. For commands that 
              need to run for a specific duration, you can set the "timeout" argument 
              to specify a hard timeout in seconds.
            * Interact with running process: If a bash command returns exit code `-1`, 
              this means the process is not yet finished. By setting `is_input` to `true`, 
              you can interact with the running process.
            * One command at a time: You can only execute one bash command at a time. 
              If you need to run multiple commands sequentially, you can use `&&` or `;` 
              to chain them together.
            
            Args:
                command: The bash command to execute. Can be empty string to view 
                        additional logs when previous exit code is `-1`. Can be `C-c` 
                        (Ctrl+C) to interrupt the currently running process.
                is_input: If "true", the command is an input to the running process. 
                         If "false", the command is a bash command to be executed.
                timeout: Optional. Sets a hard timeout in seconds for the command execution.
            
            Returns:
                The output of the command execution.
            """
            try:
                actual_timeout = timeout if timeout else 120.0  # Default 120 seconds
                
                # Execute command
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=actual_timeout,
                    cwd=str(server_ref.workspace) if server_ref.workspace.exists() else None,
                )
                
                output = result.stdout
                if result.stderr:
                    output += "\n" + result.stderr
                
                # Truncate overly long output
                max_output = 50000
                if len(output) > max_output:
                    output = output[:max_output] + "\n<response clipped>"
                
                return json.dumps({
                    "output": output.strip() if output else "(no output)",
                    "exit_code": result.returncode
                })
                
            except subprocess.TimeoutExpired:
                return json.dumps({
                    "output": "Command timed out",
                    "exit_code": -1,
                    "note": "Process not yet finished. Set is_input=true to interact."
                })
            except Exception as e:
                logger.error(f"Error executing bash: {e}")
                return json.dumps({
                    "error": str(e),
                    "exit_code": 1
                })
        
        # ============================================================================
        # Tool 2: str_replace_editor
        # Fully consistent with OpenHands str_replace_editor.py
        # ============================================================================
        
        @self.mcp.tool(name="swebench_str_replace_editor")
        def str_replace_editor(
            command: str,
            path: str,
            file_text: Optional[str] = None,
            old_str: Optional[str] = None,
            new_str: Optional[str] = None,
            insert_line: Optional[int] = None,
            view_range: Optional[List[int]] = None
        ) -> str:
            """
            Custom editing tool for viewing, creating and editing files.
            
            * State is persistent across command calls
            * If `path` is a file, `view` displays the result of applying `cat -n`. 
              If `path` is a directory, `view` lists non-hidden files up to 2 levels deep
            * The `create` command cannot be used if the specified `path` already exists
            * If a `command` generates a long output, it will be truncated
            * The `undo_edit` command will revert the last edit made to the file at `path`
            
            Notes for using the `str_replace` command:
            * The `old_str` parameter should match EXACTLY one or more consecutive lines 
              from the original file. Be mindful of whitespaces!
            * If the `old_str` parameter is not unique in the file, the replacement will 
              not be performed. Make sure to include enough context in `old_str`
            * The `new_str` parameter should contain the edited lines
            
            Args:
                command: The commands to run. Allowed options are: `view`, `create`, 
                        `str_replace`, `insert`, `undo_edit`.
                path: Absolute path to file or directory, e.g. `/workspace/file.py`
                file_text: Required for `create` command, with the content of the file
                old_str: Required for `str_replace` command, the string to replace
                new_str: For `str_replace` (optional) and `insert` (required), the new string
                insert_line: Required for `insert` command, the line number after which to insert
                view_range: Optional for `view` command, e.g. [11, 12] shows lines 11-12
            
            Returns:
                The result of the file operation.
            """
            try:
                file_path = Path(path)
                
                if command == "view":
                    # View file or directory
                    if file_path.is_dir():
                        # List directory contents
                        result = subprocess.run(
                            f"find {path} -maxdepth 2 -not -name '.*' | head -100",
                            shell=True,
                            capture_output=True,
                            text=True
                        )
                        return result.stdout.strip() if result.stdout else f"Empty directory: {path}"
                    elif file_path.exists():
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            lines = f.readlines()
                        
                        # Handle view_range
                        if view_range and len(view_range) >= 2:
                            start = max(0, view_range[0] - 1)  # Convert to 0-based
                            end = view_range[1] if view_range[1] > 0 else len(lines)
                            lines = lines[start:end]
                            line_start = view_range[0]
                        else:
                            line_start = 1
                        
                        # Add line numbers
                        numbered_lines = []
                        for i, line in enumerate(lines):
                            numbered_lines.append(f"{line_start + i:6d}\t{line.rstrip()}")
                        
                        content = "\n".join(numbered_lines)
                        
                        # Truncate overly long output
                        max_output = 30000
                        if len(content) > max_output:
                            content = content[:max_output] + "\n<response clipped>"
                        
                        return content
                    else:
                        return f"Error: File not found: {path}"
                
                elif command == "create":
                    # Create file
                    if file_path.exists():
                        return f"Error: File already exists: {path}. Use str_replace to edit."
                    
                    if file_text is None:
                        return "Error: file_text is required for create command"
                    
                    # Ensure directory exists
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(file_text)
                    
                    return f"File created successfully: {path}"
                
                elif command == "str_replace":
                    # String replacement
                    if not file_path.exists():
                        return f"Error: File not found: {path}"
                    
                    if old_str is None:
                        return "Error: old_str is required for str_replace command"
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Save history for undo
                    if path not in server_ref.edit_history:
                        server_ref.edit_history[path] = []
                    server_ref.edit_history[path].append(content)
                    
                    # Check old_str occurrence count
                    count = content.count(old_str)
                    if count == 0:
                        return f"Error: old_str not found in {path}. Make sure it matches exactly."
                    elif count > 1:
                        return f"Error: old_str found {count} times in {path}. Add more context to make it unique."
                    
                    # Execute replacement
                    new_content = content.replace(old_str, new_str or "", 1)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    return f"Successfully replaced string in {path}"
                
                elif command == "insert":
                    # Insert lines
                    if not file_path.exists():
                        return f"Error: File not found: {path}"
                    
                    if insert_line is None:
                        return "Error: insert_line is required for insert command"
                    
                    if new_str is None:
                        return "Error: new_str is required for insert command"
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Save history for undo
                    if path not in server_ref.edit_history:
                        server_ref.edit_history[path] = []
                    server_ref.edit_history[path].append("".join(lines))
                    
                    # Insert new lines
                    insert_idx = min(insert_line, len(lines))
                    new_lines = new_str.split('\n')
                    for i, line in enumerate(new_lines):
                        lines.insert(insert_idx + i, line + '\n')
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    return f"Successfully inserted {len(new_lines)} line(s) after line {insert_line} in {path}"
                
                elif command == "undo_edit":
                    # Undo edit
                    if path not in server_ref.edit_history or not server_ref.edit_history[path]:
                        return f"Error: No edit history for {path}"
                    
                    previous_content = server_ref.edit_history[path].pop()
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(previous_content)
                    
                    return f"Successfully reverted last edit to {path}"
                
                else:
                    return f"Error: Unknown command '{command}'. Allowed: view, create, str_replace, insert, undo_edit"
                
            except Exception as e:
                logger.error(f"Error in str_replace_editor: {e}")
                return json.dumps({
                    "error": str(e)
                })
        
        # ============================================================================
        # Tool 3: finish
        # Fully consistent with OpenHands finish.py
        # Modified: Auto-generate git diff patch for evaluation
        # ============================================================================
        
        @self.mcp.tool(name="swebench_finish")
        def finish(message: str) -> str:
            """
            Signals the completion of the current task.
            
            Use this tool when:
            - You have successfully completed the user's requested task
            - You cannot proceed further due to technical limitations
            
            The message should include:
            - A clear summary of actions taken and their results
            - Any next steps for the user
            - Explanation if you're unable to complete the task
            
            Args:
                message: Final message to send to the user
            
            Returns:
                Confirmation that the task was marked as complete, including the git patch.
            """
            import subprocess
            
            server_ref.task_finished = True
            server_ref.task_result = message
            
            # Auto-generate git diff patch (for SWE-Bench evaluation)
            patch = ""
            try:
                # Check if inside a git repository
                result = subprocess.run(
                    ['git', 'rev-parse', '--is-inside-work-tree'],
                    cwd=str(server_ref.workspace),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Get all changes (including staged and unstaged)
                    diff_result = subprocess.run(
                        ['git', 'diff', '--no-color', 'HEAD'],
                        cwd=str(server_ref.workspace),
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    patch = diff_result.stdout
                    
                    # If no diff, changes may already be staged, try git diff --cached
                    if not patch.strip():
                        cached_result = subprocess.run(
                            ['git', 'diff', '--no-color', '--cached'],
                            cwd=str(server_ref.workspace),
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        patch = cached_result.stdout
                        
                    logger.info(f"[SWEBench] Generated patch: {len(patch)} characters")
            except Exception as e:
                logger.warning(f"[SWEBench] Failed to generate git diff: {e}")
                patch = ""
            
            logger.info(f"[SWEBench] Task finished: {message[:200]}...")
            
            return json.dumps({
                "status": "completed",
                "message": message,
                "patch": patch,  # Contains git diff patch for evaluation
                "timestamp": datetime.now().isoformat()
            })
        
        # ============================================================================
        # Internal tool: Initialize task (prefixed with __, not visible to Agent)
        # ============================================================================
        
        @self.mcp.tool(name="__swebench_init_task")
        def init_task(
            instance_id: str,
            repo: str,
            base_commit: str,
            workspace_path: Optional[str] = None
        ) -> str:
            """
            [Internal] Initialize a SWE-Bench task environment.
            
            This is an internal tool used by the benchmark framework,
            not exposed to the agent.
            
            Args:
                instance_id: SWE-Bench instance ID
                repo: Repository name (e.g., "django/django")
                base_commit: Base commit hash
                workspace_path: Optional workspace path
            
            Returns:
                JSON status
            """
            try:
                if workspace_path:
                    server_ref.workspace = Path(workspace_path)
                
                server_ref.task_finished = False
                server_ref.task_result = None
                server_ref.edit_history = {}
                
                logger.info(f"[SWEBench] Task initialized: {instance_id}")
                
                return json.dumps({
                    "success": True,
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": base_commit,
                    "workspace": str(server_ref.workspace)
                })
                
            except Exception as e:
                logger.error(f"Error initializing task: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        # ============================================================================
        # Internal tool: Clean up task (prefixed with __, not visible to Agent)
        # ============================================================================
        
        @self.mcp.tool(name="__swebench_cleanup")
        def cleanup() -> str:
            """
            [Internal] Clean up the task environment.
            
            This is an internal tool used by the benchmark framework,
            not exposed to the agent.
            
            Returns:
                JSON status
            """
            try:
                server_ref.task_finished = False
                server_ref.task_result = None
                server_ref.edit_history = {}
                
                logger.info("[SWEBench] Task cleaned up")
                
                return json.dumps({
                    "success": True,
                    "message": "Task cleaned up"
                })
                
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })


# Create default instance for MCP use
_default_server = SWEBenchServer()

# Export MCP instance
mcp = _default_server.mcp

# Exported tool names (only 3 tools visible to Agent)
EXPOSED_TOOLS = [
    "swebench_execute_bash",
    "swebench_str_replace_editor",
    "swebench_finish",
]

# Internal tools (not visible to Agent)
INTERNAL_TOOLS = [
    "__swebench_init_task",
    "__swebench_cleanup",
]

# All tools
ALL_TOOLS = EXPOSED_TOOLS + INTERNAL_TOOLS


def get_server() -> SWEBenchServer:
    """Get default server instance"""
    return _default_server


def create_server() -> SWEBenchServer:
    """Create a new server instance"""
    return SWEBenchServer()


if __name__ == "__main__":
    import asyncio
    
    print(f"[{datetime.now()}] Starting SWEBench MCP Server...")
    print(f"Exposed tools: {EXPOSED_TOOLS}")
    print(f"Internal tools: {INTERNAL_TOOLS}")
    
    # Run MCP server
    mcp.run()
    
    print(f"[{datetime.now()}] SWEBench MCP Server finished.")
