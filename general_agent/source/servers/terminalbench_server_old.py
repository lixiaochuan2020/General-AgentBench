"""
TerminalBench MCP Server

MCP Server fully consistent with the original Terminal-Bench.
Provides only 2 tools (consistent with the original):
1. keystrokes - Send keystrokes to the terminal
2. capture-pane - Capture terminal content

Reference: terminal-bench/docker/mcp-server/server/server.py
"""

import json
import sys
import logging
import time
from typing import Any, Optional, Dict
from pathlib import Path
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Set up logger
logger = logging.getLogger(__name__)

# Add terminal-bench to Python path
TERMINAL_BENCH_PATH = Path(__file__).parent.parent.parent.parent / "benchmarks" / "terminal-bench"
if TERMINAL_BENCH_PATH.exists():
    sys.path.insert(0, str(TERMINAL_BENCH_PATH))


class TerminalBenchServer:
    """
    TerminalBench MCP Server
    
    Provides a tool interface fully consistent with the original Terminal-Bench.
    Only 2 tools: keystrokes and capture-pane
    """
    
    def __init__(self, task_path: Optional[str] = None, session_name: str = "agent"):
        """
        Initialize TerminalBench Server
        
        Args:
            task_path: Task path (directory containing docker-compose.yaml)
            session_name: Tmux session name
        """
        self.name = "terminalbench"
        self.mcp = FastMCP(self.name)
        
        self.task_path = Path(task_path) if task_path else None
        self.session_name = session_name
        
        # Runtime components (lazy initialization)
        self.terminal = None  # Terminal instance
        self.tmux_session = None  # TmuxSession instance
        
        # Task state
        self.task_initialized = False
        
        # Initialize tools
        self._register_tools()
    
    def _register_tools(self) -> None:
        """
        Register MCP tools
        
        Fully consistent with the original Terminal-Bench, only 2 tools:
        1. keystrokes - Send keystrokes to the terminal
        2. capture-pane - Capture terminal content
        """
        
        server_ref = self
        
        # ============================================================================
        # Tool 1: keystrokes
        # Fully consistent with original: terminal-bench/docker/mcp-server/server/server.py
        # ============================================================================
        
        @self.mcp.tool(name="terminalbench_keystrokes")
        def keystrokes(
            keystrokes: str,
            wait_time_sec: float = 0.0,
            append_enter: bool = False
        ) -> str:
            """
            Send keystrokes to a tmux session.
            
            Args:
                keystrokes: Keystrokes to execute in the terminal. Use tmux-style 
                           escape sequences for special characters (e.g. C-c for ctrl-c).
                wait_time_sec: The number of expected seconds to wait for the command 
                              to complete.
                append_enter: Whether to append a newline character to the end of the 
                             keystrokes. (This is necessary to execute bash commands.)
            
            Returns:
                The current terminal pane content after sending keystrokes.
            """
            try:
                # Check if initialized
                if server_ref.tmux_session is None:
                    # Try simulation mode
                    logger.warning("No tmux session available, running in simulation mode")
                    return json.dumps({
                        "status": "simulated",
                        "message": f"Would send keystrokes: {keystrokes[:100]}...",
                        "keystrokes": keystrokes,
                        "append_enter": append_enter,
                        "wait_time_sec": wait_time_sec,
                        "note": "Task environment not initialized. Use init_task first."
                    })
                
                # Actually send keystrokes
                keys = keystrokes if not append_enter else [keystrokes, "Enter"]
                
                result = server_ref.tmux_session.send_keys(
                    keys=keys,
                    min_timeout_sec=wait_time_sec,
                    max_timeout_sec=wait_time_sec,
                )
                
                # Wait then capture pane
                if wait_time_sec > 0:
                    time.sleep(wait_time_sec)
                
                pane_content = server_ref.tmux_session.capture_pane()
                
                return pane_content
                
            except Exception as e:
                logger.error(f"Error sending keystrokes: {e}")
                return json.dumps({
                    "error": str(e),
                    "keystrokes": keystrokes[:100] if keystrokes else ""
                })
        
        # ============================================================================
        # Tool 2: capture-pane
        # Fully consistent with original: terminal-bench/docker/mcp-server/server/server.py
        # ============================================================================
        
        @self.mcp.tool(name="terminalbench_capture_pane")
        def capture_pane(
            wait_before_capture_sec: float = 0.0
        ) -> str:
            """
            Capture the pane of a tmux session.
            
            Args:
                wait_before_capture_sec: The number of seconds to wait before capturing 
                                        the pane. This is useful if you just executed a 
                                        command and want to wait a bit to capture the output.
            
            Returns:
                The current content of the terminal pane.
            """
            try:
                # Check if initialized
                if server_ref.tmux_session is None:
                    logger.warning("No tmux session available, running in simulation mode")
                    return json.dumps({
                        "status": "simulated",
                        "message": "Would capture terminal pane",
                        "note": "Task environment not initialized. Use init_task first.",
                        "sample_output": "$ \n"
                    })
                
                # Wait
                if wait_before_capture_sec > 0:
                    time.sleep(wait_before_capture_sec)
                
                # Capture pane
                pane_content = server_ref.tmux_session.capture_pane()
                
                return pane_content
                
            except Exception as e:
                logger.error(f"Error capturing pane: {e}")
                return json.dumps({
                    "error": str(e)
                })
        
        # ============================================================================
        # Internal tool: Initialize task (prefixed with __, not visible to Agent)
        # ============================================================================
        
        @self.mcp.tool(name="__terminalbench_init_task")
        def init_task(
            task_path: str,
            no_rebuild: bool = False,
            disable_recording: bool = True
        ) -> str:
            """
            [Internal] Initialize a Terminal-Bench task environment.
            
            This is an internal tool used by the benchmark framework,
            not exposed to the agent.
            
            Args:
                task_path: Path to the task directory (containing docker-compose.yaml)
                no_rebuild: If True, skip rebuilding Docker images
                disable_recording: If True, disable asciinema recording
            
            Returns:
                JSON status with session info
            """
            try:
                from terminal_bench.terminal import Terminal
                from terminal_bench.handlers.trial_handler import Task, TaskPaths
                
                task_dir = Path(task_path)
                
                # Validate task directory
                if not task_dir.exists():
                    return json.dumps({
                        "success": False,
                        "error": f"Task directory does not exist: {task_dir}"
                    })
                
                docker_compose = task_dir / "docker-compose.yaml"
                if not docker_compose.exists():
                    docker_compose = task_dir / "docker-compose.yml"
                
                if not docker_compose.exists():
                    return json.dumps({
                        "success": False,
                        "error": f"No docker-compose.yaml found in: {task_dir}"
                    })
                
                # Create Terminal instance
                task = Task(
                    paths=TaskPaths(
                        task_config=task_dir / "task.yaml",
                        tests=task_dir / "tests",
                        solution=task_dir / "solution.sh",
                        docker_compose=docker_compose,
                    ),
                    name=task_dir.name,
                )
                
                server_ref.terminal = Terminal(
                    task=task,
                    no_rebuild=no_rebuild,
                    disable_recording=disable_recording,
                )
                
                # Start environment
                server_ref.terminal.start()
                
                # Get tmux session
                server_ref.tmux_session = server_ref.terminal.tmux_session
                server_ref.task_path = task_dir
                server_ref.task_initialized = True
                
                logger.info(f"Task environment initialized: {task_dir.name}")
                
                return json.dumps({
                    "success": True,
                    "task_name": task_dir.name,
                    "session_name": server_ref.session_name,
                    "message": "Task environment ready"
                })
                
            except ImportError as e:
                logger.warning(f"Terminal-Bench not available: {e}")
                return json.dumps({
                    "success": False,
                    "error": f"Terminal-Bench not available: {e}",
                    "note": "Running in simulation mode"
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
        
        @self.mcp.tool(name="__terminalbench_cleanup")
        def cleanup() -> str:
            """
            [Internal] Clean up the task environment.
            
            This is an internal tool used by the benchmark framework,
            not exposed to the agent.
            
            Returns:
                JSON status
            """
            try:
                if server_ref.terminal is not None:
                    server_ref.terminal.stop()
                    server_ref.terminal = None
                    server_ref.tmux_session = None
                    server_ref.task_initialized = False
                    
                    logger.info("Task environment cleaned up")
                    
                    return json.dumps({
                        "success": True,
                        "message": "Task environment cleaned up"
                    })
                else:
                    return json.dumps({
                        "success": True,
                        "message": "No task environment to clean up"
                    })
                    
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })


# Create default instance for MCP use
_default_server = TerminalBenchServer()

# Export MCP instance
mcp = _default_server.mcp

# Exported tool names (only 2 tools visible to Agent)
EXPOSED_TOOLS = [
    "terminalbench_keystrokes",
    "terminalbench_capture_pane",
]

# Internal tools (not visible to Agent)
INTERNAL_TOOLS = [
    "__terminalbench_init_task",
    "__terminalbench_cleanup",
]

# All tools
ALL_TOOLS = EXPOSED_TOOLS + INTERNAL_TOOLS


def get_server() -> TerminalBenchServer:
    """Get the default server instance"""
    return _default_server


def create_server(task_path: Optional[str] = None, session_name: str = "agent") -> TerminalBenchServer:
    """Create a new server instance"""
    return TerminalBenchServer(task_path=task_path, session_name=session_name)


if __name__ == "__main__":
    import asyncio
    
    print(f"[{datetime.now()}] Starting TerminalBench MCP Server...")
    print(f"Exposed tools: {EXPOSED_TOOLS}")
    print(f"Internal tools: {INTERNAL_TOOLS}")
    
    # Run MCP server
    mcp.run()
    
    print(f"[{datetime.now()}] TerminalBench MCP Server finished.")
