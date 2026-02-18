"""
TerminalBench MCP Server - Docker Container Bridge Mode

Core architecture changes:
- No longer uses Tmux directly on the host
- Interacts with Terminal-Bench Docker containers via docker exec
- Uses Terminal-Bench task directories (containing docker-compose.yaml)
- Reuses Terminal-Bench's Docker Compose mechanism

Tool design:
1. terminalbench_execute_bash - Execute bash commands inside the container
2. terminalbench_read_file - Read files inside the container
3. terminalbench_write_file - Write files inside the container
4. terminalbench_finish - Complete task

References:
- Architecture doc: docs/SWEBench_TerminalBench_Architecture.md Section 9
- Original implementation: terminal-bench/docker/mcp-server/server/server.py
"""

import json
import sys
import re
import logging
import os
import subprocess
import time
from typing import Any, Optional, Dict, List
from pathlib import Path
from datetime import datetime
from enum import Enum

from mcp.server.fastmcp import FastMCP

# Set up logger
logger = logging.getLogger(__name__)

# Terminal-Bench path
TERMINAL_BENCH_PATH = Path(__file__).parent.parent.parent.parent / "benchmarks" / "terminal-bench"
TERMINAL_BENCH_TASKS_PATH = TERMINAL_BENCH_PATH / "terminal-bench-core-0.1.1-tasks"

# Docker container internal path constants
CONTAINER_WORKSPACE = "/workspace"
CONTAINER_LOGS = "/logs"
CONTAINER_AGENT_LOGS = "/agent-logs"
CONTAINER_TESTS = "/tests"


# ============================================================================
# PytestParser - Reused from terminal_bench/parsers/pytest_parser.py
# ============================================================================

class UnitTestStatus(Enum):
    """Test status - Reused from terminal_bench/parsers/base_parser.py"""
    PASSED = "passed"
    FAILED = "failed"


class PytestTestStatus(Enum):
    """Pytest test status - Reused from terminal_bench/parsers/pytest_parser.py"""
    UNKNOWN = "unknown"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    XFAIL = "xfail"
    XPASS = "xpass"
    ERROR = "error"

    def to_test_status(self) -> UnitTestStatus:
        if self in (PytestTestStatus.PASSED, PytestTestStatus.XFAIL, PytestTestStatus.SKIPPED):
            return UnitTestStatus.PASSED
        else:
            return UnitTestStatus.FAILED


class PytestParser:
    """Pytest output parser - Reused from terminal_bench/parsers/pytest_parser.py"""
    
    SHORT_TEST_SUMMARY_INFO_PATTERN = r"=+\s*short test summary info\s*=+"
    FAILED_DESCRIPTION_DELIMETER = " - "

    def _clean_line(self, line: str) -> str:
        if line.startswith(PytestTestStatus.FAILED.name):
            parts = line.split(self.FAILED_DESCRIPTION_DELIMETER)
            if len(parts) > 1:
                line = self.FAILED_DESCRIPTION_DELIMETER.join(parts[:-1])
        return line

    def _parse_result_line(self, line: str) -> tuple:
        cleaned_line = self._clean_line(line)
        parts = cleaned_line.split(maxsplit=1)

        if len(parts) <= 1:
            return "", PytestTestStatus.UNKNOWN

        status = parts[0].strip().strip(":")

        if status not in PytestTestStatus.__members__:
            return "", PytestTestStatus.UNKNOWN

        test_path = parts[1].strip()
        test_name = test_path.split("::", maxsplit=1)[-1]

        if not test_name:
            return "", PytestTestStatus.UNKNOWN

        return test_name, PytestTestStatus[status]

    def _parse_test_results(self, lines: list) -> dict:
        results = {}
        for line in lines:
            test_name, status = self._parse_result_line(line)
            if status != PytestTestStatus.UNKNOWN:
                results[test_name] = status.to_test_status()
        return results

    def parse(self, content: str) -> dict:
        """Parse pytest output and return test result dict"""
        parts = re.split(
            pattern=self.SHORT_TEST_SUMMARY_INFO_PATTERN,
            string=content,
            flags=re.IGNORECASE,
            maxsplit=1,
        )

        if len(parts) < 2:
            raise ValueError("No short test summary info found in the provided content.")

        short_test_summary = parts[1]
        return self._parse_test_results(lines=short_test_summary.splitlines())


class TerminalBenchDockerServer:
    """
    TerminalBench MCP Server - Docker Container Bridge Mode
    
    Interacts with Terminal-Bench containers via docker exec.
    Uses Terminal-Bench's docker-compose.yaml to start containers.
    
    MCP Process Mode design:
    - Server runs as an independent process, no longer bound to a single task
    - Switches different task containers via __terminalbench_switch_container
    - Old containers are automatically cleaned up on switch
    """
    
    def __init__(self):
        """
        Initialize TerminalBench Docker Server
        
        No longer accepts task_id parameter; task switching is done via __terminalbench_switch_container.
        """
        self.name = "terminalbench"
        self.mcp = FastMCP(self.name)
        
        # Task info - initially empty, set by switch_container
        self.task_id: Optional[str] = None
        self.task_path: Optional[Path] = None
        self.container_name: Optional[str] = None
        self.container_running = False
        self.project_name: Optional[str] = None  # docker compose project name for cleanup
        
        # Task state
        self.task_finished = False
        self.task_result = None
        
        # Working directory (inside container)
        self.workspace = CONTAINER_WORKSPACE
        
        # Log directory (host)
        self.logs_path: Optional[Path] = None
        self.agent_logs_path: Optional[Path] = None
        
        # Initialize tools
        self._register_tools()
    
    def _docker_exec(self, command: str, timeout: float = 120.0) -> Dict[str, Any]:
        """
        Execute a command inside the Docker container
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
        
        Returns:
            {"output": str, "exit_code": int}
        """
        if not self.container_name or not self.container_running:
            return {
                "output": "Error: No container running. Call __init_task first.",
                "exit_code": -1
            }
        
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_name, "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            
            # Truncate overly long output
            max_output = 50000
            if len(output) > max_output:
                output = output[:max_output] + "\n<response clipped>"
            
            return {
                "output": output.strip() if output else "(no output)",
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "output": f"Command timed out after {timeout}s",
                "exit_code": -1
            }
        except Exception as e:
            logger.error(f"Docker exec error: {e}")
            return {
                "output": f"Error: {str(e)}",
                "exit_code": 1
            }
    
    def _docker_cp(self, src_path: str, dest_path: str, timeout: float = 60.0) -> bool:
        """
        Copy files or directories to the Docker container
        
        Args:
            src_path: Local source path
            dest_path: Destination path inside container
            timeout: Timeout in seconds
        
        Returns:
            True on success, False on failure
        """
        if not self.container_name or not self.container_running:
            logger.error("No container running for docker cp")
            return False
        
        try:
            result = subprocess.run(
                ["docker", "cp", src_path, f"{self.container_name}:{dest_path}"],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode != 0:
                logger.error(f"Docker cp failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Docker cp error: {e}")
            return False
    
    def _register_tools(self) -> None:
        """Register MCP tools"""
        
        server_ref = self
        
        # ============================================================================
        # Tool 1: terminalbench_execute_bash
        # Execute bash commands inside the container
        # ============================================================================
        
        @self.mcp.tool(name="terminalbench_execute_bash")
        def execute_bash(
            command: str,
            timeout: Optional[float] = None,
            wait_time_sec: float = 0.0
        ) -> str:
            """
            Execute a bash command in the Terminal-Bench Docker container.
            
            The command runs in the task's working directory.
            
            * Long running commands: For commands that may run indefinitely, it should 
              be run in the background and the output should be redirected to a file.
            * One command at a time: You can only execute one bash command at a time. 
              If you need to run multiple commands sequentially, use `&&` or `;`.
            
            Args:
                command: The bash command to execute.
                timeout: Optional timeout in seconds (default: 120).
                wait_time_sec: Time to wait after command execution before returning.
            
            Returns:
                The output of the command execution as JSON with output and exit_code.
            """
            actual_timeout = timeout if timeout else 120.0
            
            result = server_ref._docker_exec(command, timeout=actual_timeout)
            
            # Wait then return
            if wait_time_sec > 0:
                time.sleep(wait_time_sec)
            
            return json.dumps(result)
        
        # ============================================================================
        # Tool 2: terminalbench_read_file
        # Read files inside the container
        # ============================================================================
        
        @self.mcp.tool(name="terminalbench_read_file")
        def read_file(
            path: str,
            start_line: Optional[int] = None,
            end_line: Optional[int] = None
        ) -> str:
            """
            Read the contents of a file in the Docker container.
            
            Args:
                path: Absolute path to the file to read.
                start_line: Optional starting line number (1-based).
                end_line: Optional ending line number (1-based).
            
            Returns:
                The file contents or an error message.
            """
            try:
                if start_line and end_line:
                    result = server_ref._docker_exec(
                        f"sed -n '{start_line},{end_line}p' '{path}'"
                    )
                else:
                    result = server_ref._docker_exec(f"cat '{path}'")
                
                if result["exit_code"] != 0:
                    return f"Error: File not found or cannot read: {path}"
                
                return result["output"]
                
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        # ============================================================================
        # Tool 3: terminalbench_write_file
        # Write files inside the container
        # ============================================================================
        
        @self.mcp.tool(name="terminalbench_write_file")
        def write_file(
            path: str,
            content: str,
            append: bool = False
        ) -> str:
            """
            Write content to a file in the Docker container.
            
            Args:
                path: Absolute path to the file to write.
                content: The content to write.
                append: If True, append to file instead of overwriting.
            
            Returns:
                Success or error message.
            """
            try:
                import base64
                encoded = base64.b64encode(content.encode()).decode()
                
                # Ensure directory exists
                dir_path = str(Path(path).parent)
                server_ref._docker_exec(f"mkdir -p '{dir_path}'")
                
                if append:
                    cmd = f"echo '{encoded}' | base64 -d >> '{path}'"
                else:
                    cmd = f"echo '{encoded}' | base64 -d > '{path}'"
                
                result = server_ref._docker_exec(cmd, timeout=30)
                
                if result["exit_code"] == 0:
                    return f"Successfully wrote to {path}"
                else:
                    return f"Error writing file: {result['output']}"
                    
            except Exception as e:
                return f"Error writing file: {str(e)}"
        
        # ============================================================================
        # Tool 4: terminalbench_list_dir
        # List directory contents
        # ============================================================================
        
        @self.mcp.tool(name="terminalbench_list_dir")
        def list_dir(
            path: str,
            max_depth: int = 2
        ) -> str:
            """
            List contents of a directory in the Docker container.
            
            Args:
                path: Absolute path to the directory.
                max_depth: Maximum depth to list (default: 2).
            
            Returns:
                Directory listing or error message.
            """
            try:
                result = server_ref._docker_exec(
                    f"find '{path}' -maxdepth {max_depth} -not -name '.*' | head -100"
                )
                
                if result["exit_code"] != 0:
                    return f"Error: Directory not found: {path}"
                
                return result["output"]
                
            except Exception as e:
                return f"Error listing directory: {str(e)}"
        
        # ============================================================================
        # Tool 5: terminalbench_finish
        # Complete task
        # ============================================================================
        
        @self.mcp.tool(name="terminalbench_finish")
        def finish(message: str) -> str:
            """
            Signals the completion of the current task.
            
            Use this tool when:
            - You have successfully completed the task
            - You cannot proceed further due to technical limitations
            
            Args:
                message: Final message summarizing the actions taken
            
            Returns:
                Confirmation that the task was marked as complete.
            """
            server_ref.task_finished = True
            server_ref.task_result = message
            
            logger.info(f"[TerminalBench] Task finished: {message[:200]}...")
            
            return json.dumps({
                "status": "completed",
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
        
        # ============================================================================
        # Internal tool: Switch container (prefixed with __, not visible to Agent)
        # This is the core tool for MCP Process Mode, called by run.py
        # ============================================================================
        
        @self.mcp.tool(name="__terminalbench_switch_container")
        def switch_container(
            task_id: str,
            task_path: Optional[str] = None,
            output_dir: Optional[str] = None,
            no_rebuild: bool = False
        ) -> str:
            """
            [Internal] Switch to a new Terminal-Bench task container.
            
            This cleans up the current container (if any) and starts a new one.
            
            Args:
                task_id: Terminal-Bench task ID
                task_path: Optional explicit path to task directory
                output_dir: Optional output directory for logs
                no_rebuild: If True, don't rebuild Docker images
            
            Returns:
                JSON status
            """
            # 1. Clean up old container
            if server_ref.container_running and server_ref.project_name:
                logger.info(f"[TerminalBench] Cleaning up old container (project: {server_ref.project_name})")
                try:
                    if server_ref.task_path:
                        docker_compose_path = server_ref.task_path / "docker-compose.yaml"
                        if not docker_compose_path.exists():
                            docker_compose_path = server_ref.task_path / "docker-compose.yml"
                        
                        if docker_compose_path.exists():
                            cmd = [
                                "docker", "compose", "-f", str(docker_compose_path),
                                "-p", server_ref.project_name,
                                "down", "-v"
                            ]
                            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    # Also try to remove container directly
                    if server_ref.container_name:
                        subprocess.run(
                            ["docker", "rm", "-f", server_ref.container_name],
                            capture_output=True, timeout=30
                        )
                except Exception as e:
                    logger.warning(f"[TerminalBench] Error cleaning old container: {e}")
                
                server_ref.container_running = False
            
            # 2. Initialize new container
            try:
                # Find task directory
                if task_path:
                    task_dir = Path(task_path)
                else:
                    task_dir = TERMINAL_BENCH_TASKS_PATH / task_id
                
                if not task_dir.exists():
                    # Try other locations
                    alt_path = TERMINAL_BENCH_PATH / "tasks" / task_id
                    if alt_path.exists():
                        task_dir = alt_path
                    else:
                        return json.dumps({
                            "success": False,
                            "error": f"Task not found: {task_id}",
                            "searched_paths": [
                                str(TERMINAL_BENCH_TASKS_PATH / task_id),
                                str(alt_path)
                            ]
                        })
                
                docker_compose_path = task_dir / "docker-compose.yaml"
                if not docker_compose_path.exists():
                    docker_compose_path = task_dir / "docker-compose.yml"
                
                if not docker_compose_path.exists():
                    return json.dumps({
                        "success": False,
                        "error": f"docker-compose.yaml not found in {task_dir}"
                    })
                
                server_ref.task_id = task_id
                server_ref.task_path = task_dir
                
                # Set up log directory
                if output_dir:
                    logs_base = Path(output_dir)
                else:
                    logs_base = task_dir / "output"
                logs_base.mkdir(parents=True, exist_ok=True)
                
                server_ref.logs_path = logs_base / "logs"
                server_ref.agent_logs_path = logs_base / "agent-logs"
                server_ref.logs_path.mkdir(parents=True, exist_ok=True)
                server_ref.agent_logs_path.mkdir(parents=True, exist_ok=True)
                
                # Generate unique container name and project name
                # Docker project name only allows lowercase letters, digits, hyphens and underscores, and must start with a letter or digit
                import uuid
                container_suffix = str(uuid.uuid4())[:8]
                safe_task_id = task_id.replace(".", "_").replace("/", "_").lower()
                server_ref.project_name = f"terminalbench_{safe_task_id}_{container_suffix}"
                server_ref.container_name = server_ref.project_name
                
                # Set environment variables
                env = os.environ.copy()
                env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] = f"tb_eval_img_{safe_task_id}"
                env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] = server_ref.container_name
                env["T_BENCH_TASK_LOGS_PATH"] = str(server_ref.logs_path.absolute())
                env["T_BENCH_CONTAINER_LOGS_PATH"] = CONTAINER_LOGS
                env["T_BENCH_TASK_AGENT_LOGS_PATH"] = str(server_ref.agent_logs_path.absolute())
                env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] = CONTAINER_AGENT_LOGS
                env["T_BENCH_TEST_DIR"] = CONTAINER_TESTS
                
                # Start container
                build_flag = ["--build"] if not no_rebuild else []
                cmd = [
                    "docker", "compose", "-f", str(docker_compose_path),
                    "-p", server_ref.project_name,
                    "up", "-d"
                ] + build_flag
                
                logger.info(f"[TerminalBench] Starting container: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
                
                if result.returncode != 0:
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to start container: {result.stderr}",
                        "stdout": result.stdout
                    })
                
                # Wait for container to start
                time.sleep(3)
                
                # Get container ID
                ps_result = subprocess.run(
                    ["docker", "compose", "-f", str(docker_compose_path),
                     "-p", server_ref.project_name, "ps", "-q"],
                    capture_output=True, text=True, env=env
                )
                container_id = ps_result.stdout.strip().split('\n')[0]
                
                if not container_id:
                    # Try to find container directly
                    ps_result2 = subprocess.run(
                        ["docker", "ps", "-q", "-f", f"name={server_ref.container_name}"],
                        capture_output=True, text=True
                    )
                    container_id = ps_result2.stdout.strip()
                
                if container_id:
                    server_ref.container_name = container_id
                
                server_ref.container_running = True
                server_ref.task_finished = False
                server_ref.task_result = None
                
                logger.info(f"[TerminalBench] Container started: {server_ref.container_name} (project: {server_ref.project_name})")
                
                # Verify container internal environment
                verify_result = server_ref._docker_exec("pwd && ls -la", timeout=30)
                
                return json.dumps({
                    "success": True,
                    "task_id": task_id,
                    "container_name": server_ref.container_name,
                    "project_name": server_ref.project_name,
                    "task_path": str(task_dir),
                    "verification": verify_result["output"][:500]
                })
                
            except subprocess.TimeoutExpired:
                return json.dumps({
                    "success": False,
                    "error": "Timeout starting container"
                })
            except Exception as e:
                logger.error(f"Error switching container: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        # ============================================================================
        # Internal tool: Initialize task (prefixed with __, not visible to Agent)
        # Note: This tool has been superseded by __terminalbench_switch_container, kept for compatibility
        # ============================================================================
        
        @self.mcp.tool(name="__terminalbench_init_task")
        def init_task(
            task_id: str,
            task_path: Optional[str] = None,
            output_dir: Optional[str] = None,
            no_rebuild: bool = False
        ) -> str:
            """
            [Internal] Initialize a Terminal-Bench task environment.
            
            This starts the Docker container using the task's docker-compose.yaml.
            
            Args:
                task_id: Terminal-Bench task ID
                task_path: Optional explicit path to task directory
                output_dir: Optional output directory for logs
                no_rebuild: If True, don't rebuild Docker images
            
            Returns:
                JSON status
            """
            try:
                # Find task directory
                if task_path:
                    task_dir = Path(task_path)
                else:
                    task_dir = TERMINAL_BENCH_TASKS_PATH / task_id
                
                if not task_dir.exists():
                    # Try other locations
                    alt_path = TERMINAL_BENCH_PATH / "tasks" / task_id
                    if alt_path.exists():
                        task_dir = alt_path
                    else:
                        return json.dumps({
                            "success": False,
                            "error": f"Task not found: {task_id}",
                            "searched_paths": [
                                str(TERMINAL_BENCH_TASKS_PATH / task_id),
                                str(alt_path)
                            ]
                        })
                
                docker_compose_path = task_dir / "docker-compose.yaml"
                if not docker_compose_path.exists():
                    docker_compose_path = task_dir / "docker-compose.yml"
                
                if not docker_compose_path.exists():
                    return json.dumps({
                        "success": False,
                        "error": f"docker-compose.yaml not found in {task_dir}"
                    })
                
                server_ref.task_id = task_id
                server_ref.task_path = task_dir
                
                # Set up log directory
                if output_dir:
                    logs_base = Path(output_dir)
                else:
                    logs_base = task_dir / "output"
                logs_base.mkdir(parents=True, exist_ok=True)
                
                server_ref.logs_path = logs_base / "logs"
                server_ref.agent_logs_path = logs_base / "agent-logs"
                server_ref.logs_path.mkdir(parents=True, exist_ok=True)
                server_ref.agent_logs_path.mkdir(parents=True, exist_ok=True)
                
                # Generate unique container name
                import uuid
                container_suffix = str(uuid.uuid4())[:8]
                server_ref.container_name = f"terminalbench_{task_id}_{container_suffix}"
                
                # Set environment variables
                env = os.environ.copy()
                env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] = f"tb_eval_img_{task_id}"
                env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] = server_ref.container_name
                env["T_BENCH_TASK_LOGS_PATH"] = str(server_ref.logs_path.absolute())
                env["T_BENCH_CONTAINER_LOGS_PATH"] = CONTAINER_LOGS
                env["T_BENCH_TASK_AGENT_LOGS_PATH"] = str(server_ref.agent_logs_path.absolute())
                env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] = CONTAINER_AGENT_LOGS
                env["T_BENCH_TEST_DIR"] = CONTAINER_TESTS
                
                # Start container
                build_flag = ["--build"] if not no_rebuild else []
                cmd = [
                    "docker", "compose", "-f", str(docker_compose_path),
                    "-p", server_ref.container_name,
                    "up", "-d"
                ] + build_flag
                
                logger.info(f"[TerminalBench] Starting container: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
                
                if result.returncode != 0:
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to start container: {result.stderr}",
                        "stdout": result.stdout
                    })
                
                # Wait for container to start
                time.sleep(3)
                
                # Get container ID
                ps_result = subprocess.run(
                    ["docker", "compose", "-f", str(docker_compose_path),
                     "-p", server_ref.container_name, "ps", "-q"],
                    capture_output=True, text=True, env=env
                )
                container_id = ps_result.stdout.strip().split('\n')[0]
                
                if not container_id:
                    # Try to find container directly
                    ps_result2 = subprocess.run(
                        ["docker", "ps", "-q", "-f", f"name={server_ref.container_name}"],
                        capture_output=True, text=True
                    )
                    container_id = ps_result2.stdout.strip()
                
                if container_id:
                    server_ref.container_name = container_id
                
                server_ref.container_running = True
                server_ref.task_finished = False
                server_ref.task_result = None
                
                logger.info(f"[TerminalBench] Container started: {server_ref.container_name}")
                
                # Verify container internal environment
                verify_result = server_ref._docker_exec("pwd && ls -la", timeout=30)
                
                return json.dumps({
                    "success": True,
                    "task_id": task_id,
                    "container_name": server_ref.container_name,
                    "task_path": str(task_dir),
                    "verification": verify_result["output"][:500]
                })
                
            except subprocess.TimeoutExpired:
                return json.dumps({
                    "success": False,
                    "error": "Timeout starting container"
                })
            except Exception as e:
                logger.error(f"Error initializing task: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        # ============================================================================
        # Internal tool: Run tests (prefixed with __, not visible to Agent)
        # ============================================================================
        
        @self.mcp.tool(name="__terminalbench_run_tests")
        def run_tests() -> str:
            """
            [Internal] Run the Terminal-Bench test suite.
            
            This runs the tests defined in tests/run_tests.sh for the current task.
            
            Returns:
                JSON with test results and pass/fail status
            """
            try:
                if not server_ref.task_path:
                    return json.dumps({
                        "success": False,
                        "error": "No task initialized"
                    })
                
                # First copy the tests directory to the container's /tests directory
                tests_dir = server_ref.task_path / "tests"
                if tests_dir.exists():
                    # Create /tests directory
                    server_ref._docker_exec("mkdir -p /tests", timeout=30)
                    # Copy tests directory contents
                    if not server_ref._docker_cp(str(tests_dir) + "/.", "/tests/"):
                        logger.warning("Failed to copy tests directory to container")
                    else:
                        logger.info("Copied tests directory to container /tests/")
                
                # Find test script
                test_script_path = server_ref.task_path / "tests" / "run-tests.sh"
                if not test_script_path.exists():
                    # Try other locations
                    test_script_path = server_ref.task_path / "run-tests.sh"
                
                if test_script_path.exists():
                    with open(test_script_path) as f:
                        test_script = f.read()
                    
                    # Set TEST_DIR environment variable and execute test script
                    result = server_ref._docker_exec(
                        f"export TEST_DIR=/tests && bash << 'HEREDOC_END'\n{test_script}\nHEREDOC_END",
                        timeout=600  # 10 minute timeout
                    )
                    
                    output = result["output"]
                    
                    # Use PytestParser to parse results (consistent with original Terminal-Bench)
                    passed = False
                    try:
                        parser = PytestParser()
                        parser_results = parser.parse(output)
                        # All tests must pass to be considered passed (consistent with original harness._is_resolved)
                        if parser_results:
                            passed = all(
                                status == UnitTestStatus.PASSED 
                                for status in parser_results.values()
                            )
                            logger.info(f"Parsed {len(parser_results)} tests, all passed: {passed}")
                    except ValueError as e:
                        # Parsing failed, try simple check
                        logger.warning(f"PytestParser failed: {e}, falling back to simple check")
                        # Try to extract passed/failed counts from output
                        passed_match = re.search(r'(\d+)\s+passed', output)
                        failed_match = re.search(r'(\d+)\s+failed', output)
                        if passed_match and not failed_match:
                            passed = int(passed_match.group(1)) > 0
                        elif passed_match and failed_match:
                            passed = int(failed_match.group(1)) == 0
                    
                    return json.dumps({
                        "success": True,
                        "passed": passed,
                        "output": output[-10000:] if len(output) > 10000 else output,
                        "exit_code": result["exit_code"]
                    })
                else:
                    # Try running tests directly inside the container
                    result = server_ref._docker_exec(
                        "bash /tests/run-tests.sh",
                        timeout=600
                    )
                    
                    output = result["output"]
                    
                    # Use PytestParser to parse results
                    passed = False
                    try:
                        parser = PytestParser()
                        parser_results = parser.parse(output)
                        if parser_results:
                            passed = all(
                                status == UnitTestStatus.PASSED 
                                for status in parser_results.values()
                            )
                    except ValueError:
                        # Parsing failed, try simple check
                        passed_match = re.search(r'(\d+)\s+passed', output)
                        failed_match = re.search(r'(\d+)\s+failed', output)
                        if passed_match and not failed_match:
                            passed = int(passed_match.group(1)) > 0
                        elif passed_match and failed_match:
                            passed = int(failed_match.group(1)) == 0
                    
                    return json.dumps({
                        "success": True,
                        "passed": passed,
                        "output": output[-10000:] if len(output) > 10000 else output,
                        "exit_code": result["exit_code"]
                    })
                
            except Exception as e:
                logger.error(f"Error running tests: {e}")
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
            
            This stops and removes the Docker container.
            
            Returns:
                JSON status
            """
            try:
                if server_ref.task_path and server_ref.project_name:
                    docker_compose_path = server_ref.task_path / "docker-compose.yaml"
                    if not docker_compose_path.exists():
                        docker_compose_path = server_ref.task_path / "docker-compose.yml"
                    
                    if docker_compose_path.exists():
                        # Use project_name to stop and remove container
                        cmd = [
                            "docker", "compose", "-f", str(docker_compose_path),
                            "-p", server_ref.project_name,
                            "down", "-v"
                        ]
                        subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    # Also try to remove the container directly
                    if server_ref.container_name:
                        subprocess.run(
                            ["docker", "rm", "-f", server_ref.container_name],
                            capture_output=True, timeout=30
                        )
                
                server_ref.container_running = False
                server_ref.task_finished = False
                server_ref.task_result = None
                server_ref.project_name = None
                
                logger.info("[TerminalBench] Task cleaned up")
                
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
_default_server = TerminalBenchDockerServer()

# Export MCP instance
mcp = _default_server.mcp

# Exported tool names (tools visible to Agent)
EXPOSED_TOOLS = [
    "terminalbench_execute_bash",
    "terminalbench_read_file",
    "terminalbench_write_file",
    "terminalbench_list_dir",
    "terminalbench_finish",
]

# Internal tools (not visible to Agent, called by run.py)
INTERNAL_TOOLS = [
    "__terminalbench_switch_container",
    "__terminalbench_init_task",  # deprecated, use switch_container
    "__terminalbench_run_tests",
    "__terminalbench_cleanup",
]

# All tools
ALL_TOOLS = EXPOSED_TOOLS + INTERNAL_TOOLS


def get_server() -> TerminalBenchDockerServer:
    """Get the default server instance"""
    return _default_server


def create_server() -> TerminalBenchDockerServer:
    """Create a new server instance (no longer accepts task_id parameter)"""
    return TerminalBenchDockerServer()


def main():
    """MCP Server entry point"""
    print(f"[{datetime.now()}] Starting TerminalBench MCP Server (Docker Bridge Mode)...")
    print(f"Exposed tools: {EXPOSED_TOOLS}")
    print(f"Internal tools: {INTERNAL_TOOLS}")
    print(f"Terminal-Bench tasks path: {TERMINAL_BENCH_TASKS_PATH}")
    
    # Run MCP server
    mcp.run()
    
    print(f"[{datetime.now()}] TerminalBench MCP Server finished.")


if __name__ == "__main__":
    main()
