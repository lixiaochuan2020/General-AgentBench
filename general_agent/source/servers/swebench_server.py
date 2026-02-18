"""
SWEBench MCP Server - Docker Container Bridge Mode

Core architecture changes:
- No longer executes commands on the host
- Interacts with SWE-Bench Docker containers via docker exec
- Uses Terminal-Bench generated task directories (containing docker-compose.yaml)
- Uses official SWE-Bench Docker images (swebench/sweb.eval.x86_64.*)

Tool design:
1. swebench_execute_bash - Execute bash commands inside the container
2. swebench_str_replace_editor - View/create/edit files inside the container
3. swebench_finish - Complete task and generate git patch

References:
- Architecture doc: docs/SWEBench_TerminalBench_Architecture.md Section 9
- OpenHands tools: OpenHands/openhands/agenthub/codeact_agent/tools/
"""

import json
import sys
import logging
import os
import subprocess
import asyncio
from typing import Any, Optional, Dict, List
from pathlib import Path
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Set up logger
logger = logging.getLogger(__name__)

# Terminal-Bench dataset path
TERMINAL_BENCH_PATH = Path(__file__).parent.parent.parent.parent / "benchmarks" / "terminal-bench"
SWEBENCH_DATASET_PATH = TERMINAL_BENCH_PATH / "dataset" / "swebench-verified"

# Docker container internal path constants
CONTAINER_TESTBED = "/testbed"
CONTAINER_LOGS = "/logs"
CONTAINER_AGENT_LOGS = "/agent-logs"
CONTAINER_TESTS = "/tests"


class SWEBenchDockerServer:
    """
    SWEBench MCP Server - Docker Container Bridge Mode
    
    Interacts with SWE-Bench containers via docker exec instead of executing directly on the host.
    This allows using official SWE-Bench Docker images, ensuring the environment matches the original benchmark.
    
    Architecture notes (Plan A - MCP Process Mode):
    - This Server runs as an independent MCP process
    - Switches Docker containers between tasks via __swebench_switch_container tool
    - run.py (orchestrator) calls internal tools; Agent can only see 3 public tools
    """
    
    def __init__(self):
        """
        Initialize SWEBench Docker Server
        
        Note: No longer accepts task_id parameter.
        Task switching is done via the __swebench_switch_container tool.
        """
        self.name = "swebench"
        self.mcp = FastMCP(self.name)
        
        # Task info (initially None, set via switch_container)
        self.task_id: Optional[str] = None
        self.task_path: Optional[Path] = None
        self.container_name: Optional[str] = None
        self.project_name: Optional[str] = None  # docker compose project name
        self.container_running = False
        
        # File edit history (for undo_edit)
        self.edit_history: Dict[str, List[str]] = {}
        
        # Task state
        self.task_finished = False
        self.task_result = None
        self.generated_patch = ""
        
        # Working directory (inside container)
        self.workspace = CONTAINER_TESTBED
        
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
    
    def _docker_read_file(self, path: str) -> str:
        """Read a file inside the container"""
        result = self._docker_exec(f"cat '{path}'", timeout=30)
        if result["exit_code"] != 0:
            raise FileNotFoundError(f"File not found or cannot read: {path}")
        return result["output"]
    
    def _docker_write_file(self, path: str, content: str) -> bool:
        """Write a file inside the container"""
        # Use heredoc to write file, preserving format
        # Use base64 encoding to avoid special character issues
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        cmd = f"echo '{encoded}' | base64 -d > '{path}'"
        result = self._docker_exec(cmd, timeout=30)
        return result["exit_code"] == 0
    
    def _register_tools(self) -> None:
        """Register MCP tools"""
        
        server_ref = self
        
        # ============================================================================
        # Tool 1: swebench_execute_bash
        # Consistent with OpenHands bash.py behavior, but executes via docker exec
        # ============================================================================
        
        @self.mcp.tool(name="swebench_execute_bash")
        def execute_bash(
            command: str,
            is_input: str = "false",
            timeout: Optional[float] = None
        ) -> str:
            """
            Execute a bash command in the SWE-Bench Docker container.
            
            The command runs in /testbed which contains the repository code.
            
            * Long running commands: For commands that may run indefinitely, it should 
              be run in the background and the output should be redirected to a file, 
              e.g. command = `python3 app.py > server.log 2>&1 &`.
            * One command at a time: You can only execute one bash command at a time. 
              If you need to run multiple commands sequentially, use `&&` or `;`.
            
            Args:
                command: The bash command to execute.
                is_input: If "true", the command is an input to the running process.
                timeout: Optional timeout in seconds (default: 120).
            
            Returns:
                The output of the command execution as JSON with output and exit_code.
            """
            actual_timeout = timeout if timeout else 120.0
            
            # Execute command in the /testbed directory
            full_command = f"cd {server_ref.workspace} && {command}"
            result = server_ref._docker_exec(full_command, timeout=actual_timeout)
            
            return json.dumps(result)
        
        # ============================================================================
        # Tool 2: swebench_str_replace_editor
        # Consistent with OpenHands str_replace_editor.py behavior
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
            Custom editing tool for viewing, creating and editing files in the repository.
            
            * State is persistent across command calls
            * If `path` is a file, `view` displays the result of applying `cat -n`
            * If `path` is a directory, `view` lists non-hidden files up to 2 levels deep
            * The `create` command cannot be used if the specified `path` already exists
            * The `undo_edit` command will revert the last edit made to the file
            
            Notes for using the `str_replace` command:
            * The `old_str` parameter should match EXACTLY one or more consecutive lines
            * If `old_str` is not unique, the replacement will not be performed
            * Include enough context in `old_str` to make it unique
            
            Args:
                command: The command to run: `view`, `create`, `str_replace`, `insert`, `undo_edit`
                path: Absolute path to file or directory, e.g. `/testbed/file.py`
                file_text: Required for `create` command
                old_str: Required for `str_replace` command
                new_str: For `str_replace` (optional) and `insert` (required)
                insert_line: Required for `insert` command
                view_range: Optional for `view`, e.g. [11, 12] shows lines 11-12
            
            Returns:
                The result of the file operation.
            """
            try:
                if command == "view":
                    # Check if it's a file or directory
                    check_result = server_ref._docker_exec(f"test -d '{path}' && echo 'dir' || echo 'file'")
                    is_dir = "dir" in check_result["output"]
                    
                    if is_dir:
                        # List directory contents
                        result = server_ref._docker_exec(
                            f"find '{path}' -maxdepth 2 -not -name '.*' | head -100"
                        )
                        return result["output"]
                    else:
                        # Read file
                        if view_range and len(view_range) >= 2:
                            start, end = view_range[0], view_range[1]
                            result = server_ref._docker_exec(
                                f"sed -n '{start},{end}p' '{path}' | cat -n | awk '{{print {start-1}+NR\"\\t\"$0}}'"
                            )
                        else:
                            result = server_ref._docker_exec(f"cat -n '{path}'")
                        
                        if result["exit_code"] != 0:
                            return f"Error: File not found: {path}"
                        
                        # Truncate overly long output
                        content = result["output"]
                        max_output = 30000
                        if len(content) > max_output:
                            content = content[:max_output] + "\n<response clipped>"
                        
                        return content
                
                elif command == "create":
                    # Check if file exists
                    check_result = server_ref._docker_exec(f"test -f '{path}' && echo 'exists'")
                    if "exists" in check_result["output"]:
                        return f"Error: File already exists: {path}. Use str_replace to edit."
                    
                    if file_text is None:
                        return "Error: file_text is required for create command"
                    
                    # Ensure directory exists
                    dir_path = str(Path(path).parent)
                    server_ref._docker_exec(f"mkdir -p '{dir_path}'")
                    
                    # Write file
                    if server_ref._docker_write_file(path, file_text):
                        return f"File created successfully: {path}"
                    else:
                        return f"Error: Failed to create file: {path}"
                
                elif command == "str_replace":
                    if old_str is None:
                        return "Error: old_str is required for str_replace command"
                    
                    # Read file content
                    try:
                        content = server_ref._docker_read_file(path)
                    except FileNotFoundError:
                        return f"Error: File not found: {path}"
                    
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
                    
                    # Write file
                    if server_ref._docker_write_file(path, new_content):
                        return f"Successfully replaced string in {path}"
                    else:
                        return f"Error: Failed to write file: {path}"
                
                elif command == "insert":
                    if insert_line is None:
                        return "Error: insert_line is required for insert command"
                    if new_str is None:
                        return "Error: new_str is required for insert command"
                    
                    # Read file content
                    try:
                        content = server_ref._docker_read_file(path)
                    except FileNotFoundError:
                        return f"Error: File not found: {path}"
                    
                    lines = content.split('\n')
                    
                    # Save history for undo
                    if path not in server_ref.edit_history:
                        server_ref.edit_history[path] = []
                    server_ref.edit_history[path].append(content)
                    
                    # Insert new lines
                    insert_idx = min(insert_line, len(lines))
                    new_lines = new_str.split('\n')
                    for i, line in enumerate(new_lines):
                        lines.insert(insert_idx + i, line)
                    
                    new_content = '\n'.join(lines)
                    
                    if server_ref._docker_write_file(path, new_content):
                        return f"Successfully inserted {len(new_lines)} line(s) after line {insert_line} in {path}"
                    else:
                        return f"Error: Failed to write file: {path}"
                
                elif command == "undo_edit":
                    if path not in server_ref.edit_history or not server_ref.edit_history[path]:
                        return f"Error: No edit history for {path}"
                    
                    previous_content = server_ref.edit_history[path].pop()
                    
                    if server_ref._docker_write_file(path, previous_content):
                        return f"Successfully reverted last edit to {path}"
                    else:
                        return f"Error: Failed to revert file: {path}"
                
                else:
                    return f"Error: Unknown command '{command}'. Allowed: view, create, str_replace, insert, undo_edit"
                
            except Exception as e:
                logger.error(f"Error in str_replace_editor: {e}")
                return json.dumps({"error": str(e)})
        
        # ============================================================================
        # Tool 3: swebench_finish
        # Complete task and generate git diff patch
        # ============================================================================
        
        @self.mcp.tool(name="swebench_finish")
        def finish(message: str) -> str:
            """
            Signals the completion of the current task.
            
            This will:
            1. Generate a git diff patch of all changes made
            2. Mark the task as complete
            
            Use this tool when:
            - You have successfully fixed the bug
            - You cannot proceed further due to technical limitations
            
            Args:
                message: Final message summarizing the changes made
            
            Returns:
                Confirmation with the generated git patch.
            """
            server_ref.task_finished = True
            server_ref.task_result = message
            
            # Generate git diff patch
            patch = ""
            try:
                # Get all changes (including staged and unstaged)
                diff_result = server_ref._docker_exec(
                    f"cd {server_ref.workspace} && git diff --no-color HEAD",
                    timeout=60
                )
                patch = diff_result["output"]
                
                # If no diff, try git diff --cached
                if not patch.strip() or diff_result["exit_code"] != 0:
                    cached_result = server_ref._docker_exec(
                        f"cd {server_ref.workspace} && git diff --no-color --cached",
                        timeout=60
                    )
                    if cached_result["output"].strip():
                        patch = cached_result["output"]
                
                server_ref.generated_patch = patch
                logger.info(f"[SWEBench] Generated patch: {len(patch)} characters")
                
            except Exception as e:
                logger.warning(f"[SWEBench] Failed to generate git diff: {e}")
                patch = ""
            
            logger.info(f"[SWEBench] Task finished: {message[:200]}...")
            
            return json.dumps({
                "status": "completed",
                "message": message,
                "patch": patch,
                "patch_length": len(patch),
                "timestamp": datetime.now().isoformat()
            })
        
        # ============================================================================
        # Internal tool: Switch container (prefixed with __, not visible to Agent)
        # Called by run.py (orchestrator) to switch Docker containers between tasks
        # ============================================================================
        
        @self.mcp.tool(name="__swebench_switch_container")
        def switch_container(
            task_id: str,
            output_dir: Optional[str] = None,
            no_rebuild: bool = True
        ) -> str:
            """
            [Internal] Switch to a new SWE-Bench task's Docker container.
            
            This tool is called by run.py (orchestrator), NOT by the Agent.
            The Agent cannot see this tool.
            
            Workflow:
            1. Clean up old container (if any)
            2. Start new task's Docker container
            3. Update server internal state
            
            Args:
                task_id: SWE-Bench instance ID (e.g., astropy__astropy-12907)
                output_dir: Optional output directory for logs
                no_rebuild: If True, don't rebuild Docker images (default: True)
            
            Returns:
                JSON status
            """
            # ========== Step 1: Clean up old container ==========
            if server_ref.container_running and server_ref.project_name:
                try:
                    logger.info(f"[SWEBench] Cleaning up old container: {server_ref.project_name}")
                    old_docker_compose = server_ref.task_path / "docker-compose.yaml" if server_ref.task_path else None
                    if old_docker_compose and old_docker_compose.exists():
                        cleanup_cmd = [
                            "docker", "compose", "-f", str(old_docker_compose),
                            "-p", server_ref.project_name,
                            "down", "-v"
                        ]
                        subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=60)
                    
                    # Also try to remove container directly
                    if server_ref.container_name:
                        subprocess.run(
                            ["docker", "rm", "-f", server_ref.container_name],
                            capture_output=True, timeout=30
                        )
                    
                    server_ref.container_running = False
                    logger.info(f"[SWEBench] Old container cleaned up")
                except Exception as e:
                    logger.warning(f"[SWEBench] Failed to cleanup old container: {e}")
            
            # ========== Step 2: Reset state ==========
            server_ref.task_finished = False
            server_ref.task_result = None
            server_ref.edit_history = {}
            server_ref.generated_patch = ""
            # ========== Step 3: Find new task directory ==========
            try:
                task_path = SWEBENCH_DATASET_PATH / task_id
                if not task_path.exists():
                    return json.dumps({
                        "success": False,
                        "error": f"Task not found: {task_id}",
                        "searched_path": str(task_path)
                    })
                
                docker_compose_path = task_path / "docker-compose.yaml"
                if not docker_compose_path.exists():
                    return json.dumps({
                        "success": False,
                        "error": f"docker-compose.yaml not found in {task_path}"
                    })
                
                server_ref.task_id = task_id
                server_ref.task_path = task_path
                
                # ========== Step 4: Set up log directory ==========
                if output_dir:
                    logs_base = Path(output_dir)
                else:
                    logs_base = task_path / "output"
                logs_base.mkdir(parents=True, exist_ok=True)
                
                server_ref.logs_path = logs_base / "logs"
                server_ref.agent_logs_path = logs_base / "agent-logs"
                server_ref.logs_path.mkdir(parents=True, exist_ok=True)
                server_ref.agent_logs_path.mkdir(parents=True, exist_ok=True)
                
                # ========== Step 5: Generate unique container name and project name ==========
                import uuid
                container_suffix = str(uuid.uuid4())[:8]
                server_ref.project_name = f"swebench_{task_id.replace('__', '_').replace('/', '_')}_{container_suffix}"
                server_ref.container_name = None  # Will be obtained from docker compose ps later
                
                # Set environment variables
                env = os.environ.copy()
                env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] = f"swebench_img_{task_id.replace('__', '_')}"
                env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] = server_ref.project_name
                env["T_BENCH_TASK_LOGS_PATH"] = str(server_ref.logs_path.absolute())
                env["T_BENCH_CONTAINER_LOGS_PATH"] = CONTAINER_LOGS
                env["T_BENCH_TASK_AGENT_LOGS_PATH"] = str(server_ref.agent_logs_path.absolute())
                env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] = CONTAINER_AGENT_LOGS
                env["T_BENCH_TEST_DIR"] = CONTAINER_TESTS
                
                # ========== Step 6: Start new container ==========
                build_flag = [] if no_rebuild else ["--build"]
                cmd = [
                    "docker", "compose", "-f", str(docker_compose_path),
                    "-p", server_ref.project_name,
                    "up", "-d"
                ] + build_flag
                
                logger.info(f"[SWEBench] Starting container: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
                
                if result.returncode != 0:
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to start container: {result.stderr}",
                        "stdout": result.stdout
                    })
                
                # ========== Step 7: Wait for container to start and get container ID ==========
                import time
                time.sleep(2)
                
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
                        ["docker", "ps", "-q", "-f", f"name={server_ref.project_name}"],
                        capture_output=True, text=True
                    )
                    container_id = ps_result2.stdout.strip()
                
                if container_id:
                    server_ref.container_name = container_id
                else:
                    return json.dumps({
                        "success": False,
                        "error": "Failed to get container ID after starting"
                    })
                
                server_ref.container_running = True
                
                logger.info(f"[SWEBench] Container started: {server_ref.container_name} (project: {server_ref.project_name})")
                
                # ========== Step 8: Verify container internal environment ==========
                verify_result = server_ref._docker_exec("pwd && ls -la", timeout=30)
                
                return json.dumps({
                    "success": True,
                    "task_id": task_id,
                    "container_name": server_ref.container_name,
                    "project_name": server_ref.project_name,
                    "workspace": server_ref.workspace,
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
        # Uses swebench.harness.grading to get standardized FAIL_TO_PASS/PASS_TO_PASS results
        # ============================================================================
        
        @self.mcp.tool(name="__swebench_run_tests")
        def run_tests() -> str:
            """
            [Internal] Run the SWE-Bench test suite with full evaluation.
            
            This runs the tests and uses swebench.harness.grading to get
            standardized test results including FAIL_TO_PASS and PASS_TO_PASS status.
            
            Returns:
                JSON with test results matching original SWE-Bench format:
                - resolved: bool - Whether the issue was resolved
                - tests_status: dict - Detailed test status with FAIL_TO_PASS/PASS_TO_PASS
                - report: dict - Full evaluation report
            """
            try:
                if not server_ref.task_path:
                    return json.dumps({
                        "success": False,
                        "error": "No task initialized"
                    })
                
                if not server_ref.task_id:
                    return json.dumps({
                        "success": False,
                        "error": "No task_id set"
                    })
                
                # First copy test configuration to container
                tests_config = server_ref.task_path / "tests" / "config.json"
                if tests_config.exists():
                    with open(tests_config) as f:
                        config_content = f.read()
                    server_ref._docker_exec("mkdir -p /tests")
                    server_ref._docker_write_file("/tests/config.json", config_content)
                
                # Run test script
                run_tests_sh = server_ref.task_path / "run-tests.sh"
                if not run_tests_sh.exists():
                    return json.dumps({
                        "success": False,
                        "error": f"run-tests.sh not found in {server_ref.task_path}"
                    })
                
                with open(run_tests_sh) as f:
                    test_script = f.read()
                
                # Execute test script
                result = server_ref._docker_exec(
                    f"cd {server_ref.workspace} && bash << 'HEREDOC_END'\n{test_script}\nHEREDOC_END",
                    timeout=1800  # 30 minute timeout
                )
                
                test_output = result["output"]
                exit_code = result["exit_code"]
                
                # Try to parse results using swebench library (consistent with original benchmark)
                resolved = False
                tests_status = {}
                report = {}
                
                try:
                    from swebench.harness.grading import get_eval_report
                    from swebench.harness.test_spec.test_spec import make_test_spec
                    from swebench.harness.utils import load_swebench_dataset
                    
                    # Load instance info
                    dataset_path = SWEBENCH_DATASET_PATH
                    task_json = server_ref.task_path / "task.json"
                    
                    if task_json.exists():
                        with open(task_json) as f:
                            instance = json.load(f)
                    else:
                        # Try to load from HuggingFace dataset
                        dataset = load_swebench_dataset('princeton-nlp/SWE-bench_Verified', 'test')
                        instance = None
                        for item in dataset:
                            if item['instance_id'] == server_ref.task_id:
                                instance = item
                                break
                        
                        if not instance:
                            raise ValueError(f"Instance {server_ref.task_id} not found in dataset")
                    
                    # Create test_spec
                    test_spec = make_test_spec(instance)
                    
                    # Get patch
                    diff_result = server_ref._docker_exec(
                        f"cd {server_ref.workspace} && git diff --no-color HEAD",
                        timeout=60
                    )
                    patch = diff_result["output"]
                    if not patch.strip():
                        diff_result = server_ref._docker_exec(
                            f"cd {server_ref.workspace} && git diff --no-color --cached",
                            timeout=60
                        )
                        patch = diff_result["output"]
                    
                    # Save test output to temp file
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        log_dir = Path(temp_dir) / 'logs' / server_ref.task_id.lower()
                        log_dir.mkdir(parents=True, exist_ok=True)
                        test_output_path = log_dir / 'test_output.txt'
                        test_output_path.write_text(test_output)
                        
                        # Use swebench's get_eval_report to parse results
                        eval_report = get_eval_report(
                            test_spec=test_spec,
                            prediction={
                                'model_patch': patch,
                                'instance_id': server_ref.task_id,
                            },
                            include_tests_status=True,
                            test_log_path=str(test_output_path),
                        )
                        
                        if server_ref.task_id in eval_report:
                            instance_report = eval_report[server_ref.task_id]
                            resolved = instance_report.get('resolved', False)
                            
                            # Extract tests_status (containing detailed FAIL_TO_PASS and PASS_TO_PASS status)
                            if 'tests_status' in instance_report:
                                tests_status = instance_report['tests_status']
                            
                            # Build complete report
                            report = {
                                'resolved': resolved,
                                'patch_exists': instance_report.get('patch_exists', bool(patch)),
                                'patch_successfully_applied': instance_report.get('patch_successfully_applied', True),
                                'tests_status': tests_status,
                            }
                            
                            logger.info(f"[SWE-Bench] Evaluation result: resolved={resolved}")
                            logger.info(f"[SWE-Bench] tests_status keys: {list(tests_status.keys()) if tests_status else 'none'}")
                
                except ImportError:
                    logger.warning("swebench library not available, using simple pass/fail detection")
                    # Fall back to simple detection
                    resolved = "PASSED" in test_output and "SWEBench results" in test_output
                    report = {
                        'resolved': resolved,
                        'fallback_detection': True,
                    }
                    
                except Exception as e:
                    logger.warning(f"Error using swebench grading: {e}, using simple detection")
                    resolved = "PASSED" in test_output and "SWEBench results" in test_output
                    report = {
                        'resolved': resolved,
                        'fallback_detection': True,
                        'grading_error': str(e),
                    }
                
                return json.dumps({
                    "success": True,
                    "passed": resolved,
                    "resolved": resolved,
                    "tests_status": tests_status,  # {"FAIL_TO_PASS": {...}, "PASS_TO_PASS": {...}}
                    "report": report,
                    "output": test_output[-10000:] if len(test_output) > 10000 else test_output,
                    "exit_code": exit_code
                })
                
            except Exception as e:
                logger.error(f"Error running tests: {e}")
                import traceback
                traceback.print_exc()
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        # ============================================================================
        # Internal tool: Get patch (prefixed with __, not visible to Agent)
        # ============================================================================
        
        @self.mcp.tool(name="__swebench_get_patch")
        def get_patch() -> str:
            """
            [Internal] Get the git diff patch of all changes.
            
            Returns:
                JSON with the patch content
            """
            try:
                diff_result = server_ref._docker_exec(
                    f"cd {server_ref.workspace} && git diff --no-color HEAD",
                    timeout=60
                )
                patch = diff_result["output"]
                
                if not patch.strip():
                    diff_result = server_ref._docker_exec(
                        f"cd {server_ref.workspace} && git diff --no-color --cached",
                        timeout=60
                    )
                    patch = diff_result["output"]
                
                return json.dumps({
                    "success": True,
                    "patch": patch,
                    "length": len(patch)
                })
                
            except Exception as e:
                logger.error(f"Error getting patch: {e}")
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
            
            This stops and removes the Docker container.
            
            Returns:
                JSON status
            """
            try:
                if server_ref.task_path and server_ref.project_name:
                    docker_compose_path = server_ref.task_path / "docker-compose.yaml"
                    
                    # Stop and remove container (using project_name)
                    cmd = [
                        "docker", "compose", "-f", str(docker_compose_path),
                        "-p", server_ref.project_name,
                        "down", "-v"
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    logger.info(f"[SWEBench] docker compose down result: {result.returncode}")
                    
                    # Also try to remove container directly
                    if server_ref.container_name:
                        subprocess.run(
                            ["docker", "rm", "-f", server_ref.container_name],
                            capture_output=True, timeout=30
                        )
                
                server_ref.container_running = False
                server_ref.container_name = None
                server_ref.project_name = None
                server_ref.task_id = None
                server_ref.task_path = None
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
_default_server = SWEBenchDockerServer()

# Export MCP instance
mcp = _default_server.mcp

# Exported tool names (only 3 tools visible to Agent)
EXPOSED_TOOLS = [
    "swebench_execute_bash",
    "swebench_str_replace_editor",
    "swebench_finish",
]

# Internal tools (not visible to Agent, called by run.py orchestrator)
INTERNAL_TOOLS = [
    "__swebench_switch_container",  # Switch Docker containers (replaces the old init_task)
    "__swebench_run_tests",
    "__swebench_get_patch",
    "__swebench_cleanup",
]

# All tools
ALL_TOOLS = EXPOSED_TOOLS + INTERNAL_TOOLS


def get_server() -> SWEBenchDockerServer:
    """Get default server instance"""
    return _default_server


def create_server() -> SWEBenchDockerServer:
    """
    Create a new server instance
    
    Note: No longer accepts task_id parameter.
    Task switching is done via the __swebench_switch_container tool.
    """
    return SWEBenchDockerServer()


def main():
    """
    MCP Server entry point
    
    This function allows swebench_server to run as an independent MCP process:
    python -m source.servers.swebench_server
    
    Architecture notes:
    - This Server runs as a long-running MCP process
    - Switches Docker containers between tasks via __swebench_switch_container
    - run.py (orchestrator) calls this Server via MCP protocol
    """
    import sys
    
    print(f"[{datetime.now()}] Starting SWEBench MCP Server (Docker Bridge Mode)...", file=sys.stderr)
    print(f"Exposed tools (Agent visible): {EXPOSED_TOOLS}", file=sys.stderr)
    print(f"Internal tools (run.py only): {INTERNAL_TOOLS}", file=sys.stderr)
    print(f"SWE-Bench dataset path: {SWEBENCH_DATASET_PATH}", file=sys.stderr)
    
    # Run MCP server (using stdio communication)
    mcp.run()
    
    print(f"[{datetime.now()}] SWEBench MCP Server finished.", file=sys.stderr)


if __name__ == "__main__":
    main()
