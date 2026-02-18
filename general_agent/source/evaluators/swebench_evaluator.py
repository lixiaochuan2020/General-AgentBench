"""SWE-Bench Evaluator - Directly calls OpenHands original evaluation code

Fully reuses OpenHands evaluation flow:
- OpenHands/evaluation/benchmarks/swe_bench/eval_infer.py: process_instance
- swebench.harness.grading.get_eval_report: evaluation results

Evaluation flow (fully consistent with original benchmark):
1. Extract agent-generated patch from trace
2. Apply patch using OpenHands Runtime (Docker)
3. Run SWE-Bench eval_script tests
4. Parse test results using swebench library to determine if resolved
"""

import os
import sys
import json
import tempfile
import subprocess
import time
import copy
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# Add OpenHands to path
OPENHANDS_PATH = Path(__file__).parent.parent.parent.parent / "benchmarks" / "OpenHands"
if str(OPENHANDS_PATH) not in sys.path:
    sys.path.insert(0, str(OPENHANDS_PATH))


@dataclass
class SWEBenchEvalResult:
    """SWE-Bench evaluation result - consistent with OpenHands format
    
    Contains all fields from original benchmark:
    - report: evaluation report summary
    - tests_status: detailed status of each test (FAIL_TO_PASS, PASS_TO_PASS success/failure lists)
    """
    resolved: bool = False
    failed_apply_patch: bool = False
    empty_generation: bool = False
    error_eval: bool = False
    test_timeout: bool = False
    test_output: str = ""
    apply_patch_output: str = ""
    error_message: str = ""
    
    # OpenHands compatible fields
    report: dict = field(default_factory=lambda: {
        'empty_generation': False,
        'resolved': False,
        'failed_apply_patch': False,
        'error_eval': False,
        'test_timeout': False,
    })
    
    # Detailed status of each test (consistent with original SWE-Bench)
    # Format: {"FAIL_TO_PASS": {"success": [...], "failure": [...]}, 
    #          "PASS_TO_PASS": {"success": [...], "failure": [...]}}
    tests_status: dict = field(default_factory=dict)


def _check_openhands_available() -> bool:
    """Check if OpenHands is available"""
    try:
        from openhands.core.main import create_runtime
        from openhands.events.action import CmdRunAction
        return True
    except ImportError:
        return False


def _check_swebench_available() -> bool:
    """Check if swebench is available"""
    try:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec, SWEbenchInstance
        from swebench.harness.utils import load_swebench_dataset
        return True
    except ImportError:
        return False


class SWEBenchEvaluator:
    """
    SWE-Bench Evaluator - Directly calls OpenHands original evaluation
    
    This fully reuses the original benchmark evaluator, not a simplified version.
    
    Evaluation requirements:
    1. OpenHands library available
    2. swebench library available  
    3. Docker environment
    4. SWE-Bench dataset (for obtaining test_spec)
    """
    
    # Docker image prefix
    DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
    
    def __init__(self, timeout: int = 1800, dataset_name: str = 'princeton-nlp/SWE-bench', split: str = 'test'):
        """
        Initialize evaluator
        
        Args:
            timeout: Test timeout in seconds
            dataset_name: SWE-Bench dataset name
            split: Dataset split
        """
        self.timeout = timeout
        self.dataset_name = dataset_name
        self.split = split
        
        self._openhands_available = _check_openhands_available()
        self._swebench_available = _check_swebench_available()
        
        if not self._openhands_available:
            logger.warning("OpenHands library not installed, full evaluation not available")
        if not self._swebench_available:
            logger.warning("swebench library not installed, full evaluation not available")
        
        # Lazy-load dataset
        self._dataset = None
        self._instance_id_to_instance = None
        self._conditional_imports = None
    
    def _load_dataset(self):
        """Lazy-load SWE-Bench dataset"""
        if self._dataset is not None:
            return
        
        if not self._swebench_available:
            raise RuntimeError("swebench library not installed, cannot load dataset")
        
        from swebench.harness.grading import get_eval_report
        from swebench.harness.run_evaluation import APPLY_PATCH_FAIL, APPLY_PATCH_PASS
        from swebench.harness.test_spec.test_spec import SWEbenchInstance, make_test_spec
        from swebench.harness.utils import load_swebench_dataset
        
        logger.info(f"Loading SWE-Bench dataset: {self.dataset_name} ({self.split})")
        self._dataset = load_swebench_dataset(self.dataset_name, self.split)
        self._instance_id_to_instance = {
            instance['instance_id']: instance for instance in self._dataset
        }
        
        # Save imported functions
        self._get_eval_report = get_eval_report
        self._make_test_spec = make_test_spec
        self._APPLY_PATCH_FAIL = APPLY_PATCH_FAIL
        self._APPLY_PATCH_PASS = APPLY_PATCH_PASS
        
        logger.info(f"Dataset loaded, {len(self._dataset)} instances total")
    
    def extract_patch_from_trace(self, trace: dict) -> str:
        """
        Extract agent-generated patch from trace
        
        Search by priority:
        1. patch field in JSON returned by swebench_finish tool
        2. patch field in swebench_finish tool arguments
        3. Content starting with diff in tool results
        4. diff --git content in final_response
        """
        messages = trace.get('messages', [])
        patch = ""
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            # 1. Find tool return result from swebench_finish (containing patch field)
            if msg.get('role') == 'tool':
                tool_name = str(msg.get('name', '')).lower()
                if 'finish' in tool_name and 'swebench' in tool_name:
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        try:
                            result = json.loads(content)
                            if isinstance(result, dict) and result.get('patch'):
                                patch = result['patch']
                                if patch.strip():
                                    logger.info(f"Extracted patch from swebench_finish return value: {len(patch)} chars")
                                    return self._process_git_patch(patch)
                        except json.JSONDecodeError:
                            pass
            
            # 2. Find patch in tool call arguments
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                for tc in msg.get('tool_calls', []):
                    if isinstance(tc, dict) and 'function' in tc:
                        func_name = tc['function'].get('name', '')
                        if 'finish' in func_name.lower():
                            # Try to parse arguments
                            try:
                                args = tc['function'].get('arguments', '{}')
                                if isinstance(args, str):
                                    args = json.loads(args)
                                p = args.get('patch', '') or args.get('result', '')
                                if p and p.strip():
                                    patch = p
                            except:
                                pass
            
            # 3. Find patch in tool results
            if msg.get('role') == 'tool' and 'swebench' in str(msg.get('name', '')).lower():
                content = msg.get('content', '')
                if isinstance(content, str) and content.startswith('diff '):
                    patch = content
        
        # 4. Check for diff in final_response
        final_response = trace.get('final_response', '')
        if not patch and final_response:
            # Try to extract patch from final_response
            if 'diff --git' in final_response:
                lines = final_response.split('\n')
                in_patch = False
                patch_lines = []
                for line in lines:
                    if line.startswith('diff --git'):
                        in_patch = True
                    if in_patch:
                        patch_lines.append(line)
                if patch_lines:
                    patch = '\n'.join(patch_lines)
        
        # 5. Check tool_result in steps (alternative trace format)
        if not patch:
            steps = trace.get('steps', [])
            for step in steps:
                if isinstance(step, dict):
                    tool_name = step.get('tool_name', '') or step.get('tool', '')
                    if 'finish' in str(tool_name).lower() and 'swebench' in str(tool_name).lower():
                        tool_result = step.get('tool_result', '')
                        if isinstance(tool_result, str):
                            try:
                                result = json.loads(tool_result)
                                if isinstance(result, dict) and result.get('patch'):
                                    patch = result['patch']
                                    if patch.strip():
                                        logger.info(f"Extracted patch from steps: {len(patch)} chars")
                                        return self._process_git_patch(patch)
                            except json.JSONDecodeError:
                                pass
        
        return self._process_git_patch(patch)
    
    def _process_git_patch(self, patch: str) -> str:
        """
        Process git patch, clean up format
        
        Reused from OpenHands/evaluation/benchmarks/swe_bench/eval_infer.py
        """
        if not isinstance(patch, str):
            return ''

        if not patch.strip():
            return ''

        patch = patch.replace('\r\n', '\n')
        
        # Find the first "diff --git" line, remove everything before it
        lines = patch.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('diff --git'):
                patch = '\n'.join(lines[i:])
                break

        patch = patch.rstrip() + '\n'
        return patch
    
    def _get_docker_image(self, instance_id: str) -> str:
        """Get Docker image for the instance
        
        Reused from OpenHands/evaluation/benchmarks/swe_bench/run_infer.py
        """
        # Try using OpenHands function
        try:
            from evaluation.benchmarks.swe_bench.run_infer import get_instance_docker_image
            return get_instance_docker_image(instance_id)
        except ImportError:
            pass
        
        # Fall back to manual construction
        # Format: {prefix}swe-bench-eval-{repo_name}:{instance_id}
        parts = instance_id.split('__')
        if len(parts) >= 2:
            repo_name = parts[0].replace('-', '_')
            return f"{self.DOCKER_IMAGE_PREFIX}swe-bench-eval-{repo_name}:{instance_id}"
        return f"{self.DOCKER_IMAGE_PREFIX}swe-bench-eval:latest"
    
    def evaluate_with_openhands(
        self, 
        trace: dict, 
        task_data: dict,
        instance_id: str,
    ) -> SWEBenchEvalResult:
        """
        Use OpenHands original evaluation flow
        
        Directly calls OpenHands process_instance function
        """
        result = SWEBenchEvalResult()
        
        # Extract patch
        patch = self.extract_patch_from_trace(trace)
        
        if not patch:
            result.empty_generation = True
            result.report['empty_generation'] = True
            return result
        
        if not self._openhands_available or not self._swebench_available:
            logger.warning("OpenHands or swebench not available, falling back to Docker evaluation")
            return self.evaluate_with_docker(trace, task_data, instance_id)
        
        try:
            # Load dataset
            self._load_dataset()
            
            # Get instance info
            if instance_id not in self._instance_id_to_instance:
                logger.error(f"Instance {instance_id} not in dataset")
                result.error_eval = True
                result.error_message = f"Instance {instance_id} not found in dataset"
                return result
            
            instance = self._instance_id_to_instance[instance_id]
            test_spec = self._make_test_spec(instance)
            
            # Import OpenHands evaluation functions
            from openhands.core.main import create_runtime
            from openhands.events.action import CmdRunAction
            from openhands.events.observation import CmdOutputObservation
            from openhands.utils.async_utils import call_async_from_sync
            from evaluation.benchmarks.swe_bench.run_infer import get_instance_docker_image
            from evaluation.utils.shared import get_default_sandbox_config_for_eval, get_openhands_config_for_eval
            
            # Configure runtime
            base_container_image = get_instance_docker_image(instance_id)
            sandbox_config = get_default_sandbox_config_for_eval()
            sandbox_config.base_container_image = base_container_image
            config = get_openhands_config_for_eval(
                runtime=os.environ.get('RUNTIME', 'docker'),
                sandbox_config=sandbox_config,
            )
            
            # Create runtime
            runtime = create_runtime(config)
            call_async_from_sync(runtime.connect)
            
            try:
                # Copy patch file
                with tempfile.TemporaryDirectory() as temp_dir:
                    patch_file_path = os.path.join(temp_dir, 'patch.diff')
                    with open(patch_file_path, 'w') as f:
                        f.write(patch)
                    runtime.copy_to(patch_file_path, '/tmp')
                    
                    # Copy eval script
                    eval_script_path = os.path.join(temp_dir, 'eval.sh')
                    with open(eval_script_path, 'w') as f:
                        f.write(test_spec.eval_script)
                    runtime.copy_to(eval_script_path, '/tmp')
                
                # Set execution permissions
                action = CmdRunAction(command='chmod +x /tmp/eval.sh')
                action.set_hard_timeout(600)
                obs = runtime.run_action(action)
                
                # Apply patch
                exec_command = (
                    'cd /testbed && '
                    "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                    "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
                    "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                    "echo 'APPLY_PATCH_FAIL')))"
                )
                action = CmdRunAction(command=exec_command)
                action.set_hard_timeout(600)
                obs = runtime.run_action(action)
                
                if isinstance(obs, CmdOutputObservation):
                    result.apply_patch_output = obs.content
                    
                    if 'APPLY_PATCH_FAIL' in obs.content:
                        result.failed_apply_patch = True
                        result.report['failed_apply_patch'] = True
                        return result
                
                # Run tests
                log_file = '/tmp/eval_output.log'
                action = CmdRunAction(command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!')
                action.set_hard_timeout(300)
                obs = runtime.run_action(action)
                
                if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                    pid = obs.content.split()[-1].strip()
                    
                    # Wait for test completion
                    start_time = time.time()
                    while True:
                        if time.time() - start_time > self.timeout:
                            result.test_timeout = True
                            result.report['test_timeout'] = True
                            break
                        
                        check_action = CmdRunAction(command=f'ps -p {pid} > /dev/null; echo $?')
                        check_action.set_hard_timeout(300)
                        check_obs = runtime.run_action(check_action)
                        
                        if isinstance(check_obs, CmdOutputObservation) and check_obs.content.split()[-1].strip() == '1':
                            break
                        
                        time.sleep(30)
                    
                    # Read test output
                    cat_action = CmdRunAction(command=f'cat {log_file}')
                    cat_action.set_hard_timeout(300)
                    cat_obs = runtime.run_action(cat_action)
                    
                    if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                        result.test_output = cat_obs.content
                        
                        # Parse results using swebench
                        with tempfile.TemporaryDirectory() as temp_dir:
                            log_dir = os.path.join(temp_dir, 'logs', instance_id.lower())
                            os.makedirs(log_dir, exist_ok=True)
                            test_output_path = os.path.join(log_dir, 'test_output.txt')
                            with open(test_output_path, 'w') as f:
                                f.write(result.test_output)
                            
                            try:
                                report = self._get_eval_report(
                                    test_spec=test_spec,
                                    prediction={
                                        'model_patch': patch,
                                        'instance_id': instance_id,
                                    },
                                    include_tests_status=True,
                                    test_log_path=test_output_path,
                                )
                                
                                if instance_id in report:
                                    instance_report = report[instance_id]
                                    result.resolved = instance_report.get('resolved', False)
                                    result.report['resolved'] = result.resolved
                                    
                                    # Save complete tests_status (detailed status of each test)
                                    if 'tests_status' in instance_report:
                                        result.tests_status = instance_report['tests_status']
                                        result.report['tests_status'] = instance_report['tests_status']
                                    
                                    # Save other useful fields
                                    for key in ['patch_exists', 'patch_successfully_applied']:
                                        if key in instance_report:
                                            result.report[key] = instance_report[key]
                                    
                            except Exception as e:
                                logger.error(f"Failed to parse test output: {e}")
                                result.error_eval = True
                                result.error_message = str(e)
                else:
                    result.error_eval = True
                    result.error_message = "Failed to start evaluation"
                    
            finally:
                runtime.close()
                
        except Exception as e:
            logger.error(f"OpenHands evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            result.error_eval = True
            result.error_message = str(e)
            result.report['error_eval'] = True
        
        return result
    
    def evaluate_with_docker(
        self, 
        trace: dict, 
        task_data: dict,
        instance_id: str,
    ) -> SWEBenchEvalResult:
        """
        Evaluate using Docker (without OpenHands Runtime dependency)
        
        This is the fallback when OpenHands is not available,
        still using swebench's get_eval_report to parse results
        """
        result = SWEBenchEvalResult()
        
        # Extract patch
        patch = self.extract_patch_from_trace(trace)
        
        if not patch:
            result.empty_generation = True
            result.report['empty_generation'] = True
            return result
        
        if not self._swebench_available:
            logger.error("swebench library not installed, cannot evaluate")
            result.error_eval = True
            result.error_message = "swebench library not installed"
            return result
        
        try:
            # Load dataset
            self._load_dataset()
            
            # Get instance info
            if instance_id not in self._instance_id_to_instance:
                logger.error(f"Instance {instance_id} not in dataset")
                result.error_eval = True
                result.error_message = f"Instance {instance_id} not found in dataset"
                return result
            
            instance = self._instance_id_to_instance[instance_id]
            test_spec = self._make_test_spec(instance)
            
            # Get Docker image name
            docker_image = self._get_docker_image(instance_id)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                patch_file = temp_path / 'patch.diff'
                patch_file.write_text(patch)
                
                eval_script_file = temp_path / 'eval.sh'
                eval_script_file.write_text(test_spec.eval_script)
                
                # Create and run Docker container
                container_name = f"swebench_eval_{instance_id.replace('/', '_').replace('__', '_')[:30]}"
                
                # Start container
                start_cmd = [
                    'docker', 'run', '-d', '--name', container_name,
                    '-v', f'{patch_file}:/tmp/patch.diff:ro',
                    '-v', f'{eval_script_file}:/tmp/eval.sh:ro',
                    docker_image,
                    'sleep', 'infinity'
                ]
                
                try:
                    subprocess.run(start_cmd, check=True, capture_output=True, timeout=120)
                except subprocess.CalledProcessError as e:
                    result.error_eval = True
                    result.error_message = f"Failed to start container: {e.stderr.decode()}"
                    return result
                except FileNotFoundError:
                    result.error_eval = True
                    result.error_message = "Docker not found"
                    return result
                
                try:
                    # Set execution permissions
                    subprocess.run(
                        ['docker', 'exec', container_name, 'chmod', '+x', '/tmp/eval.sh'],
                        capture_output=True, timeout=60
                    )
                    
                    # Apply patch
                    apply_cmd = [
                        'docker', 'exec', container_name, 'bash', '-c',
                        "cd /testbed && "
                        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                        "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
                        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                        "echo 'APPLY_PATCH_FAIL')))"
                    ]
                    
                    apply_result = subprocess.run(
                        apply_cmd, capture_output=True, text=True, timeout=600
                    )
                    result.apply_patch_output = apply_result.stdout
                    
                    if 'APPLY_PATCH_FAIL' in apply_result.stdout:
                        result.failed_apply_patch = True
                        result.report['failed_apply_patch'] = True
                        return result
                    
                    # Run tests
                    test_cmd = ['docker', 'exec', container_name, 'bash', '/tmp/eval.sh']
                    
                    try:
                        test_result = subprocess.run(
                            test_cmd, capture_output=True, text=True, timeout=self.timeout
                        )
                        result.test_output = test_result.stdout + test_result.stderr
                    except subprocess.TimeoutExpired:
                        result.test_timeout = True
                        result.report['test_timeout'] = True
                        return result
                    
                    # Parse results using swebench
                    log_dir = temp_path / 'logs' / instance_id.lower()
                    log_dir.mkdir(parents=True, exist_ok=True)
                    test_output_path = log_dir / 'test_output.txt'
                    test_output_path.write_text(result.test_output)
                    
                    try:
                        report = self._get_eval_report(
                            test_spec=test_spec,
                            prediction={
                                'model_patch': patch,
                                'instance_id': instance_id,
                            },
                            include_tests_status=True,
                            test_log_path=str(test_output_path),
                        )
                        
                        if instance_id in report:
                            instance_report = report[instance_id]
                            result.resolved = instance_report.get('resolved', False)
                            result.report['resolved'] = result.resolved
                            
                            # Save complete tests_status (detailed status of each test)
                            if 'tests_status' in instance_report:
                                result.tests_status = instance_report['tests_status']
                                result.report['tests_status'] = instance_report['tests_status']
                            
                            # Save other useful fields
                            for key in ['patch_exists', 'patch_successfully_applied']:
                                if key in instance_report:
                                    result.report[key] = instance_report[key]
                            
                    except Exception as e:
                        logger.error(f"Failed to parse test output: {e}")
                        result.error_eval = True
                        result.error_message = str(e)
                    
                finally:
                    # Clean up container
                    subprocess.run(
                        ['docker', 'rm', '-f', container_name],
                        capture_output=True, timeout=30
                    )
                    
        except Exception as e:
            logger.error(f"Docker evaluation failed: {e}")
            result.error_eval = True
            result.error_message = str(e)
        
        return result
    
    def evaluate(
        self,
        trace: dict,
        task_data: dict,
        instance_id: str = None,
    ) -> tuple[float, bool, dict]:
        """
        Evaluation entry point - uses the exact same evaluation flow as OpenHands
        
        Returns:
            (reward, success, details)
            - reward: 0.0 or 1.0
            - success: whether successful
            - details: detailed info (consistent with OpenHands format)
        """
        instance_id = instance_id or task_data.get('instance_id', 'unknown')
        
        # Prefer OpenHands evaluation
        if self._openhands_available and self._swebench_available:
            result = self.evaluate_with_openhands(trace, task_data, instance_id)
        elif self._swebench_available:
            result = self.evaluate_with_docker(trace, task_data, instance_id)
        else:
            # Cannot evaluate
            logger.error("Cannot evaluate: swebench library required")
            return 0.0, False, {
                'error': 'swebench library not installed',
                'resolved': False,
            }
        
        # Calculate reward (consistent with OpenHands)
        if result.resolved:
            reward = 1.0
            success = True
        else:
            reward = 0.0
            success = False
        
        # Return detailed info (consistent with OpenHands format, contains all original benchmark fields)
        details = {
            'report': result.report,
            'resolved': result.resolved,
            'failed_apply_patch': result.failed_apply_patch,
            'empty_generation': result.empty_generation,
            'error_eval': result.error_eval,
            'test_timeout': result.test_timeout,
            'error_message': result.error_message,
            'test_output': result.test_output[:5000] if result.test_output else "",  # Truncate to avoid excessive length
            'apply_patch_output': result.apply_patch_output[:2000] if result.apply_patch_output else "",
            # Detailed status of each test (consistent with original SWE-Bench format)
            # Format: {"FAIL_TO_PASS": {"success": [...], "failure": [...]}, 
            #          "PASS_TO_PASS": {"success": [...], "failure": [...]}}
            'tests_status': result.tests_status,
        }
        
        return reward, success, details

    def evaluate_from_server_result(
        self,
        test_result: dict,
        patch: str = "",
        instance_id: str = "",
    ) -> tuple[float, bool, dict]:
        """
        Evaluate from SWEBenchDockerServer test results
        
        This is the evaluation entry point for the new Docker bridge mode.
        The server has already run the tests in a Docker container,
        we only need to extract info from the results.
        
        Args:
            test_result: JSON parsed result returned by __swebench_run_tests
            patch: agent-generated patch
            instance_id: task instance ID
            
        Returns:
            (reward, success, details)
        """
        # Extract test results
        success = test_result.get("success", False)
        passed = test_result.get("passed", False)
        test_output = test_result.get("output", "")
        error = test_result.get("error", "")
        
        # Determine if resolved
        resolved = success and passed
        
        # Calculate reward
        reward = 1.0 if resolved else 0.0
        
        # Build details (compatible with original format)
        details = {
            'report': {
                'empty_generation': not patch or not patch.strip(),
                'resolved': resolved,
                'failed_apply_patch': False,  # In new architecture, patch is applied at runtime
                'error_eval': not success,
                'test_timeout': "timeout" in test_output.lower() if test_output else False,
            },
            'resolved': resolved,
            'failed_apply_patch': False,
            'empty_generation': not patch or not patch.strip(),
            'error_eval': not success,
            'test_timeout': False,
            'error_message': error,
            'test_output': test_output[:5000] if test_output else "",
            'apply_patch_output': "",
            'tests_status': {},  # Could be parsed from test_output, but simplified here
            'instance_id': instance_id,
            'patch_length': len(patch) if patch else 0,
        }
        
        return reward, resolved, details
