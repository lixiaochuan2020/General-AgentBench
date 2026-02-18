#!/usr/bin/env python3
"""
MCP-Benchmark Unified Entry Point

All-in-One Agent System:
- Uses Universal Agent (sees all tools from all benchmarks)
- Uses native evaluators from each benchmark (for easy comparison)
- tau2 tasks use tau2's UserSimulator

Run examples:
    # Load mixed tasks from JSON file
    python run.py --task-file data/tau2_multi_domain_test.json --model gpt-4o

    # Run a specific tau2 domain
    python run.py --benchmark tau2 --domain airline --model gpt-4o

    # Run all tau2 domains
    python run.py --benchmark tau2 --model gpt-4o
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from loguru import logger
import sys as _sys

# Configure loguru default output level to INFO
logger.remove()  # Remove default handler
logger.add(_sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Add tau2-bench to path
TAU2_BENCH_PATH = Path(__file__).parent.parent / "benchmarks" / "tau2-bench" / "src"
if str(TAU2_BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(TAU2_BENCH_PATH))

# Add MathHay to path (for directly calling original benchmark functions)
MATHHAY_PATH = Path(__file__).parent.parent / "benchmarks" / "MathHay"
if str(MATHHAY_PATH) not in sys.path:
    sys.path.insert(0, str(MATHHAY_PATH))

# Import evaluators
from source.evaluators import SWEBenchEvaluator, TerminalBenchEvaluator

from source.host import BenchmarkHost
from source.agent import UniversalAgent, ToolCall as AgentToolCall
from source.native_evaluators import Tau2Evaluator, MCPBenchEvaluator, SearchEvaluator, MathHayEvaluator, NativeEvalResult
from source.llm_api import LiteLLMAPI, OpenAIAPI

# Sequential Scaling module
from source.scaling import ScalingConfig, ScalingCheckpoint, CheckpointStore, ScalingController

# Import configuration module
from source.config import (
    get_temperature,
    get_max_execution_rounds,
    get_task_timeout,
    get_task_delay,
    get_judge_model,
)

# mcp-bench related imports
from source.mcpbench_tasks import (
    load_mcpbench_tasks, 
    load_single_server_tasks,
    MCPBenchTask,
)
from source.servers.mcpbench_server import (
    get_servers_for_task,
    get_simple_test_servers,
    list_available_servers,
    get_mcpbench_server_configs,
    convert_to_host_config,
)

# tau2 related imports
from tau2.user.user_simulator import UserSimulator as Tau2UserSimulator
from tau2.user.base import STOP, TRANSFER, OUT_OF_SCOPE
from tau2.data_model.message import AssistantMessage
from tau2.registry import registry as tau2_registry


# Server configuration
DEFAULT_SERVERS = {
    "tau2-airline": {
        "command": sys.executable,
        "args": ["-m", "source.servers.tau2_server", "--domain", "airline"]
    },
    "tau2-retail": {
        "command": sys.executable,
        "args": ["-m", "source.servers.tau2_server", "--domain", "retail"]
    },
    "tau2-telecom": {
        "command": sys.executable,
        "args": ["-m", "source.servers.tau2_server", "--domain", "telecom"]
    },
    # Unified Search Server (supports browsecomp, mind2web, webvoyager)
    "search": {
        "command": sys.executable,
        "args": ["-m", "source.servers.search_server", "--search-engine", "serper"]
    },
    # MathHay Server (long-context math reasoning - does not actually use tools, only for framework consistency)
    "mathhay": {
        "command": sys.executable,
        "args": ["-m", "source.servers.mathhay_server"]
    },
    # TerminalBench Server (Docker Compose + Tmux runtime for terminal tasks)
    "terminalbench": {
        "command": sys.executable,
        "args": ["-m", "source.servers.terminalbench_server"]
    },
    # SWE-Bench Server (Docker Container Bridge Mode)
    # Runs as a standalone MCP process, switches task containers via __swebench_switch_container
    "swebench": {
        "command": sys.executable,
        "args": ["-m", "source.servers.swebench_server"]
    },
}


def get_all_servers() -> dict[str, dict]:
    """
    Get all server configurations (tau2 + mcp-bench)
    
    Returns:
        Configuration dictionary for all servers
    """
    all_servers = {}
    
    # 1. Add tau2 servers (3)
    all_servers.update(DEFAULT_SERVERS)
    
    # 2. Add mcp-bench servers (28)
    # Use native server names (e.g. BioMCP, SWEBench), tools use Server__tool format for distinction (Bedrock compatible)
    mcpbench_configs = get_mcpbench_server_configs()
    for name, raw_config in mcpbench_configs.items():
        host_config = convert_to_host_config(name, raw_config)
        if host_config:
            all_servers[name] = host_config
    
    return all_servers


def get_mcpbench_servers_for_domain(domain: str) -> list[str]:
    """
    Parse task domain and return the required server list
    
    Args:
        domain: Task domain, e.g. "BioMCP" or "Scientific Computing+BioMCP+Math MCP"
        
    Returns:
        List of server names (corresponding to the name in mcpbench-{name} format)
    """
    # domain format: "Server1" or "Server1+Server2+Server3"
    # Split by " + " (note the spaces)
    if " + " in domain:
        return [s.strip() for s in domain.split(" + ")]
    elif "+" in domain:
        return [s.strip() for s in domain.split("+")]
    else:
        return [domain.strip()]


def get_mcpbench_server_tool_counts() -> dict[str, int]:
    """
    Get the tool count for each MCP-Bench server
    
    Returns:
        {server_name: tool_count} dictionary
    """
    import json
    mcp_servers_info_path = Path(__file__).parent.parent / "benchmarks" / "mcp-bench" / "mcp_servers_info.json"
    
    if not mcp_servers_info_path.exists():
        logger.warning(f"mcp_servers_info.json not found at {mcp_servers_info_path}, using defaults")
        # Return hardcoded default values (from documentation)
        return {
            "OpenAPI Explorer": 2, "Unit Converter": 16, "Wikipedia": 9, "Google Maps": 7,
            "Bibliomantic": 4, "BioMCP": 35, "Call for Papers": 1, "Car Price Evaluator": 3,
            "Context7": 2, "DEX Paprika": 11, "FruityVice": 1, "Game Trends": 7, "Huge Icons": 3,
            "Hugging Face": 10, "Math MCP": 13, "NixOS": 18, "OSINT Intelligence": 7, "Reddit": 2,
            "National Parks": 6, "Medical Calculator": 22, "Metropolitan Museum": 3, "Movie Recommender": 1,
            "NASA Data": 21, "OKX Exchange": 2, "Paper Search": 19, "Scientific Computing": 26,
            "Weather Data": 4, "Time MCP": 2,
        }
    
    with open(mcp_servers_info_path) as f:
        data = json.load(f)
    
    servers = data.get("servers", {})
    return {name: len(info.get("tools", [])) for name, info in servers.items()}


def calculate_dynamic_distraction_count(
    domain: str,
    max_total_tools: int = 125,
) -> int:
    """
    Calculate dynamic distraction tool count for GPT-5 mode
    
    Args:
        domain: Task domain (e.g. "BioMCP" or "Scientific Computing+BioMCP+Math MCP")
        max_total_tools: Maximum total tools (default 125, leaving 3 as buffer)
        
    Returns:
        Number of available distraction tools
    """
    # Get server tool count information
    server_tool_counts = get_mcpbench_server_tool_counts()
    
    # Parse domain to get required servers
    required_servers = get_mcpbench_servers_for_domain(domain)
    
    # Calculate required tool count
    required_tools_count = 0
    for server in required_servers:
        tool_count = server_tool_counts.get(server, 0)
        if tool_count == 0:
            logger.warning(f"Unknown server '{server}' in domain '{domain}', assuming 10 tools")
            tool_count = 10
        required_tools_count += tool_count
    
    # Calculate available distraction count
    distraction_count = max(0, max_total_tools - required_tools_count)
    
    logger.info(f"GPT-5 mode: domain='{domain}', required_tools={required_tools_count}, "
                f"distraction={distraction_count}, total={required_tools_count + distraction_count}")
    
    return distraction_count


def load_tau2_tasks(domain: str, task_type: str = "lite"):
    """Load tau2 tasks"""
    from tau2.data_model.tasks import Task
    
    tau2_data_path = Path(__file__).parent.parent / "benchmarks" / "tau2-bench" / "data" / "tau2" / "domains" / domain
    
    if task_type == "lite":
        task_file = tau2_data_path / "tasks_lite.json"
    else:
        task_file = tau2_data_path / "tasks.json"
    
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    
    with open(task_file) as f:
        tasks_data = json.load(f)
    
    return [Task.model_validate(t) for t in tasks_data]


def load_swebench_tasks(task_ids: Optional[list[str]] = None, max_tasks: Optional[int] = None) -> list[dict]:
    """
    Load SWE-Bench tasks (from Terminal-Bench adapter generated dataset)
    
    Args:
        task_ids: Specific task ID list (e.g. ["astropy__astropy-12907"])
        max_tasks: Maximum number of tasks
    
    Returns:
        Task list, each containing instance_id, instruction, task_path, etc.
    """
    import yaml
    
    swebench_dataset_path = Path(__file__).parent.parent / "benchmarks" / "terminal-bench" / "dataset" / "swebench-verified"
    
    if not swebench_dataset_path.exists():
        raise FileNotFoundError(
            f"SWE-Bench dataset not found at {swebench_dataset_path}. "
            "Please run Terminal-Bench adapter first: cd terminal-bench/adapters/swebench && uv run run_adapter.py"
        )
    
    tasks = []
    task_dirs = sorted(swebench_dataset_path.iterdir())
    
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        
        task_id = task_dir.name
        
        # Filter specific tasks
        if task_ids and task_id not in task_ids:
            continue
        
        task_yaml = task_dir / "task.yaml"
        if not task_yaml.exists():
            continue
        
        with open(task_yaml) as f:
            task_data = yaml.safe_load(f)
        
        # Read test configuration
        tests_config = task_dir / "tests" / "config.json"
        swebench_instance = {}
        if tests_config.exists():
            with open(tests_config) as f:
                swebench_instance = json.load(f)
        
        tasks.append({
            "instance_id": task_id,
            "instruction": task_data.get("instruction", ""),
            "task_path": str(task_dir),
            "swebench_instance": swebench_instance,
            "difficulty": task_data.get("difficulty", "unknown"),
            "category": task_data.get("category", "debugging"),
        })
        
        if max_tasks and len(tasks) >= max_tasks:
            break
    
    logger.info(f"Loaded {len(tasks)} SWE-Bench tasks from {swebench_dataset_path}")
    return tasks


def load_terminalbench_tasks(task_ids: Optional[list[str]] = None, max_tasks: Optional[int] = None) -> list[dict]:
    """
    Load TerminalBench tasks (from terminal-bench-core task directory)
    
    Args:
        task_ids: Specific task ID list (e.g. ["chess-best-move", "build-linux-kernel-qemu"])
        max_tasks: Maximum number of tasks
    
    Returns:
        Task list, each containing id, instruction, runtime, etc.
    """
    import yaml
    
    terminalbench_tasks_path = Path(__file__).parent.parent / "benchmarks" / "terminal-bench" / "terminal-bench-core-0.1.1-tasks"
    
    if not terminalbench_tasks_path.exists():
        raise FileNotFoundError(
            f"TerminalBench tasks not found at {terminalbench_tasks_path}. "
            "Please ensure the terminal-bench repository is properly set up."
        )
    
    tasks = []
    task_dirs = sorted(terminalbench_tasks_path.iterdir())
    
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        
        task_id = task_dir.name
        
        # Filter specific tasks
        if task_ids and task_id not in task_ids:
            continue
        
        task_yaml = task_dir / "task.yaml"
        if not task_yaml.exists():
            continue
        
        with open(task_yaml) as f:
            task_data = yaml.safe_load(f)
        
        # Build format compatible with the terminalbench branch in run_from_task_file
        tasks.append({
            "id": task_id,
            "instruction": task_data.get("instruction", ""),
            "runtime": {
                "task_path": str(task_dir),
            },
            "difficulty": task_data.get("difficulty", "unknown"),
            "category": task_data.get("category", "unknown"),
            "max_agent_timeout_sec": task_data.get("max_agent_timeout_sec", 360.0),
            "max_test_timeout_sec": task_data.get("max_test_timeout_sec", 60.0),
        })
        
        if max_tasks and len(tasks) >= max_tasks:
            break
    
    logger.info(f"Loaded {len(tasks)} TerminalBench tasks from {terminalbench_tasks_path}")
    return tasks


def load_tau2_policy(domain: str) -> str:
    """Load tau2 policy"""
    tau2_data_path = Path(__file__).parent.parent / "benchmarks" / "tau2-bench" / "data" / "tau2" / "domains" / domain
    
    if domain == "telecom":
        main_policy_file = tau2_data_path / "main_policy.md"
        tech_support_file = tau2_data_path / "tech_support_manual.md"
        
        main_policy = ""
        tech_support_policy = ""
        
        if main_policy_file.exists():
            main_policy = main_policy_file.read_text()
        if tech_support_file.exists():
            tech_support_policy = tech_support_file.read_text()
        
        return (
            "<main_policy>\n" + main_policy + "\n</main_policy>\n"
            + "<tech_support_policy>\n" + tech_support_policy + "\n</tech_support_policy>"
        )
    else:
        policy_file = tau2_data_path / "policy.md"
        if not policy_file.exists():
            policy_file = tau2_data_path / "main_policy.md"
        
        if policy_file.exists():
            return policy_file.read_text()
        return ""


def get_tau2_user_tools(domain: str):
    """
    Get user tools for a tau2 domain
    
    User tools are tools used by the User Simulator, e.g. for telecom:
    - get_status_bar, run_speed_test, reboot_phone, etc.
    
    Args:
        domain: tau2 domain name
        
    Returns:
        list[Tool]: User tool list, returns None if the domain has no user tools
    """
    try:
        env_constructor = tau2_registry.get_env_constructor(domain)
        environment = env_constructor()
        if environment.user_tools is not None:
            return environment.get_user_tools()
        return None
    except Exception as e:
        logger.warning(f"Failed to get user tools for domain {domain}: {e}")
        return None


def get_tau2_user_toolkit(domain: str):
    """
    Get the user toolkit for a tau2 domain (for executing user tools)
    
    Args:
        domain: tau2 domain name
        
    Returns:
        ToolKit: User toolkit, returns None if the domain has none
    """
    try:
        env_constructor = tau2_registry.get_env_constructor(domain)
        environment = env_constructor()
        return environment.user_tools
    except Exception as e:
        logger.warning(f"Failed to get user toolkit for domain {domain}: {e}")
        return None


class Tau2UserSimulatorAdapter:
    """
    Adapter: Adapts tau2's UserSimulator to the interface required by UniversalAgent
    
    tau2 UserSimulator uses generate_next_message(message, state) -> (response, state)
    UniversalAgent needs respond(agent_message) -> user_response
    
    Important: The adapter now also handles user tool calls, ensuring they are correctly recorded in the trace
    """
    
    def __init__(
        self,
        user_scenario,
        user_llm: str,
        user_llm_args: dict,
        user_tools: Optional[list] = None,
        user_toolkit=None,  # tau2 user toolkit for executing user tools (fallback only)
        mcp_execute_user_tool: Optional[callable] = None,  # MCP tool call callback
        domain: str = "",  # domain name, used to construct MCP tool names
    ):
        """
        Initialize the adapter
        
        Args:
            user_scenario: User scenario (tau2 Task.user_scenario)
            user_llm: LLM model name used by the User Simulator
            user_llm_args: LLM parameters
            user_tools: User tools (optional)
            user_toolkit: tau2 user toolkit (fallback, used when MCP calls are unavailable)
            mcp_execute_user_tool: MCP tool call callback function, signature: (tool_name, arguments) -> result
            domain: domain name
        """
        self.simulator = Tau2UserSimulator(
            tools=user_tools,
            instructions=user_scenario,
            llm=user_llm,
            llm_args=user_llm_args,
        )
        self.state = self.simulator.get_init_state()
        self.user_toolkit = user_toolkit
        self.mcp_execute_user_tool = mcp_execute_user_tool
        self.domain = domain
        self._last_user_message = None  # Save the last UserMessage to get tool_calls
        self._last_tool_results = []  # Save User tool execution results (for generating ToolMessage)
    
    async def _execute_user_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute user tool, only via MCP calls (ensures using correctly initialized Environment)"""
        import json
        
        if not self.mcp_execute_user_tool:
            logger.error(f"[USER_TOOL] No MCP callback available for {tool_name}")
            return f"Error: No MCP callback available to execute {tool_name}"
        
        try:
            mcp_tool_name = f"__{self.domain}_execute_user_tool"
            logger.debug(f"[USER_TOOL] Calling MCP: {mcp_tool_name}({tool_name}, {arguments})")
            result = await self.mcp_execute_user_tool(
                mcp_tool_name,
                {"tool_name": tool_name, "arguments_json": json.dumps(arguments)}
            )
            # Extract text content from CallToolResult object
            if hasattr(result, 'content') and result.content:
                text_content = result.content[0].text if result.content else ""
                logger.debug(f"[USER_TOOL] MCP result: {text_content[:100]}...")
                return text_content
            else:
                return str(result) if result else ""
        except Exception as e:
            logger.error(f"[USER_TOOL] MCP call failed: {e}")
            return f"Error: {e}"
    
    def respond(self, agent_message: str) -> Optional[str]:
        """
        Respond to an Agent message
        
        Args:
            agent_message: Message sent by the Agent
            
        Returns:
            User reply, or None if it's a stop signal
        """
        # Convert agent message to tau2 format
        assistant_msg = AssistantMessage(
            role="assistant",
            content=agent_message,
        )
        
        # Call tau2 UserSimulator
        user_message, self.state = self.simulator.generate_next_message(
            message=assistant_msg,
            state=self.state,
        )
        
        # Save the full user_message for later retrieval of tool_calls
        self._last_user_message = user_message
        
        user_response = user_message.content
        
        # Sync method does not support user tool calls (MCP calls are async)
        # If there are tool calls, respond_async should be used instead
        if user_message.tool_calls:
            logger.warning(f"[USER_TOOL] respond() called with tool_calls, but sync method cannot execute MCP calls. Use respond_async() instead.")
            tool_results = [f"[{tc.name}]: Error: sync method cannot execute user tools" for tc in user_message.tool_calls]
            tool_info = "\n".join(tool_results)
            user_response = f"{user_response}\n\n[User actions failed - use async]\n{tool_info}"
        
        # Check if this is a stop signal
        if self.simulator.is_stop(user_message):
            # Return response with [STOP] marker, so the Agent knows to stop
            return user_response
        
        return user_response
    
    async def respond_async(self, agent_message: str) -> Optional[str]:
        """
        Async respond to an Agent message (supports executing user tools via MCP)
        
        Args:
            agent_message: Message sent by the Agent
            
        Returns:
            User reply, or None if it's a stop signal
        """
        # Convert agent message to tau2 format
        assistant_msg = AssistantMessage(
            role="assistant",
            content=agent_message,
        )
        
        # Call tau2 UserSimulator
        user_message, self.state = self.simulator.generate_next_message(
            message=assistant_msg,
            state=self.state,
        )
        
        # Save the full user_message for later retrieval of tool_calls
        self._last_user_message = user_message
        self._last_tool_results = []  # Clear previous results
        
        user_response = user_message.content
        
        # If there are user tool calls, execute them via MCP
        if user_message.tool_calls:
            tool_results = []
            for tool_call in user_message.tool_calls:
                result = await self._execute_user_tool(tool_call.name, tool_call.arguments)
                # Save detailed results for generating ToolMessage
                tool_id = tool_call.id if tool_call.id else f"user_{tool_call.name}"
                self._last_tool_results.append({
                    "id": tool_id,
                    "name": tool_call.name,
                    "result": result
                })
                tool_results.append(f"[{tool_call.name}]: {result}")
            
            # Append tool execution results to the user response
            if tool_results:
                tool_info = "\n".join(tool_results)
                user_response = f"{user_response}\n\n[User actions performed]\n{tool_info}"
        
        # Check if this is a stop signal
        if self.simulator.is_stop(user_message):
            return user_response
        
        return user_response
    
    def get_last_tool_calls(self):
        """Get tool calls from the last user message (for recording in the trace)
        
        Converts tau2's ToolCall to source.agent.ToolCall format,
        ensuring the tau2 evaluator can correctly identify User-executed tool calls during set_state() replay.
        """
        if self._last_user_message and self._last_user_message.tool_calls:
            # Convert tau2 ToolCall to AgentToolCall format
            converted = []
            for tc in self._last_user_message.tool_calls:
                converted.append(AgentToolCall(
                    name=tc.name,
                    arguments=tc.arguments,
                    id=tc.id if tc.id else f"user_{tc.name}"
                ))
            return converted
        return None
    
    def get_last_tool_results(self):
        """Get the results of the last User tool execution (for generating ToolMessage)
        
        Return format: [{"id": "...", "name": "...", "result": "..."}, ...]
        """
        return self._last_tool_results if self._last_tool_results else None
    
    def is_stop_signal(self, response: str) -> bool:
        """Check if the response contains a stop signal"""
        if response is None:
            return True
        return STOP in response or TRANSFER in response or OUT_OF_SCOPE in response
    
    async def replay_message_history(self, messages: list[dict]) -> None:
        """
        Restore User Simulator state from a checkpoint
        
        Core method for Sequential Scaling: restores User Simulator state by replaying message history.
        
        How it works:
        1. Calls the tau2 server's __{domain}_replay_message_history tool
        2. The tau2 server internally calls environment.set_state() to replay all tool calls
        3. Environment state is restored to the state at checkpoint save time
        4. Rebuilds the User Simulator's internal state
        
        Args:
            messages: Complete message history saved in the checkpoint (list[dict])
        """
        if self.mcp_execute_user_tool is None:
            logger.warning("[REPLAY] No MCP callback available, cannot restore User Simulator state")
            return
        
        try:
            # 1. Call tau2 server to replay message history and restore environment state
            import json
            mcp_tool_name = f"__{self.domain}_replay_message_history"
            logger.info(f"[REPLAY] Calling {mcp_tool_name} with {len(messages)} messages")
            
            result = await self.mcp_execute_user_tool(
                mcp_tool_name,
                {"messages_json": json.dumps(messages)}
            )
            
            # Parse result
            if hasattr(result, 'content') and result.content:
                result_text = result.content[0].text if result.content else ""
            else:
                result_text = str(result) if result else ""
            
            result_data = json.loads(result_text) if result_text else {}
            if result_data.get("status") == "success":
                logger.info(f"[REPLAY] Environment state restored: {result_data.get('replayed_messages', 0)} messages replayed")
            else:
                logger.warning(f"[REPLAY] Failed to restore environment: {result_data.get('message', 'unknown error')}")
            
            # 2. Rebuild User Simulator's internal state
            # User Simulator's state is derived from message history
            # Need to simulate the entire conversation history to restore simulator.state
            from tau2.data_model.message import AssistantMessage
            
            # Reset simulator state
            self.state = self.simulator.get_init_state()
            
            # Extract assistant message sequence from message history
            # User Simulator's generate_next_message needs assistant messages as input
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content") or ""
                    assistant_msg = AssistantMessage(
                        role="assistant",
                        content=content,
                    )
                    # Call generate_next_message to update state (but don't use the returned user_message)
                    try:
                        _, self.state = self.simulator.generate_next_message(
                            message=assistant_msg,
                            state=self.state,
                        )
                    except Exception as e:
                        logger.debug(f"[REPLAY] State update for message skipped: {e}")
            
            logger.info(f"[REPLAY] User Simulator state restored successfully")
            
        except Exception as e:
            import traceback
            logger.error(f"[REPLAY] Failed to restore User Simulator state: {e}")
            logger.debug(f"[REPLAY] Traceback: {traceback.format_exc()}")


def load_tasks_from_file(task_file: Path) -> list[dict]:
    """
    Load mixed tasks from a JSON file
    
    File format:
    [
        {
            "benchmark": "tau2",
            "domain": "airline",
            "task": { ... original tau2 task object ... }
        },
        ...
    ]
    
    Args:
        task_file: JSON file path
        
    Returns:
        Task list
    """
    with open(task_file) as f:
        tasks = json.load(f)
    logger.info(f"Loaded {len(tasks)} tasks from {task_file}")
    return tasks


async def run_from_task_file(
    agent: UniversalAgent,
    task_file: Path | None,
    user_llm: str,
    user_llm_args: dict,
    output_dir: Path,
    host,  # BenchmarkHost instance, used to call internal tools
    distraction_count: int | None = None,
    tool_seed: int | None = None,
    simulation_seed: int = 0,
    task_entries: list | None = None,
    resume: bool = True,
    # Sequential Scaling parameters
    checkpoint_store: Optional[CheckpointStore] = None,
    token_budget: Optional[int] = None,
    sequential_reuse: bool = False,
    # GPT-5 compatible mode parameters
    gpt5_mode: bool = False,
    max_tools: int = 125,
) -> dict:
    """
    Load tasks from a JSON file and run them
    
    Supports mixed task files (tau2 + mcpbench + swebench)
    
    Args:
        agent: UniversalAgent instance
        task_file: Task file path (ignored if task_entries is not empty)
        user_llm: LLM model name used by UserSimulator
        user_llm_args: UserSimulator LLM parameters
        output_dir: Results output directory
        distraction_count: Number of distraction servers (None = use all tools)
        tool_seed: Random seed for tool selection
        simulation_seed: Simulation random seed (aligned with tau2-bench)
        task_entries: Directly provided task list (takes priority over task_file)
        checkpoint_store: CheckpointStore for saving scaling checkpoints
        token_budget: Token budget for scaling mode
        sequential_reuse: If True, try to resume from a previous budget's checkpoint
        
    Returns:
        Run result statistics
    """
    from tau2.data_model.tasks import Task
    
    # Create subdirectories
    trace_dir = output_dir / "traces"
    eval_dir = output_dir / "evaluations"
    trace_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper function: Save Scaling Checkpoint
    def save_scaling_checkpoint(
        benchmark: str, 
        task_id: str, 
        trace, 
        agent: UniversalAgent
    ):
        """Save scaling checkpoint if in scaling mode"""
        if checkpoint_store is None or token_budget is None:
            return
        
        checkpoint = ScalingCheckpoint(
            task_id=task_id,
            benchmark=benchmark,
            budget_level=token_budget,
            messages=[m.to_dict() for m in trace.messages],
            cumulative_tokens=trace.total_tokens,
            extend_rounds=agent.extend_rounds.copy(),
            stop_rounds=agent.stop_rounds.copy(),
            total_prompt_tokens=trace.total_prompt_tokens,
            total_output_tokens=trace.total_output_tokens,
            rounds=[{
                "round_number": r.round_number,
                "incremental_tokens": r.incremental_tokens,
                "cumulative_total_tokens": r.cumulative_total_tokens,
            } for r in trace.rounds],
            final_response=trace.final_response,
            seed=agent.scaling_config.seed if agent.scaling_config else 42,
        )
        checkpoint_path = checkpoint_store.save(checkpoint)
        logger.info(f"[SCALING] Saved checkpoint: {checkpoint_path}")
        logger.info(f"[SCALING] Budget={token_budget}, Tokens={trace.total_tokens}, "
                   f"EXTEND={len(agent.extend_rounds)}, STOP={len(agent.stop_rounds)}")
    
    # Load tasks (prioritize directly provided task_entries)
    if task_entries is None:
        if task_file is None:
            raise ValueError("Either task_file or task_entries must be provided")
        task_entries = load_tasks_from_file(task_file)
    
    # Evaluators grouped by benchmark
    evaluators = {
        "tau2": Tau2Evaluator(solo_mode=False),
    }
    
    # mcpbench evaluator (lazy loaded)
    mcpbench_evaluator = None
    available_tools = None
    
    # Cache policy (avoid repeated loading)
    policy_cache = {}
    
    all_results = []
    
    # Save original parameters for dynamic switching
    original_max_steps = agent.max_steps
    original_max_tokens = agent.llm.max_tokens if hasattr(agent.llm, 'max_tokens') else None
    
    for i, entry in enumerate(task_entries):
        benchmark = entry["benchmark"]
        domain = entry["domain"]
        # Different benchmarks have different formats:
        # - search: data is directly in the entry
        # - tau2/mcpbench/swebench/terminalbench: task data is in the "task" field
        if benchmark == "search":
            task_data = entry  # The entire entry is the task data
        else:
            task_data = entry.get("task", entry)  # Prefer "task" field, otherwise use the entire entry
        
        # === swebench and terminalbench are mutually exclusive ===
        # When running swebench, exclude terminalbench tools, and vice versa
        # This setting applies to all modes (not just distraction mode)
        if benchmark == "swebench":
            agent.set_excluded_clients(["terminalbench"])
        elif benchmark == "terminalbench":
            agent.set_excluded_clients(["swebench"])
        else:
            agent.set_excluded_clients(None)  # Other benchmarks don't exclude any tools
        
        # === Unified parameter settings: all benchmarks use the same max_steps=500, max_tokens=32768, temperature=0.7 ===
        agent.max_steps = 500
        if hasattr(agent.llm, 'max_tokens'):
            agent.llm.max_tokens = 32768
        if hasattr(agent.llm, 'temperature'):
            agent.llm.temperature = 0.7
        logger.debug(f"  [{benchmark}] max_steps=500, max_tokens=32768, temperature=0.7")
        
        # Determine effective_distraction_count (dynamically calculated in GPT-5 mode)
        effective_distraction_count = distraction_count
        
        # GPT-5 mode: dynamically calculate distraction count for mcpbench tasks
        if gpt5_mode and benchmark == "mcpbench":
            task_domain = entry.get("domain", "")
            dynamic_distraction = calculate_dynamic_distraction_count(task_domain, max_tools)
            
            # If user specified distraction_count, take the smaller value; otherwise use the dynamically calculated value
            if distraction_count is not None:
                effective_distraction_count = min(distraction_count, dynamic_distraction)
            else:
                effective_distraction_count = dynamic_distraction
            
            logger.info(f"  [GPT-5 mode] domain={task_domain}, dynamic_distraction={dynamic_distraction}, "
                        f"effective={effective_distraction_count}")
        
        # Set tool filtering (if enabled, or for mcpbench tasks in GPT-5 mode)
        if effective_distraction_count is not None:
            # swebench and terminalbench are mutually exclusive: exclude one's tools when running the other
            excluded_clients = None
            
            if benchmark == "tau2":
                # tau2: determine required servers based on domain
                required_clients = [f"tau2-{domain}"]
            elif benchmark in ("browsecomp", "mind2web", "webvoyager"):
                # search domain: use the corresponding benchmark server
                required_clients = [benchmark]
            elif benchmark == "search":
                # Compatible with old search benchmark: determine server based on type field
                task_type = task_data.get("type", "browsecomp")
                required_clients = [task_type]
            elif benchmark == "terminalbench":
                # terminalbench: use terminalbench server, exclude swebench
                required_clients = ["terminalbench"]
                excluded_clients = ["swebench"]
            elif benchmark == "swebench":
                # swebench: use swebench server, exclude terminalbench
                required_clients = ["swebench"]
                excluded_clients = ["terminalbench"]
            else:
                # mcpbench: read servers field from task
                # Server names directly use names from commands.json (e.g. "BioMCP"), no prefix needed
                servers = task_data.get("servers", [entry.get("domain", "")])
                required_clients = list(servers)
            
            # Use task index as part of the seed, ensuring reproducibility while varying per task
            effective_seed = (tool_seed or 0) + i
            num_active = agent.set_active_tools(
                required_client_names=required_clients,
                distraction_count=effective_distraction_count,
                seed=effective_seed,
                excluded_client_names=excluded_clients,
            )
            logger.info(f"  Active tools: {num_active} (required: {required_clients}, distraction: {effective_distraction_count}, excluded: {excluded_clients})")
        
        if benchmark == "tau2":
            # ===== tau2 task =====
            # Create tau2 Task object
            task = Task.model_validate(task_data)
            task_id = task.id
            
            # Generate unique result filename
            # Use hash to ensure long task_ids don't collide
            import hashlib
            task_id_str = str(task_id)
            safe_task_id = task_id_str.replace("/", "_").replace("|", "_").replace("[", "").replace("]", "")[:50]
            # Add hash suffix to ensure uniqueness
            task_hash = hashlib.md5(task_id_str.encode()).hexdigest()[:8]
            result_name = f"{benchmark}_{domain}_{safe_task_id}_{task_hash}"
            
            # ===== Resume check =====
            trace_file = trace_dir / f"{result_name}.json"
            if resume and trace_file.exists():
                logger.info(f"[{i+1}/{len(task_entries)}] SKIP (already completed): {benchmark}/{domain}/{task_id}")
                # Load existing results for statistics
                try:
                    with open(trace_file) as f:
                        existing_data = json.load(f)
                    eval_file = eval_dir / f"{result_name}.json"
                    if eval_file.exists():
                        with open(eval_file) as f:
                            existing_eval = json.load(f)
                        reward = existing_eval.get("simulation_result", {}).get("reward", 0.0)
                        all_results.append({
                            "benchmark": benchmark,
                            "domain": domain,
                            "task_id": task_id,
                            "reward": reward,
                            "success": reward >= 1.0,
                            "skipped": True,
                        })
                except Exception as e:
                    logger.warning(f"Failed to load existing result: {e}")
                continue
            
            logger.info(f"[{i+1}/{len(task_entries)}] Running: {benchmark}/{domain}/{task_id}")
            logger.info(f"============================================================")
            logger.info(f"[SIMULATION] Starting tau2 task: {task_id}")
            logger.info(f"[SIMULATION] Domain: {domain}")
            
            # ===== Reset environment and apply initial state =====
            # This is crucial for correct evaluation! Each task needs to start from the correct initial state
            try:
                # 1. Reset environment to default state
                reset_tool = f"__{domain}_reset_environment"
                logger.info(f"[SIMULATION] Calling {reset_tool} to reset environment...")
                reset_result = await agent.host.call_tool(reset_tool, {})
                logger.info(f"[SIMULATION] Reset environment result: {reset_result}")
                
                # 2. Apply task's initial state (if any)
                if task.initial_state and task.initial_state.initialization_actions:
                    # Convert initialization_actions to JSON
                    # Note: EnvFunctionCall uses the func_name attribute, not name
                    init_actions = []
                    for action in task.initial_state.initialization_actions:
                        init_actions.append({
                            "name": action.func_name,  # EnvFunctionCall uses func_name, not name
                            "arguments": action.arguments,
                            "env_type": action.env_type.value if hasattr(action.env_type, 'value') else str(action.env_type),
                        })
                    apply_tool = f"__{domain}_apply_initial_state"
                    await agent.host.call_tool(apply_tool, {
                        "initialization_actions_json": json.dumps(init_actions)
                    })
                    logger.info(f"[SIMULATION] Applied {len(init_actions)} initialization actions for task {task_id}")
                    for idx, action in enumerate(init_actions):
                        logger.info(f"[SIMULATION]   init_action[{idx}]: {action['env_type']}.{action['name']}({action['arguments']})")
            except Exception as e:
                logger.warning(f"Failed to reset/initialize environment for task {task_id}: {e}")
            
            # Load policy (using cache)
            if domain not in policy_cache:
                policy_cache[domain] = load_tau2_policy(domain)
            policy = policy_cache[domain]
            
            # Get user tools (using cache)
            user_tools_cache_key = f"user_tools_{domain}"
            if user_tools_cache_key not in policy_cache:
                policy_cache[user_tools_cache_key] = get_tau2_user_tools(domain)
            user_tools = policy_cache[user_tools_cache_key]
            
            # Get user toolkit (using cache, as fallback)
            user_toolkit_cache_key = f"user_toolkit_{domain}"
            if user_toolkit_cache_key not in policy_cache:
                policy_cache[user_toolkit_cache_key] = get_tau2_user_toolkit(domain)
            user_toolkit = policy_cache[user_toolkit_cache_key]
            
            # Create MCP user tool execution callback
            async def mcp_execute_user_tool(tool_name: str, arguments: dict):
                """Call user tool via MCP"""
                return await agent.host.call_tool(tool_name, arguments)
            
            # Create UserSimulator adapter
            user_simulator = Tau2UserSimulatorAdapter(
                user_scenario=task.user_scenario,
                user_llm=user_llm,
                user_llm_args=user_llm_args,
                user_tools=user_tools,
                user_toolkit=user_toolkit,
                mcp_execute_user_tool=mcp_execute_user_tool,
                domain=domain,
            )
            
            # Run agent (supports Sequential Scaling checkpoint reuse)
            logger.info(f"[SIMULATION] Starting agent-user conversation...")
            
            # Sequential Scaling: Try to resume from previous budget's checkpoint
            prefix_checkpoint = None
            if sequential_reuse and checkpoint_store and token_budget:
                # Find the best prefix checkpoint (smaller budget)
                prefix_checkpoint = checkpoint_store.find_best_prefix_checkpoint(
                    benchmark=benchmark,
                    task_id=task_id,
                    target_budget=token_budget,
                )
                if prefix_checkpoint:
                    logger.info(f"[SEQUENTIAL] Found prefix checkpoint at budget {prefix_checkpoint.budget_level}, "
                              f"will resume from it for target budget {token_budget}")
            
            if prefix_checkpoint:
                # Clean stop prompts and resume from checkpoint
                cleaned_checkpoint = checkpoint_store.clean_stop_prompts(prefix_checkpoint)
                trace = await agent.run_from_checkpoint_with_user_simulator(
                    checkpoint=cleaned_checkpoint,
                    target_budget=token_budget,
                    user_simulator=user_simulator,
                    policy=policy,
                )
                logger.info(f"[SEQUENTIAL] Resumed from checkpoint: {len(cleaned_checkpoint.messages)} messages recovered")
            else:
                trace = await agent.run_with_user_simulator(
                    task_id=task_id,
                    user_simulator=user_simulator,
                    policy=policy,
                )
            logger.info(f"[SIMULATION] Conversation completed: {len(trace.messages)} messages, {trace.total_steps} steps")
            if trace.error:
                logger.warning(f"[SIMULATION] Error: {trace.error}")
            
            # Save trace
            trace_file = trace_dir / f"{result_name}.json"
            trace_data = {
                "benchmark": benchmark,
                "domain": domain,
                "task_id": task_id,
                "trace": trace.to_dict(),
                "total_steps": trace.total_steps,
                "error": trace.error,
            }
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2, default=str)
            
            # Save Scaling Checkpoint (if in Scaling mode)
            save_scaling_checkpoint(benchmark, task_id, trace, agent)
            
            # Evaluate
            logger.info(f"[EVALUATION] Starting tau2 evaluation for task: {task_id}")
            evaluator = evaluators.get(benchmark)
            if evaluator:
                eval_result = evaluator.evaluate(trace, task, domain)
                
                # Save evaluation (tau2-bench native format: simulation format)
                eval_file = eval_dir / f"{result_name}.json"
                
                # Get model name
                llm_model = agent.llm.model if hasattr(agent.llm, 'model') else "unknown"
                
                # Determine the actual termination_reason (based on trace.error, not evaluation result)
                if trace.error:
                    if "max steps" in trace.error.lower() or "max_steps" in trace.error.lower():
                        actual_termination_reason = "max_steps"
                    else:
                        actual_termination_reason = "too_many_errors"
                else:
                    actual_termination_reason = "user_stop"
                
                # Build tau2-bench native simulation format evaluation data
                eval_data = {
                    "id": str(uuid.uuid4()),
                    "task_id": task_id,
                    "timestamp": datetime.now().isoformat(),
                    "start_time": trace.start_time if isinstance(trace.start_time, str) else (trace.start_time.isoformat() if trace.start_time else None),
                    "end_time": trace.end_time if isinstance(trace.end_time, str) else (trace.end_time.isoformat() if trace.end_time else None),
                    "duration": trace.duration,
                    "termination_reason": actual_termination_reason,
                    "agent_cost": 0.0,  # Needs to be calculated from trace
                    "user_cost": 0.0,
                    "reward_info": {
                        "reward": eval_result.reward,
                        "db_check": eval_result.info.get("db_check", {}) if eval_result.info else {},
                        "env_assertions": eval_result.info.get("env_assertions", []) if eval_result.info else [],
                        "action_checks": eval_result.info.get("action_checks", []) if eval_result.info else [],
                        "nl_assertions": eval_result.info.get("nl_assertions", []) if eval_result.info else [],
                        "communicate_checks": eval_result.info.get("communicate_checks") if eval_result.info else None,
                        "reward_basis": eval_result.info.get("reward_basis", []) if eval_result.info else [],
                        "reward_breakdown": eval_result.info.get("reward_breakdown", {}) if eval_result.info else {},
                        "info": eval_result.info.get("info", {}) if eval_result.info else {},
                    },
                    "messages": [msg.to_dict() if hasattr(msg, 'to_dict') else msg for msg in trace.messages],
                    "trial": 0,
                    "seed": simulation_seed,
                    "model_name": llm_model,
                    "domain": domain,
                }
                
                # If evaluation failed, add error info
                if eval_result.error:
                    eval_data["error"] = eval_result.error
                
                with open(eval_file, "w") as f:
                    json.dump(eval_data, f, indent=2, default=str, ensure_ascii=False)
                
                logger.info(f"[EVALUATION] ============================================================")
                logger.info(f"[EVALUATION] Task: {task_id}")
                logger.info(f"[EVALUATION] Reward: {eval_result.reward:.2f}")
                logger.info(f"[EVALUATION] Success: {eval_result.success}")
                if eval_result.info:
                    logger.info(f"[EVALUATION] Reward breakdown: {eval_result.info.get('reward_breakdown', {})}")
                    if eval_result.info.get('db_check'):
                        logger.info(f"[EVALUATION] DB check: {eval_result.info.get('db_check')}")
                    if eval_result.info.get('env_assertions'):
                        for idx, ea in enumerate(eval_result.info.get('env_assertions', [])):
                            logger.info(f"[EVALUATION]   env_assertion[{idx}]: {ea}")
                    if eval_result.info.get('action_checks'):
                        for idx, ac in enumerate(eval_result.info.get('action_checks', [])):
                            logger.info(f"[EVALUATION]   action_check[{idx}]: {ac}")
                if eval_result.error:
                    logger.warning(f"[EVALUATION] Error: {eval_result.error}")
                logger.info(f"[EVALUATION] ============================================================")
                
                all_results.append({
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_id": task_id,
                    "reward": eval_result.reward,
                    "success": eval_result.success,
                    "total_steps": trace.total_steps,
                    "error": trace.error or eval_result.error,
                })
            else:
                logger.warning(f"  -> No evaluator for benchmark: {benchmark}")
                all_results.append({
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_id": task_id,
                    "reward": None,
                    "success": None,
                    "total_steps": trace.total_steps,
                    "error": trace.error,
                })
                
        elif benchmark == "mcpbench":
            # ===== mcpbench task =====
            task_id = task_data.get("task_id", f"mcpbench_{i}")
            servers = task_data.get("servers", [])
            fuzzy_description = task_data.get("fuzzy_description", task_data.get("task_description", ""))
            task_description = task_data.get("task_description", "")
            dependency_analysis = task_data.get("dependency_analysis", "")
            
            # Generate unique result filename
            safe_task_id = str(task_id).replace("/", "_").replace("|", "_")[:50]
            result_name = f"{benchmark}_{safe_task_id}"
            
            # ===== Resume check =====
            trace_file = trace_dir / f"{result_name}.json"
            if resume and trace_file.exists():
                logger.info(f"[{i+1}/{len(task_entries)}] SKIP (already completed): {benchmark}/{task_id}")
                # Load existing results for statistics
                try:
                    eval_file = eval_dir / f"{result_name}.json"
                    if eval_file.exists():
                        with open(eval_file) as f:
                            existing_eval = json.load(f)
                        # Calculate reward using 2-axis formula:
                        # s_rule = avg(valid_tool_name_rate, input_schema_compliance, execution_success_rate)
                        # s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
                        # reward = (s_rule + s_llm) / 2
                        ev = existing_eval.get("evaluation", {})
                        valid_tool_name_rate = ev.get("valid_tool_name_rate", 1.0) or 1.0
                        input_schema_compliance = ev.get("input_schema_compliance", 1.0) or 1.0
                        execution_success_rate = ev.get("execution_success_rate", 1.0)
                        if execution_success_rate is None:
                            execution_success_rate = 1.0
                        
                        task_fulfillment = ev.get("task_fulfillment", 0) or 0
                        grounding = ev.get("grounding", 0) or 0
                        tool_appropriateness = ev.get("tool_appropriateness", 0) or 0
                        parameter_accuracy = ev.get("parameter_accuracy", 0) or 0
                        dependency_awareness = ev.get("dependency_awareness", 0) or 0
                        
                        s_rule = (valid_tool_name_rate + input_schema_compliance + execution_success_rate) / 3
                        s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
                        reward = (s_rule + s_llm) / 2
                        all_results.append({
                            "benchmark": benchmark,
                            "domain": domain,
                            "task_id": task_id,
                            "servers": servers,
                            "reward": reward,
                            "success": reward >= 0.8,
                            "skipped": True,
                        })
                except Exception as e:
                    logger.warning(f"Failed to load existing result: {e}")
                continue
            
            logger.info(f"[{i+1}/{len(task_entries)}] Running: {benchmark}/{task_id}")
            logger.info(f"  Servers: {servers}")
            
            # Run agent (mcpbench is single-turn task, supports Sequential Scaling checkpoint reuse)
            # Sequential Scaling: Try to resume from previous budget's checkpoint
            prefix_checkpoint = None
            if sequential_reuse and checkpoint_store and token_budget:
                prefix_checkpoint = checkpoint_store.find_best_prefix_checkpoint(
                    benchmark=benchmark,
                    task_id=task_id,
                    target_budget=token_budget,
                )
                if prefix_checkpoint:
                    logger.info(f"[SEQUENTIAL] Found prefix checkpoint at budget {prefix_checkpoint.budget_level}, "
                              f"will resume from it for target budget {token_budget}")
            
            if prefix_checkpoint:
                # Clean stop prompts and resume from checkpoint
                cleaned_checkpoint = checkpoint_store.clean_stop_prompts(prefix_checkpoint)
                trace = await agent.run_from_checkpoint(
                    checkpoint=cleaned_checkpoint,
                    target_budget=token_budget,
                )
                logger.info(f"[SEQUENTIAL] Resumed from checkpoint: {len(cleaned_checkpoint.messages)} messages recovered")
            else:
                trace = await agent.run(
                    task_id=task_id,
                    instruction=fuzzy_description,
                    policy=None,
                )
            
            # Save trace
            trace_file = trace_dir / f"{result_name}.json"
            trace_data = {
                "benchmark": benchmark,
                "domain": domain,
                "task_id": task_id,
                "servers": servers,
                "trace": trace.to_dict(),
                "total_steps": trace.total_steps,
                "error": trace.error,
            }
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Save Scaling Checkpoint (if in Scaling mode)
            save_scaling_checkpoint(benchmark, task_id, trace, agent)
            
            # Evaluate (lazy-load mcpbench evaluator)
            if mcpbench_evaluator is None:
                mcpbench_evaluator = MCPBenchEvaluator(llm_model="bedrock/openai.gpt-oss-120b-1:0")
                available_tools = mcpbench_evaluator.build_available_tools(agent.get_tools_schema())
            
            eval_result = await mcpbench_evaluator.evaluate_async(
                trace=trace,
                task_description=task_description,
                fuzzy_description=fuzzy_description,
                dependency_analysis=dependency_analysis,
                available_tools=available_tools,
            )
            
            # Save evaluation (MCP native format)
            eval_file = eval_dir / f"{result_name}.json"
            
            # Get model name
            llm_model = agent.llm.model if hasattr(agent.llm, 'model') else "unknown"
            
            # Build MCP native format evaluation data
            eval_data = {
                "task_id": task_id,
                "server_name": ", ".join(servers) if servers else domain,
                "model_name": llm_model,
                "task_description": task_description,
                "status": "completed" if not trace.error else "error",
                "total_rounds": len(trace.rounds),
                "rounds_detail": [
                    {
                        "round_number": r.round_number,
                        "reasoning": r.reasoning,
                        "should_continue": r.should_continue,
                        "prompt_tokens": r.prompt_tokens,
                        "output_tokens": r.output_tokens,
                        "round_total_tokens": r.round_total_tokens,
                        "cumulative_total_tokens": r.cumulative_total_tokens,
                        "incremental_tokens": r.incremental_tokens,
                        "tools_executed": r.tools_executed,
                        "planned_tools": r.planned_tools,  # Tools planned by LLM for execution
                        "executions": r.executions,
                    }
                    for r in trace.rounds
                ],
                "final_solution": trace.final_response or "",
                # Save accumulated_info (compressed and uncompressed versions)
                "accumulated_info": getattr(agent, 'accumulated_information', ""),
                "accumulated_info_uncompressed": getattr(agent, 'accumulated_information_uncompressed', ""),
                "evaluation": {
                    "task_fulfillment_reasoning": "",
                    "grounding_reasoning": "",
                    "tool_appropriateness_reasoning": "",
                    "parameter_accuracy_reasoning": "",
                    "dependency_awareness_reasoning": "",
                    "parallelism_efficiency_reasoning": "",
                    "task_fulfillment": eval_result.info.get("task_fulfillment", 0) if eval_result.info else 0,
                    "grounding": eval_result.info.get("grounding", 0) if eval_result.info else 0,
                    "tool_appropriateness": eval_result.info.get("tool_appropriateness", 0) if eval_result.info else 0,
                    "parameter_accuracy": eval_result.info.get("parameter_accuracy", 0) if eval_result.info else 0,
                    "dependency_awareness": eval_result.info.get("dependency_awareness", 0) if eval_result.info else 0,
                    "parallelism_and_efficiency": eval_result.info.get("parallelism_and_efficiency", 0) if eval_result.info else 0,
                    "task_completion_score": eval_result.reward * 10 if eval_result.reward else 0,
                    "tool_selection_score": eval_result.info.get("tool_appropriateness", 0) if eval_result.info else 0,
                    "planning_effectiveness_and_efficiency_score": eval_result.info.get("parallelism_and_efficiency", 0) if eval_result.info else 0,
                    "input_schema_compliance": 1.0,
                    "valid_tool_name_rate": 1.0,
                    # Calculate actual tool execution success rate (from rounds statistics)
                    "execution_success_rate": (
                        sum(1 for r in trace.rounds for e in r.executions if e.get('success', False)) / 
                        sum(1 for r in trace.rounds for _ in r.executions)
                        if sum(1 for r in trace.rounds for _ in r.executions) > 0 else None
                    ),
                    "valid_call_failure_rate": 0.0,
                    "planning_json_compliance": 1.0,
                    "server_utilization_metrics": {
                        "server_count": len(servers) if servers else 1,
                        "cross_server_coordination": len(servers) > 1 if servers else False,
                        "server_distribution": {s: 1 for s in servers} if servers else {},
                    },
                    "evaluation_timestamp": datetime.now().timestamp(),
                },
                "execution_time": trace.duration,
                "agent_execution_time": trace.duration,
                "evaluation_time": 0.0,  # Temporarily set to 0
                "total_output_tokens": trace.total_output_tokens,
                "total_prompt_tokens": trace.total_prompt_tokens,
                "total_tokens": trace.total_tokens,
            }
            
            # Extract reasoning from raw_response (if available)
            if eval_result.info and "raw_response" in eval_result.info:
                try:
                    raw_eval = json.loads(eval_result.info["raw_response"])
                    if "reasoning" in raw_eval and isinstance(raw_eval["reasoning"], dict):
                        reasoning = raw_eval["reasoning"]
                        eval_data["evaluation"]["task_fulfillment_reasoning"] = reasoning.get("task_fulfillment", "")
                        eval_data["evaluation"]["grounding_reasoning"] = reasoning.get("grounding", "")
                        eval_data["evaluation"]["tool_appropriateness_reasoning"] = reasoning.get("tool_appropriateness", "")
                        eval_data["evaluation"]["parameter_accuracy_reasoning"] = reasoning.get("parameter_accuracy", "")
                        eval_data["evaluation"]["dependency_awareness_reasoning"] = reasoning.get("dependency_awareness", "")
                        eval_data["evaluation"]["parallelism_efficiency_reasoning"] = reasoning.get("parallelism_and_efficiency", "")
                except:
                    pass
            
            with open(eval_file, "w") as f:
                json.dump(eval_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Calculate reward using 2-axis formula
            # s_rule = avg(valid_tool_name_rate, input_schema_compliance, execution_success_rate)
            # s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
            # reward = (s_rule + s_llm) / 2
            ev = eval_data["evaluation"]
            valid_tool_name_rate = ev.get("valid_tool_name_rate", 1.0) or 1.0
            input_schema_compliance = ev.get("input_schema_compliance", 1.0) or 1.0
            execution_success_rate = ev.get("execution_success_rate", 1.0)
            if execution_success_rate is None:
                execution_success_rate = 1.0
            
            task_fulfillment = ev.get("task_fulfillment", 0) or 0
            grounding = ev.get("grounding", 0) or 0
            tool_appropriateness = ev.get("tool_appropriateness", 0) or 0
            parameter_accuracy = ev.get("parameter_accuracy", 0) or 0
            dependency_awareness = ev.get("dependency_awareness", 0) or 0
            
            s_rule = (valid_tool_name_rate + input_schema_compliance + execution_success_rate) / 3
            s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
            reward_2axis = (s_rule + s_llm) / 2
            success_2axis = reward_2axis >= 0.8
            
            logger.info(f"  -> reward={reward_2axis:.4f} (s_rule={s_rule:.4f}, s_llm={s_llm:.4f}), success={success_2axis}")
            if eval_result.info:
                scores = eval_result.info
                logger.info(f"     Task: {scores.get('task_fulfillment', 0):.1f}, "
                           f"Grounding: {scores.get('grounding', 0):.1f}, "
                           f"Tools: {scores.get('tool_appropriateness', 0):.1f}")
            
            all_results.append({
                "benchmark": benchmark,
                "domain": domain,
                "task_id": task_id,
                "servers": servers,
                "reward": reward_2axis,
                "success": success_2axis,
                "total_steps": trace.total_steps,
                "error": trace.error or eval_result.error,
            })
        elif benchmark == "search" or benchmark in ("browsecomp", "mind2web", "webvoyager"):
            # ===== search domain task (browsecomp/mind2web/webvoyager) =====
            task_id = str(task_data.get("id", f"search_{i}"))
            question_raw = task_data.get("question", "")
            
            # Handle question format: may be a dict string (mind2web/webvoyager) or plain text (browsecomp)
            # If dict string format "{'task_id': ..., 'task_description': ...}", extract task_description
            question = question_raw
            if isinstance(question_raw, str) and question_raw.strip().startswith("{") and "task_description" in question_raw:
                try:
                    import ast
                    question_dict = ast.literal_eval(question_raw)
                    if isinstance(question_dict, dict) and "task_description" in question_dict:
                        question = question_dict["task_description"]
                except (ValueError, SyntaxError):
                    # Parse failed, use original string
                    pass
            
            dataset = entry.get("dataset", task_data.get("type", "browsecomp"))
            task_type = task_data.get("type", benchmark if benchmark != "search" else "browsecomp")
            
            # Use the unified search server
            server_name = "search"
            
            # Generate unique result filename
            safe_task_id = str(task_id).replace("/", "_").replace("|", "_")[:50]
            result_name = f"{task_type}_{safe_task_id}"
            
            # ===== Resume check =====
            trace_file = trace_dir / f"{result_name}.json"
            if resume and trace_file.exists():
                logger.info(f"[{i+1}/{len(task_entries)}] SKIP (already completed): {task_type}/{task_id}")
                # Load existing results for statistics
                try:
                    eval_file = eval_dir / f"{result_name}.json"
                    if eval_file.exists():
                        with open(eval_file) as f:
                            existing_eval = json.load(f)
                        reward = existing_eval.get("reward", 0.0)
                        all_results.append({
                            "benchmark": task_type,
                            "domain": task_type,
                            "task_id": task_id,
                            "reward": reward,
                            "success": reward >= 0.8,
                            "skipped": True,
                        })
                except Exception as e:
                    logger.warning(f"Failed to load existing result: {e}")
                continue
            
            logger.info(f"[{i+1}/{len(task_entries)}] Running: {task_type}/{task_id}")
            
            # Set tool filtering (if enabled)
            if distraction_count is not None:
                required_clients = [server_name]
                effective_seed = (tool_seed or 0) + i
                num_active = agent.set_active_tools(
                    required_client_names=required_clients,
                    distraction_count=distraction_count,
                    seed=effective_seed,
                )
                logger.info(f"  Active tools: {num_active} (required: {required_clients}, distraction: {distraction_count})")
            
            # Reset search server state
            reset_tool_name = f"{server_name}__reset_state"
            try:
                await agent.host.call_tool(reset_tool_name, {})
            except Exception as e:
                logger.warning(f"Failed to reset {server_name} state: {e}")
            
            # Build instruction (based on original deepresearch benchmark prompt, adapted for MCP tool calls)
            # Original version uses <search> tags, here we use search__web_search tool instead
            instruction = f"""You should pay attention to the format of your output. If You want to give the final answer, You should put the answer between <answer> and </answer>.
Question: {question}"""
            
            # Run agent (search benchmark, supports Sequential Scaling checkpoint reuse)
            # Sequential Scaling: Try to resume from previous budget's checkpoint
            prefix_checkpoint = None
            if sequential_reuse and checkpoint_store and token_budget:
                # NOTE: Use task_type (browsecomp/mind2web/webvoyager) instead of benchmark (search)
                # because save_scaling_checkpoint saves using task_type
                prefix_checkpoint = checkpoint_store.find_best_prefix_checkpoint(
                    benchmark=task_type,
                    task_id=task_id,
                    target_budget=token_budget,
                )
                if prefix_checkpoint:
                    logger.info(f"[SEQUENTIAL] Found prefix checkpoint at budget {prefix_checkpoint.budget_level}, "
                              f"will resume from it for target budget {token_budget}")
            
            if prefix_checkpoint:
                # Clean stop prompts and resume from checkpoint
                cleaned_checkpoint = checkpoint_store.clean_stop_prompts(prefix_checkpoint)
                trace = await agent.run_from_checkpoint(
                    checkpoint=cleaned_checkpoint,
                    target_budget=token_budget,
                    use_synthesize=False,
                    benchmark="search",
                )
                logger.info(f"[SEQUENTIAL] Resumed from checkpoint: {len(cleaned_checkpoint.messages)} messages recovered")
            else:
                trace = await agent.run(
                    task_id=task_id,
                    instruction=instruction,
                    policy=None,
                    use_synthesize=False,
                    benchmark="search",
                )
            
            # Get search statistics
            search_count = 0
            get_answer_tool = f"{server_name}__get_answer"
            try:
                answer_info = await agent.host.call_tool(get_answer_tool, {})
                import json as json_module
                if hasattr(answer_info, 'content'):
                    answer_str = answer_info.content[0].text if answer_info.content else "{}"
                else:
                    answer_str = str(answer_info)
                answer_data = json_module.loads(answer_str)
                search_count = answer_data.get("search_count", 0)
            except Exception as e:
                logger.warning(f"Failed to get search stats: {e}")
            
            # Parse <answer></answer> tag to get final answer (consistent with original deepresearch benchmark)
            final_answer = trace.final_response or ""
            if "<answer>" in final_answer and "</answer>" in final_answer:
                try:
                    answer_match = final_answer.split("<answer>")[1].split("</answer>")[0].strip()
                    final_answer = answer_match
                except:
                    final_answer = "did not find answer"  # Parse failed
            else:
                final_answer = "did not find answer"  # No <answer> tag
            trace.final_response = final_answer
            
            # Save trace
            trace_file = trace_dir / f"{result_name}.json"
            trace_data = {
                "benchmark": task_type,  # browsecomp/mind2web/webvoyager
                "dataset": task_type,
                "task_id": task_id,
                "question": question,
                "trace": trace.to_dict(),
                "total_steps": trace.total_steps,
                "search_count": search_count,
                "error": trace.error,
            }
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Save Scaling Checkpoint (if in Scaling mode)
            save_scaling_checkpoint(task_type, task_id, trace, agent)
            
            # Evaluate (using SearchEvaluator)
            search_evaluator = SearchEvaluator()
            eval_result = search_evaluator.evaluate(
                trace=trace,
                task=task_data,
                search_count=search_count,
            )
            
            # Save evaluation (search native format: result_{id}.json)
            eval_file = eval_dir / f"result_{task_id}.json"
            
            # Get model name
            llm_model = agent.llm.model if hasattr(agent.llm, 'model') else "unknown"
            
            # Calculate token statistics
            context_lengths = []
            for r in trace.rounds:
                context_lengths.append({
                    "input": r.prompt_tokens,
                    "output": r.output_tokens,
                    "total": r.round_total_tokens
                })
            
            # Build search native format evaluation data (fully consistent with deepresearch)
            eval_data = {
                "model": llm_model,
                "question": question,
                "answer": trace.final_response or "",
                "turns": len(trace.rounds),
                "search count": search_count,
                "script count": 0,
                "context lengths": context_lengths,
                "total_input_tokens": trace.total_prompt_tokens,
                "total_output_tokens": trace.total_output_tokens,
                "total_tokens": trace.total_tokens,
                "score": eval_result.info.get("score", 0) if eval_result.info else 0,
                "ground_truth": task_data.get("golden_answer", task_data.get("answer", "")),
            }
            
            with open(eval_file, "w") as f:
                json.dump(eval_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"  -> score={eval_result.reward:.0f}, search_count={search_count}")
            
            all_results.append({
                "benchmark": task_type,  # Use specific benchmark type (browsecomp/mind2web/webvoyager)
                "domain": task_type,
                "task_id": task_id,
                "reward": eval_result.reward,
                "success": eval_result.success,
                "total_steps": trace.total_steps,
                "search_count": search_count,
                "error": trace.error or eval_result.error,
            })
        elif benchmark == "mathhay":
            # ===== MathHay task (long-context math reasoning) =====
            # MathHay is a long-context math reasoning benchmark that does not use external tools
            # All information is provided in the prompt, the model needs to find relevant information from documents and calculate the answer
            task_id = str(task_data.get("id", f"mathhay_{i}"))
            question = task_data.get("question", "")
            golden_answer = task_data.get("golden_answer", 0.0)
            solution = task_data.get("solution", "")
            task_type_str = task_data.get("task_type", "3s3d")  # e.g., "3s3d" = 3 steps, 3 documents
            relevant_docs = task_data.get("relevant_documents", [])
            irrelevant_indices = task_data.get("irrelevant_documents_indices", [])
            
            # Generate unique result filename
            safe_task_id = str(task_id).replace("/", "_").replace("|", "_")[:50]
            result_name = f"mathhay_{safe_task_id}"
            
            # ===== Resume check =====
            trace_file = trace_dir / f"{result_name}.json"
            if resume and trace_file.exists():
                logger.info(f"[{i+1}/{len(task_entries)}] SKIP (already completed): mathhay/{task_id}")
                # Load existing results for statistics
                try:
                    eval_file = eval_dir / f"{result_name}.json"
                    if eval_file.exists():
                        with open(eval_file) as f:
                            existing_eval = json.load(f)
                        reward = existing_eval.get("reward", 0.0)
                        all_results.append({
                            "benchmark": "mathhay",
                            "domain": task_type_str,
                            "task_id": task_id,
                            "reward": reward,
                            "success": reward >= 0.8,
                            "skipped": True,
                        })
                except Exception as e:
                    logger.warning(f"Failed to load existing result: {e}")
                continue
            
            logger.info(f"[{i+1}/{len(task_entries)}] Running: mathhay/{task_id}")
            logger.info(f"  Task type: {task_type_str}, Relevant docs: {len(relevant_docs)}, Irrelevant indices: {len(irrelevant_indices)}")
            
            # Load irrelevant documents (from mathhay_documents.json)
            documents_file = task_file.parent / "mathhay_documents.json"
            if not documents_file.exists():
                logger.error(f"MathHay documents file not found: {documents_file}")
                continue
            
            with open(documents_file, "r") as f:
                all_documents_data = json.load(f)
            
            # mathhay_documents.json format: {"3s3d": [...], "2s2d": [...], ...}
            # Select the corresponding document list based on task_type
            if isinstance(all_documents_data, dict):
                all_documents = all_documents_data.get(task_type_str, [])
                if not all_documents:
                    # Try other possible keys
                    for key in all_documents_data:
                        if all_documents_data[key]:
                            all_documents = all_documents_data[key]
                            logger.warning(f"  Using documents from '{key}' instead of '{task_type_str}'")
                            break
            else:
                all_documents = all_documents_data
            
            # ===== Use original MathHay functions (evaluation.evaluation module) =====
            # Import original tokenization and decode_tokens functions
            try:
                from evaluation.evaluation import tokenization, decode_tokens, prompt_len_cal
                use_mathhay_functions = True
            except Exception as e:
                logger.warning(f"  Failed to import MathHay evaluation functions: {e}, using fallback")
                use_mathhay_functions = False
                import tiktoken
                enc = tiktoken.encoding_for_model("gpt-4o")
                def tokenization(text):
                    return enc.encode(text, disallowed_special=())
                def decode_tokens(tokens):
                    return enc.decode(tokens)
            
            # Step 1: Get irrelevant documents and tokenize
            irrelevant_docs_list = [all_documents[idx]["Document"] for idx in irrelevant_indices if idx < len(all_documents)]
            joint_irrelevant = "\n\n".join(irrelevant_docs_list)
            joint_irrelevant_tokens = tokenization(joint_irrelevant)
            
            # Step 2: Process relevant documents
            num_docs = len(relevant_docs)
            relevant_doc_tokens = [tokenization(doc) for doc in relevant_docs]
            relevant_document = "".join(relevant_docs)
            relevant_document_tokens = tokenization(relevant_document)
            
            # Step 3: Calculate prompt_len (using original prompt_len_cal function)
            if use_mathhay_functions:
                prompt_len = prompt_len_cal(None, question)  # llm parameter is not actually used
            else:
                prompt_len = 400  # fallback
            
            # Step 4: Calculate set_haystack_len
            # Original MathHay parameters: haystack_len=128000, special handling for 128K -2000 = 126000
            # Using tiktoken (gpt-4o) as tokenizer
            # 
            # Strategy: keep haystack unchanged from original, only save space through tool compression
            # - Claude models: enable tool compression (saves ~30% tokens)
            # - Other models: no compression
            # 
            # Note: Claude + distraction=all may still exceed context limit
            # - haystack (126K) + compressed tools (52K) = 178K tiktoken
            # - 178K × 1.35 ≈ 240K Claude tokens > 200K limit
            # - In this case API will return error, need to reduce distraction count
            # 
            # 2026-01-26: Reduced to 100K to support Novita API (actual limit ~122K tokens)
            # 2026-01-28: Reduced to 80K to support Fireworks AI DeepSeek-R1 (avoid exceeding context limit)
            set_haystack_len = 80000  # Reduced to 80K (original: 126000)
            
            llm_model = agent.llm.model if hasattr(agent.llm, 'model') else "unknown"
            model_lower = llm_model.lower()
            is_claude = "claude" in model_lower or "anthropic" in model_lower
            
            if is_claude:
                logger.info(f"  Claude model: haystack={set_haystack_len:,} (original), tool compression enabled")
            else:
                logger.info(f"  Model: haystack={set_haystack_len:,} (original)")
            
            # Step 5: Calculate rest_tokens
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            
            # Step 6: Truncate irrelevant tokens
            irrelevant_tokens = list(joint_irrelevant_tokens[:rest_tokens])
            if len(joint_irrelevant_tokens) > rest_tokens:
                logger.info(f"  Truncated irrelevant docs: {len(joint_irrelevant_tokens):,} -> {rest_tokens:,} tokens")
            
            # Step 7: Calculate placement (original default: placement='middle')
            placement_index = len(irrelevant_tokens) // 2
            
            # Step 8: Insert relevant documents (insert in reverse order at the same position)
            final_tokens = irrelevant_tokens
            for idx in range(num_docs - 1, -1, -1):
                final_tokens[placement_index:placement_index] = relevant_doc_tokens[idx]
            
            # Step 9: Decode to text
            full_context = decode_tokens(final_tokens)
            logger.info(f"  Final context: {len(final_tokens):,} tokens (placement: middle at {placement_index:,})")
            
            # Step 10: Build instruction (consistent with original prompt template)
            format_instructions = '''The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object should not be wrapped in triple backticks.

Here is the output schema:
```
{"properties": {"reasoning": {"title": "Reasoning", "description": "Solution process.", "type": "string"}, "answer": {"title": "Answer", "description": "The final numerical answer to the question, deduced through reasoning.", "type": "number"}}, "required": ["reasoning", "answer"]}
```'''
            
            instruction = f"""Long-Context Documents:
{full_context}

Question:
{question}

{format_instructions}
"""
            
            # Set MathHay parameters
            # Note: although original MathHay is a single-turn task, when using distraction tools
            # the model may call tools and needs extra steps to give the final answer
            # Set max_steps=5 to allow continuing after tool calls
            agent.max_steps = 5
            if hasattr(agent.llm, 'max_tokens'):
                # Thinking models need more tokens (thinking process also uses output tokens)
                # Original max_tokens=1024, but thinking models like Qwen3-Next need more
                agent.llm.max_tokens = 16384  # Increased to 16k to support thinking models
            
            # ===== Sequential Scaling: Checkpoint reuse =====
            if sequential_reuse and checkpoint_store and token_budget:
                prefix_checkpoint = checkpoint_store.find_best_prefix_checkpoint(
                    benchmark=benchmark,
                    task_id=task_id,
                    target_budget=token_budget
                )
                if prefix_checkpoint:
                    logger.info(f"[SEQUENTIAL] Found prefix checkpoint for mathhay/{task_id} "
                               f"from budget {prefix_checkpoint.budget_level} -> {token_budget}")
                    # Clean stop prompts
                    cleaned_checkpoint = checkpoint_store.clean_stop_prompts(prefix_checkpoint)
                    # Continue running from checkpoint
                    trace = await agent.run_from_checkpoint(
                        checkpoint=cleaned_checkpoint,
                        target_budget=token_budget
                    )
                else:
                    logger.info(f"[SEQUENTIAL] No prefix checkpoint found for mathhay/{task_id}, running from scratch")
                    # No reusable checkpoint, run from scratch
                    trace = await agent.run(
                        task_id=task_id,
                        instruction=instruction,
                        policy=None,
                        use_synthesize=False,
                    )
            else:
                # Non-sequential mode, run normally
                trace = await agent.run(
                    task_id=task_id,
                    instruction=instruction,
                    policy=None,
                    use_synthesize=False,
                )
            
            # Parse JSON format answer (original MathHay uses JSON schema)
            # Format: {"reasoning": "...", "answer": 123.45}
            final_answer = trace.final_response or ""
            predicted_answer = None
            predicted_reasoning = ""
            
            # Use original MathHay's extract_json_from_string function to parse JSON
            try:
                from bench_generation.utils.tools import extract_json_from_string
                parsed = extract_json_from_string(final_answer)
                if parsed and "answer" in parsed:
                    predicted_answer = float(parsed.get("answer", 0))
                    predicted_reasoning = parsed.get("reasoning", "")
                    logger.debug(f"  Parsed answer using MathHay extract_json_from_string: {predicted_answer}")
            except Exception as e:
                logger.debug(f"  Failed to use MathHay extract_json_from_string: {e}")
            
            # If original function parsing fails, try fallback method
            if predicted_answer is None:
                try:
                    import re
                    # Find JSON block
                    json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', final_answer, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        parsed = json.loads(json_str)
                        predicted_answer = float(parsed.get("answer", 0))
                        predicted_reasoning = parsed.get("reasoning", "")
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            
            # If JSON parsing fails, try <answer> tag
            if predicted_answer is None:
                if "<answer>" in final_answer and "</answer>" in final_answer:
                    try:
                        answer_str = final_answer.split("<answer>")[1].split("</answer>")[0].strip()
                        import re
                        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", answer_str)
                        if numbers:
                            predicted_answer = float(numbers[-1])
                    except:
                        pass
            
            # If still not found, try to extract numbers from the last text
            if predicted_answer is None:
                import re
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", final_answer)
                if numbers:
                    try:
                        predicted_answer = float(numbers[-1])
                    except:
                        predicted_answer = None
            
            # Save trace
            trace_file = trace_dir / f"{result_name}.json"
            trace_data = {
                "benchmark": "mathhay",
                "dataset": task_type_str,
                "task_id": task_id,
                "question": question,
                "golden_answer": golden_answer,
                "predicted_answer": predicted_answer,
                "predicted_reasoning": predicted_reasoning,
                "trace": trace.to_dict(),
                "total_steps": trace.total_steps,
                "error": trace.error,
            }
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Save Scaling Checkpoint (if in Scaling mode)
            save_scaling_checkpoint("mathhay", task_id, trace, agent)
            
            # Evaluate (using MathHayEvaluator)
            mathhay_evaluator = MathHayEvaluator(use_llm_verification=True)
            eval_result = mathhay_evaluator.evaluate(
                trace=trace,
                task=task_data,
            )
            
            # Save evaluation (MathHay format)
            eval_file = eval_dir / f"{result_name}.json"
            
            # Get model name
            llm_model = agent.llm.model if hasattr(agent.llm, 'model') else "unknown"
            
            # Build MathHay format evaluation data
            eval_data = {
                "task_id": task_id,
                "model_name": llm_model,
                "task_type": task_type_str,
                "question": question,
                "golden_answer": golden_answer,
                "predicted_answer": predicted_answer,
                "raw_response": trace.final_response,
                "score": eval_result.reward,
                "is_correct": eval_result.success,
                "numerical_match": eval_result.info.get("numerical_match", False) if eval_result.info else False,
                "llm_judge": eval_result.info.get("llm_judge", "None") if eval_result.info else "None",
                "total_tokens": trace.total_tokens,
                "total_input_tokens": trace.total_prompt_tokens,
                "total_output_tokens": trace.total_output_tokens,
                "num_relevant_docs": len(relevant_docs),
                "num_irrelevant_docs": len(irrelevant_indices),
                "context_length": len(full_context),
            }
            
            with open(eval_file, "w") as f:
                json.dump(eval_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"  -> score={eval_result.reward:.0f}, predicted={predicted_answer}, golden={golden_answer}")
            
            all_results.append({
                "benchmark": "mathhay",
                "domain": task_type_str,
                "task_id": task_id,
                "reward": eval_result.reward,
                "success": eval_result.success,
                "total_steps": trace.total_steps,
                "error": trace.error or eval_result.error,
            })
        elif benchmark == "terminalbench":
            # ===== TerminalBench task (using MCP architecture) =====
            # Note: terminalbench server runs as a standalone MCP process, switches containers via __terminalbench_switch_container
            
            task_id = task_data.get("id", f"terminalbench_{i}")
            instruction = task_data.get("instruction", "")
            runtime = task_data.get("runtime", {})
            task_path = runtime.get("task_path", "")
            
            # Generate unique result filename
            safe_task_id = str(task_id).replace("/", "_").replace("|", "_")[:50]
            result_name = f"{benchmark}_{safe_task_id}"
            
            # ===== Resume check =====
            trace_file = trace_dir / f"{result_name}.json"
            if resume and trace_file.exists():
                logger.info(f"[{i+1}/{len(task_entries)}] SKIP (already completed): {benchmark}/{task_id}")
                # Load existing results for statistics
                try:
                    eval_file = eval_dir / f"{result_name}.json"
                    if eval_file.exists():
                        with open(eval_file) as f:
                            existing_eval = json.load(f)
                        reward = existing_eval.get("reward", 0.0)
                        all_results.append({
                            "benchmark": benchmark,
                            "domain": benchmark,
                            "task_id": task_id,
                            "reward": reward,
                            "success": reward >= 1.0,
                            "skipped": True,
                        })
                except Exception as e:
                    logger.warning(f"Failed to load existing result: {e}")
                continue
            
            logger.info(f"[{i+1}/{len(task_entries)}] Running: {benchmark}/{task_id}")
            logger.info(f"  Task path: {task_path}")
            
            try:
                # Set output path
                output_path = output_dir / "terminalbench" / safe_task_id
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Switch Docker container via MCP Host internal tool call
                # Note: this is called by run.py (orchestrator), not by the Agent
                switch_result_raw = await host.call_internal_tool(
                    "__terminalbench_switch_container",
                    {
                        "task_id": task_id,
                        "task_path": task_path if task_path else None,
                        "output_dir": str(output_path),
                        "no_rebuild": True
                    }
                )
                
                # Process switch result
                if hasattr(switch_result_raw, 'content'):
                    switch_result_text = switch_result_raw.content[0].text if switch_result_raw.content else "{}"
                else:
                    switch_result_text = str(switch_result_raw)
                
                switch_result = json.loads(switch_result_text)
                
                if not switch_result.get("success"):
                    logger.error(f"  Failed to switch TerminalBench container: {switch_result.get('error')}")
                    all_results.append({
                        "benchmark": benchmark,
                        "domain": domain,
                        "task_id": task_id,
                        "reward": 0.0,
                        "success": False,
                        "total_steps": 0,
                        "error": f"Container switch failed: {switch_result.get('error')}",
                    })
                    continue
                
                logger.info(f"  Container started: {switch_result.get('container_name')} (project: {switch_result.get('project_name')})")
                
                # Use unified agent.run() (same as tau2/mcpbench/swebench)
                # Agent can only see public tools: terminalbench_execute_bash, terminalbench_read_file, etc.
                trace = await agent.run(
                    task_id=task_id,
                    instruction=instruction,
                )
                
                # Save trace
                trace_file = trace_dir / f"{result_name}.json"
                trace_data = {
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_id": task_id,
                    "task_path": task_path,
                    "trace": trace.to_dict(),
                    "total_steps": trace.total_steps,
                    "error": trace.error,
                }
                with open(trace_file, "w") as f:
                    json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
                
                # Save Scaling Checkpoint (if in Scaling mode)
                save_scaling_checkpoint(benchmark, task_id, trace, agent)
                
                # Run tests via MCP Host internal tool call
                test_result_raw = await host.call_internal_tool("__terminalbench_run_tests", {})
                if hasattr(test_result_raw, 'content'):
                    test_result_text = test_result_raw.content[0].text if test_result_raw.content else "{}"
                else:
                    test_result_text = str(test_result_raw)
                test_result = json.loads(test_result_text)
                
                test_passed = test_result.get("passed", False) if test_result.get("success") else False
                test_output = test_result.get("output", "")
                
                # Calculate reward
                eval_result_reward = 1.0 if test_passed else 0.0
                eval_result_success = test_passed
                
                # Save evaluation
                eval_file = eval_dir / f"{result_name}.json"
                llm_model = agent.llm.model if hasattr(agent.llm, 'model') else "unknown"
                
                eval_data = {
                    "task_id": task_id,
                    "model_name": llm_model,
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_path": task_path,
                    "status": "completed" if not trace.error else "error",
                    "total_rounds": len(trace.rounds),
                    "final_response": trace.final_response or "",
                    "reward": eval_result_reward,
                    "success": eval_result_success,
                    "test_passed": test_passed,
                    "test_output": test_output[-5000:] if len(test_output) > 5000 else test_output,
                    "total_tokens": trace.total_tokens,
                    "execution_time": trace.duration,
                    "note": "Evaluated with TerminalBench Docker container via MCP (unified architecture)",
                }
                
                with open(eval_file, "w") as f:
                    json.dump(eval_data, f, indent=2, default=str, ensure_ascii=False)
                
                logger.info(f"  -> reward={eval_result_reward:.2f}, success={eval_result_success}")
                
                all_results.append({
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_id": task_id,
                    "reward": eval_result_reward,
                    "success": eval_result_success,
                    "total_steps": trace.total_steps,
                    "error": trace.error,
                })
                
            except Exception as e:
                logger.error(f"  Error running TerminalBench task: {e}")
                all_results.append({
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_id": task_id,
                    "reward": 0.0,
                    "success": False,
                    "total_steps": 0,
                    "error": str(e),
                })
            # Note: container cleanup is now automatically handled by __terminalbench_switch_container during switching
            # or cleaned up uniformly during host.shutdown()
        elif benchmark == "swebench":
            # ===== SWE-Bench task (using MCP architecture) =====
            # Note: swebench server runs as a standalone MCP process, switches containers via __swebench_switch_container
            
            task_id = task_data.get("instance_id", task_data.get("id", f"swebench_{i}"))
            instruction = task_data.get("instruction", "")
            # Compatible with two formats: direct task_path or nested in runtime
            task_path = task_data.get("task_path", "") or task_data.get("runtime", {}).get("task_path", "")
            
            # swebench_instance may be provided directly, or may need to be loaded from file
            swebench_instance = task_data.get("swebench_instance", {})
            if not swebench_instance and task_path:
                # Try to load from tests/config.json
                config_path = Path(task_path) / "tests" / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        swebench_instance = json.load(f)
                    logger.debug(f"  Loaded swebench_instance from {config_path}")
            
            # Generate unique result filename
            safe_task_id = str(task_id).replace("/", "_").replace("__", "_")[:50]
            result_name = f"{benchmark}_{safe_task_id}"
            
            # ===== Resume check =====
            trace_file = trace_dir / f"{result_name}.json"
            if resume and trace_file.exists():
                logger.info(f"[{i+1}/{len(task_entries)}] SKIP (already completed): {benchmark}/{task_id}")
                # Load existing results for statistics
                try:
                    eval_file = eval_dir / f"{result_name}.json"
                    if eval_file.exists():
                        with open(eval_file) as f:
                            existing_eval = json.load(f)
                        reward = existing_eval.get("reward", 0.0)
                        all_results.append({
                            "benchmark": benchmark,
                            "domain": benchmark,
                            "task_id": task_id,
                            "reward": reward,
                            "success": reward >= 1.0,
                            "skipped": True,
                        })
                except Exception as e:
                    logger.warning(f"Failed to load existing result: {e}")
                continue
            
            logger.info(f"[{i+1}/{len(task_entries)}] Running: {benchmark}/{task_id}")
            logger.info(f"  Task path: {task_path}")
            
            try:
                # Set output path
                output_path = output_dir / "swebench" / safe_task_id
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Switch Docker container via MCP Host internal tool call
                # Note: this is called by run.py (orchestrator), not by the Agent
                switch_result_raw = await host.call_internal_tool(
                    "__swebench_switch_container",
                    {
                        "task_id": task_id,
                        "output_dir": str(output_path),
                        "no_rebuild": True
                    }
                )
                
                # Process switch result
                if hasattr(switch_result_raw, 'content'):
                    # MCP return format
                    switch_result_text = switch_result_raw.content[0].text if switch_result_raw.content else "{}"
                else:
                    switch_result_text = str(switch_result_raw)
                
                switch_result = json.loads(switch_result_text)
                
                if not switch_result.get("success"):
                    logger.error(f"  Failed to switch SWE-Bench container: {switch_result.get('error')}")
                    all_results.append({
                        "benchmark": benchmark,
                        "domain": domain,
                        "task_id": task_id,
                        "reward": 0.0,
                        "success": False,
                        "total_steps": 0,
                        "error": f"Container switch failed: {switch_result.get('error')}",
                    })
                    continue
                
                logger.info(f"  Container started: {switch_result.get('container_name')} (project: {switch_result.get('project_name')})")
                
                # Build task instruction
                full_instruction = f"""You are a software engineer working on fixing a bug in a Python repository.

The repository is located at /testbed. Please fix the following issue:

{instruction}

Use the available tools to:
1. Explore the repository structure
2. Find the relevant files
3. Understand the bug
4. Make the necessary code changes
5. Verify your changes work

When you have completed the fix, use the swebench_finish tool to submit your solution."""
                
                # Use unified agent.run() (same as tau2/mcpbench)
                # Agent can only see 3 public tools: swebench_execute_bash, swebench_str_replace_editor, swebench_finish
                trace = await agent.run(
                    task_id=task_id,
                    instruction=full_instruction,
                    benchmark="swebench",
                )
                
                # Save trace
                trace_file = trace_dir / f"{result_name}.json"
                trace_data = {
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_id": task_id,
                    "task_path": task_path,
                    "trace": trace.to_dict(),
                    "total_steps": trace.total_steps,
                    "error": trace.error,
                }
                with open(trace_file, "w") as f:
                    json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
                
                # Save Scaling Checkpoint (if in Scaling mode)
                save_scaling_checkpoint(benchmark, task_id, trace, agent)
                
                # Get patch via MCP Host internal tool call
                patch_result_raw = await host.call_internal_tool("__swebench_get_patch", {})
                if hasattr(patch_result_raw, 'content'):
                    patch_result_text = patch_result_raw.content[0].text if patch_result_raw.content else "{}"
                else:
                    patch_result_text = str(patch_result_raw)
                patch_result = json.loads(patch_result_text)
                patch = patch_result.get("patch", "") if patch_result.get("success") else ""
                
                # Run tests via MCP Host internal tool call
                test_result_raw = await host.call_internal_tool("__swebench_run_tests", {})
                if hasattr(test_result_raw, 'content'):
                    test_result_text = test_result_raw.content[0].text if test_result_raw.content else "{}"
                else:
                    test_result_text = str(test_result_raw)
                test_result = json.loads(test_result_text)
                
                # Get standard SWE-Bench evaluation results
                test_passed = test_result.get("resolved", False) if test_result.get("success") else False
                test_output = test_result.get("output", "")
                tests_status = test_result.get("tests_status", {})  # FAIL_TO_PASS, PASS_TO_PASS
                eval_report = test_result.get("report", {})
                
                # Calculate reward
                eval_result_reward = 1.0 if test_passed else 0.0
                eval_result_success = test_passed
                
                # Save evaluation
                eval_file = eval_dir / f"{result_name}.json"
                llm_model = agent.llm.model if hasattr(agent.llm, 'model') else "unknown"
                
                eval_data = {
                    "task_id": task_id,
                    "model_name": llm_model,
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_path": task_path,
                    "status": "completed" if not trace.error else "error",
                    "total_rounds": len(trace.rounds),
                    "final_response": trace.final_response or "",
                    "reward": eval_result_reward,
                    "success": eval_result_success,
                    "resolved": test_passed,  # SWE-Bench standard field
                    "patch": patch,  # Model-generated patch
                    "gold_patch": swebench_instance.get("patch", ""),  # ground truth patch
                    "patch_length": len(patch),
                    "gold_patch_length": len(swebench_instance.get("patch", "")),
                    "test_passed": test_passed,
                    # SWE-Bench standard field: detailed status for each test
                    # Format: {"FAIL_TO_PASS": {"success": [...], "failure": [...]}, 
                    #        "PASS_TO_PASS": {"success": [...], "failure": [...]}}
                    "tests_status": tests_status,
                    "report": eval_report,
                    "test_output": test_output[-5000:] if len(test_output) > 5000 else test_output,
                    "total_tokens": trace.total_tokens,
                    "execution_time": trace.duration,
                    "note": "Evaluated with SWE-Bench Docker container via MCP (unified architecture, using swebench.harness.grading)",
                }
                
                with open(eval_file, "w") as f:
                    json.dump(eval_data, f, indent=2, default=str, ensure_ascii=False)
                
                logger.info(f"  -> reward={eval_result_reward:.2f}, success={eval_result_success}, patch_length={len(patch)}")
                
                all_results.append({
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_id": task_id,
                    "reward": eval_result_reward,
                    "success": eval_result_success,
                    "total_steps": trace.total_steps,
                    "error": trace.error,
                })
                
            except Exception as e:
                logger.error(f"  Error running SWE-Bench task: {e}")
                all_results.append({
                    "benchmark": benchmark,
                    "domain": domain,
                    "task_id": task_id,
                    "reward": 0.0,
                    "success": False,
                    "total_steps": 0,
                    "error": str(e),
                })
            # Note: container cleanup is now automatically handled by __swebench_switch_container during switching
            # or cleaned up uniformly during host.shutdown()
        else:
            logger.warning(f"[{i+1}/{len(task_entries)}] Unknown benchmark: {benchmark}, skipping")
            continue
    
    # Calculate statistics
    evaluated = [r for r in all_results if r["reward"] is not None]
    total = len(all_results)
    successes = sum(1 for r in evaluated if r["success"])
    avg_reward = sum(r["reward"] for r in evaluated) / len(evaluated) if evaluated else 0
    
    # Group statistics by benchmark
    benchmark_stats = {}
    for r in evaluated:
        b = r["benchmark"]
        if b not in benchmark_stats:
            benchmark_stats[b] = {
                "total": 0, 
                "success": 0, 
                "reward_sum": 0,
                "domains": {}
            }
        benchmark_stats[b]["total"] += 1
        if r["success"]:
            benchmark_stats[b]["success"] += 1
        benchmark_stats[b]["reward_sum"] += r["reward"]
        
        # Group by domain (within benchmark)
        d = r["domain"]
        if d not in benchmark_stats[b]["domains"]:
            benchmark_stats[b]["domains"][d] = {"total": 0, "success": 0, "reward_sum": 0}
        benchmark_stats[b]["domains"][d]["total"] += 1
        if r["success"]:
            benchmark_stats[b]["domains"][d]["success"] += 1
        benchmark_stats[b]["domains"][d]["reward_sum"] += r["reward"]
    
    # Calculate benchmark-level success_rate and avg_reward
    for b, bs in benchmark_stats.items():
        bs["success_rate"] = bs["success"] / bs["total"] if bs["total"] > 0 else 0
        bs["avg_reward"] = bs["reward_sum"] / bs["total"] if bs["total"] > 0 else 0
        del bs["reward_sum"]
        
        # Calculate domain-level success_rate and avg_reward
        for d, ds in bs["domains"].items():
            ds["success_rate"] = ds["success"] / ds["total"] if ds["total"] > 0 else 0
            ds["avg_reward"] = ds["reward_sum"] / ds["total"] if ds["total"] > 0 else 0
            del ds["reward_sum"]
    
    stats = {
        "task_file": str(task_file),
        "total_tasks": total,
        "evaluated_tasks": len(evaluated),
        "successful_tasks": successes,
        "success_rate": successes / len(evaluated) if evaluated else 0,
        "average_reward": avg_reward,
        "benchmark_stats": benchmark_stats,
        "results": all_results,
    }
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_file}")
    
    return stats


async def run_tau2_benchmark(
    agent: UniversalAgent,
    domains: list[str],
    user_llm: str,
    user_llm_args: dict,
    task_type: str = "lite",
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
    output_dir: Optional[Path] = None,
    # Sequential Scaling parameters
    checkpoint_store: Optional[CheckpointStore] = None,
    token_budget: Optional[int] = None,
) -> dict:
    """
    Run tau2 benchmark (load by domain)
    
    Args:
        agent: UniversalAgent instance
        domains: List of domains to run
        user_llm: LLM model name used by UserSimulator
        user_llm_args: UserSimulator LLM parameters
        task_type: "lite" or "full"
        task_ids: Specific task IDs (optional)
        num_tasks: Number of tasks per domain (optional)
        output_dir: Results output directory
        checkpoint_store: CheckpointStore for saving scaling checkpoints
        token_budget: Token budget for scaling mode
        
    Returns:
        Run result statistics
    """
    # Create subdirectories
    trace_dir = output_dir / "traces" if output_dir else None
    eval_dir = output_dir / "evaluations" if output_dir else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
    if eval_dir:
        eval_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = Tau2Evaluator(solo_mode=False)
    all_results = []
    
    for domain in domains:
        logger.info(f"Running tau2 benchmark for domain: {domain}")
        
        # Load tasks
        tasks = load_tau2_tasks(domain, task_type)
        
        # Filter tasks
        if task_ids:
            tasks = [t for t in tasks if t.id in task_ids]
        if num_tasks:
            tasks = tasks[:num_tasks]
        
        # Load policy
        policy = load_tau2_policy(domain)
        
        # Get user tools (for User Simulator)
        user_tools = get_tau2_user_tools(domain)
        if user_tools:
            logger.info(f"Loaded {len(user_tools)} user tools for User Simulator")
        
        # Get user toolkit (for executing user tools, as fallback)
        user_toolkit = get_tau2_user_toolkit(domain)
        
        logger.info(f"Running {len(tasks)} tasks for domain {domain}")
        
        for i, task in enumerate(tasks):
            logger.info(f"[{i+1}/{len(tasks)}] Running task: {task.id}")
            
            # Generate safe filename
            # Use hash to ensure long task_ids don't collide
            import hashlib
            task_id_str = str(task.id)
            safe_task_id = task_id_str.replace("/", "_").replace("|", "_").replace("[", "").replace("]", "")[:50]
            # Add hash suffix to ensure uniqueness
            task_hash = hashlib.md5(task_id_str.encode()).hexdigest()[:8]
            result_name = f"{domain}_{safe_task_id}_{task_hash}"
            
            # ===== Reset environment and apply initial state =====
            # This is crucial for correct evaluation! Each task needs to start from the correct initial state
            try:
                # 1. Reset environment to default state
                reset_tool = f"__{domain}_reset_environment"
                await agent.host.call_tool(reset_tool, {})
                logger.debug(f"Reset environment for domain {domain}")
                
                # 2. Apply task's initial state (if any)
                if task.initial_state and task.initial_state.initialization_actions:
                    from tau2.data_model.tasks import EnvFunctionCall
                    # Convert initialization_actions to JSON
                    # Note: EnvFunctionCall uses the func_name attribute, not name
                    init_actions = []
                    for action in task.initial_state.initialization_actions:
                        init_actions.append({
                            "name": action.func_name,  # EnvFunctionCall uses func_name, not name
                            "arguments": action.arguments,
                            "env_type": action.env_type.value if hasattr(action.env_type, 'value') else str(action.env_type),
                        })
                    apply_tool = f"__{domain}_apply_initial_state"
                    await agent.host.call_tool(apply_tool, {
                        "initialization_actions_json": json.dumps(init_actions)
                    })
                    logger.debug(f"Applied {len(init_actions)} initialization actions for task {task.id}")
            except Exception as e:
                logger.warning(f"Failed to reset/initialize environment for task {task.id}: {e}")
            
            # Create MCP user tool execution callback
            async def mcp_execute_user_tool(tool_name: str, arguments: dict):
                """Call user tool via MCP"""
                return await agent.host.call_tool(tool_name, arguments)
            
            # Create UserSimulator adapter (pass user_tools, user_toolkit and MCP callback)
            user_simulator = Tau2UserSimulatorAdapter(
                user_scenario=task.user_scenario,
                user_llm=user_llm,
                user_llm_args=user_llm_args,
                user_tools=user_tools,
                user_toolkit=user_toolkit,
                mcp_execute_user_tool=mcp_execute_user_tool,
                domain=domain,
            )
            
            # Run agent
            trace = await agent.run_with_user_simulator(
                task_id=task.id,
                user_simulator=user_simulator,
                policy=policy,
            )
            
            # Save trace
            if trace_dir:
                trace_file = trace_dir / f"{result_name}.json"
                trace_data = {
                    "benchmark": "tau2",
                    "domain": domain,
                    "task_id": task.id,
                    "trace": trace.to_dict(),
                    "total_steps": trace.total_steps,
                    "error": trace.error,
                }
                with open(trace_file, "w") as f:
                    json.dump(trace_data, f, indent=2, default=str)
                
                # Save Scaling Checkpoint (if in Scaling mode)
                if checkpoint_store is not None and token_budget is not None:
                    checkpoint = ScalingCheckpoint(
                        task_id=task.id,
                        benchmark="tau2",
                        budget_level=token_budget,
                        messages=[m.to_dict() for m in trace.messages],
                        cumulative_tokens=trace.total_tokens,
                        extend_rounds=agent.extend_rounds.copy(),
                        stop_rounds=agent.stop_rounds.copy(),
                        total_prompt_tokens=trace.total_prompt_tokens,
                        total_output_tokens=trace.total_output_tokens,
                        rounds=[{
                            "round_number": r.round_number,
                            "incremental_tokens": r.incremental_tokens,
                            "cumulative_total_tokens": r.cumulative_total_tokens,
                        } for r in trace.rounds],
                        final_response=trace.final_response,
                        seed=agent.scaling_config.seed if agent.scaling_config else 42,
                    )
                    checkpoint_path = checkpoint_store.save(checkpoint)
                    logger.info(f"[SCALING] Saved checkpoint: {checkpoint_path}")
            
            # Evaluate using native evaluator
            eval_result = evaluator.evaluate(trace, task, domain)
            
            # Save evaluation
            if eval_dir:
                eval_file = eval_dir / f"{result_name}.json"
                eval_data = {
                    "benchmark": "tau2",
                    "domain": domain,
                    "task_id": task.id,
                    "reward": eval_result.reward,
                    "success": eval_result.success,
                    "error": eval_result.error,
                    "info": eval_result.info,
                }
                with open(eval_file, "w") as f:
                    json.dump(eval_data, f, indent=2, default=str)
            
            logger.info(f"Task {task.id}: reward={eval_result.reward:.2f}, success={eval_result.success}")
            
            all_results.append({
                "domain": domain,
                "task_id": task.id,
                "reward": eval_result.reward,
                "success": eval_result.success,
                "total_steps": trace.total_steps,
                "error": trace.error or eval_result.error,
            })
    
    # Calculate statistics
    total = len(all_results)
    successes = sum(1 for r in all_results if r["success"])
    avg_reward = sum(r["reward"] for r in all_results) / total if total > 0 else 0
    
    # Group statistics by domain
    domain_stats = {}
    for r in all_results:
        d = r["domain"]
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "success": 0, "reward_sum": 0}
        domain_stats[d]["total"] += 1
        if r["success"]:
            domain_stats[d]["success"] += 1
        domain_stats[d]["reward_sum"] += r["reward"]
    
    for d, s in domain_stats.items():
        s["success_rate"] = s["success"] / s["total"] if s["total"] > 0 else 0
        s["avg_reward"] = s["reward_sum"] / s["total"] if s["total"] > 0 else 0
        del s["reward_sum"]
    
    stats = {
        "total_tasks": total,
        "successful_tasks": successes,
        "success_rate": successes / total if total > 0 else 0,
        "average_reward": avg_reward,
        "domain_stats": domain_stats,
        "results": all_results,
    }
    
    # Save complete results
    if output_dir:
        output_file = output_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    
    return stats


async def run_mcpbench_benchmark(
    agent: UniversalAgent,
    tasks: list[MCPBenchTask],
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run mcp-bench benchmark
    
    mcp-bench task characteristics:
    - Single-turn tasks (no User Simulator needed)
    - Agent directly executes tasks described by fuzzy_description
    - Uses LLM-as-judge evaluation
    
    Args:
        agent: UniversalAgent instance
        tasks: MCPBenchTask task list
        output_dir: Results output directory
        
    Returns:
        Run result statistics
    """
    # Create subdirectories
    trace_dir = output_dir / "traces" if output_dir else None
    eval_dir = output_dir / "evaluations" if output_dir else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
    if eval_dir:
        eval_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = MCPBenchEvaluator(llm_model="bedrock/openai.gpt-oss-120b-1:0")
    all_results = []
    
    # Get currently available tools (for evaluation)
    available_tools = evaluator.build_available_tools(agent.get_tools_schema())
    
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] Running task: {task.task_id}")
        logger.info(f"  Servers: {task.servers}")
        
        # Generate safe filename
        safe_task_id = task.task_id.replace("/", "_").replace("|", "_")[:50]
        result_name = f"mcpbench_{safe_task_id}"
        
        # Run agent (mcp-bench is single-turn task)
        # Agent sees the fuzzy_description
        trace = await agent.run(
            task_id=task.task_id,
            instruction=task.fuzzy_description,
            policy=None,  # mcp-bench has no policy
        )
        
        # Save trace
        if trace_dir:
            trace_file = trace_dir / f"{result_name}.json"
            trace_data = {
                "benchmark": "mcpbench",
                "task_id": task.task_id,
                "servers": task.servers,
                "combination_type": task.combination_type,
                "trace": trace.to_dict(),
                "total_steps": trace.total_steps,
                "error": trace.error,
            }
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
        
        # Evaluate
        eval_result = await evaluator.evaluate_async(
            trace=trace,
            task_description=task.task_description,
            fuzzy_description=task.fuzzy_description,
            dependency_analysis=task.dependency_analysis,
            available_tools=available_tools,
        )
        
        # Calculate rule-based metrics
        total_executions = sum(1 for r in trace.rounds for _ in r.executions)
        successful_executions = sum(1 for r in trace.rounds for e in r.executions if e.get('success', False))
        execution_success_rate = successful_executions / total_executions if total_executions > 0 else 1.0
        valid_tool_name_rate = 1.0  # Assume all tool names are valid
        input_schema_compliance = 1.0  # Assume all parameters comply with schema
        
        # Calculate reward using 2-axis formula
        # s_rule = avg(valid_tool_name_rate, input_schema_compliance, execution_success_rate)
        # s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
        # reward = (s_rule + s_llm) / 2
        task_fulfillment = eval_result.info.get("task_fulfillment", 0) if eval_result.info else 0
        grounding = eval_result.info.get("grounding", 0) if eval_result.info else 0
        tool_appropriateness = eval_result.info.get("tool_appropriateness", 0) if eval_result.info else 0
        parameter_accuracy = eval_result.info.get("parameter_accuracy", 0) if eval_result.info else 0
        dependency_awareness = eval_result.info.get("dependency_awareness", 0) if eval_result.info else 0
        
        s_rule = (valid_tool_name_rate + input_schema_compliance + execution_success_rate) / 3
        s_llm = (task_fulfillment + grounding + tool_appropriateness + parameter_accuracy + dependency_awareness) / 5 / 10.0
        reward_2axis = (s_rule + s_llm) / 2
        success_2axis = reward_2axis >= 0.8
        
        # Save evaluation
        if eval_dir:
            eval_file = eval_dir / f"{result_name}.json"
            eval_data = {
                "benchmark": "mcpbench",
                "task_id": task.task_id,
                "reward": reward_2axis,
                "success": success_2axis,
                "error": eval_result.error,
                "info": eval_result.info,
                "axis_scores": {
                    "s_rule": s_rule,
                    "s_llm": s_llm,
                },
            }
            with open(eval_file, "w") as f:
                json.dump(eval_data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"  -> reward={reward_2axis:.4f} (s_rule={s_rule:.4f}, s_llm={s_llm:.4f}), success={success_2axis}")
        if eval_result.info:
            scores = eval_result.info
            logger.info(f"     Task: {scores.get('task_fulfillment', 0):.1f}, "
                       f"Grounding: {scores.get('grounding', 0):.1f}, "
                       f"Tools: {scores.get('tool_appropriateness', 0):.1f}")
        
        all_results.append({
            "task_id": task.task_id,
            "servers": task.servers,
            "combination_type": task.combination_type,
            "reward": reward_2axis,
            "success": success_2axis,
            "total_steps": trace.total_steps,
            "error": trace.error or eval_result.error,
        })
    
    # Calculate statistics
    total = len(all_results)
    successes = sum(1 for r in all_results if r["success"])
    avg_reward = sum(r["reward"] for r in all_results) / total if total > 0 else 0
    
    # Group statistics by combination_type
    type_stats = {}
    for r in all_results:
        t = r["combination_type"]
        if t not in type_stats:
            type_stats[t] = {"total": 0, "success": 0, "reward_sum": 0}
        type_stats[t]["total"] += 1
        if r["success"]:
            type_stats[t]["success"] += 1
        type_stats[t]["reward_sum"] += r["reward"]
    
    for t, s in type_stats.items():
        s["success_rate"] = s["success"] / s["total"] if s["total"] > 0 else 0
        s["avg_reward"] = s["reward_sum"] / s["total"] if s["total"] > 0 else 0
        del s["reward_sum"]
    
    stats = {
        "benchmark": "mcpbench",
        "total_tasks": total,
        "successful_tasks": successes,
        "success_rate": successes / total if total > 0 else 0,
        "average_reward": avg_reward,
        "type_stats": type_stats,
        "results": all_results,
    }
    
    # Save complete results
    if output_dir:
        output_file = output_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    
    return stats


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MCP-Benchmark Runner")
    
    # Task source configuration (choose one)
    parser.add_argument("--task-file", type=str, default=None,
                        help="Load tasks from JSON file (e.g., data/tau2_multi_domain_test.json)")
    
    # Basic configuration (when not using --task-file)
    parser.add_argument("--benchmark", type=str, default="tau2",
                        choices=["tau2", "mcpbench", "swebench", "terminalbench", "mathhay", "search", "all"],
                        help="Benchmark to run (tau2, mcpbench, swebench, terminalbench, mathhay, search, or all)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain to run (for tau2: airline, retail, telecom; for mcpbench: server name)")
    parser.add_argument("--task-type", type=str, default="lite",
                        choices=["lite", "full"],
                        help="Task type")
    parser.add_argument("--task-ids", type=str, nargs="+", default=None,
                        help="Specific task IDs to run")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Number of tasks per domain")
    
    # mcp-bench specific configuration
    parser.add_argument("--mcpbench-type", type=str, default="single_server",
                        choices=["single_server", "2_server", "3_server"],
                        help="MCP-bench task type (single_server, 2_server, 3_server)")
    parser.add_argument("--include-distractors", action="store_true", default=False,
                        help="Include distractor servers for mcp-bench tasks")
    
    # LLM configuration (defaults read from mcp-bench config file)
    parser.add_argument("--model", type=str, default="bedrock/openai.gpt-oss-120b-1:0",
                        help="LLM model name (default: bedrock/openai.gpt-oss-120b-1:0)")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"Sampling temperature (default from config: {get_temperature()})")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens")
    parser.add_argument("--use-litellm", action="store_true", default=True,
                        help="Use LiteLLM client (default: True, for Bedrock)")
    
    # User Simulator LLM configuration (tau2 specific, defaults aligned with tau2-bench)
    parser.add_argument("--user-model", type=str, default="bedrock/openai.gpt-oss-120b-1:0",
                        help="LLM model for UserSimulator (default: bedrock/openai.gpt-oss-120b-1:0)")
    parser.add_argument("--user-temperature", type=float, default=0.0,
                        help="Temperature for UserSimulator LLM (default: 0.0 for deterministic user)")
    
    # Agent configuration (defaults read from mcp-bench config file)
    parser.add_argument("--max-steps", type=int, default=None,
                        help=f"Max steps per task (default from config: {get_max_execution_rounds()})")
    parser.add_argument("--task-timeout", type=int, default=None,
                        help=f"Task timeout in seconds (default from config: {get_task_timeout()})")
    
    # Sequential Scaling configuration
    parser.add_argument("--token-budget", type=int, default=None,
                        help="Token budget for sequential scaling mode (e.g., 8000, 16000, 32000). "
                             "If set, enables scaling mode which ignores max_steps and uses EXTEND/STOP mechanisms.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for storing scaling checkpoints (default: checkpoints)")
    parser.add_argument("--scaling-seed", type=int, default=42,
                        help="Random seed for LLM calls in scaling mode, ensures prefix consistency (default: 42)")
    parser.add_argument("--sequential-reuse", action="store_true", default=False,
                        help="Enable sequential checkpoint reuse: resume from previous budget's checkpoint "
                             "instead of starting fresh. Requires --token-budget to be set.")
    parser.add_argument("--force-max-steps", action="store_true", default=False,
                        help="Force max_steps limit even in scaling mode (when --token-budget is set). "
                             "By default, scaling mode ignores max_steps and only uses token budget.")
    
    # Tool filtering configuration
    parser.add_argument("--distraction-count", type=str, default=None,
                        help="Number of distraction tools: integer N (use N distraction tools), "
                             "'all' (use all tools with logging), or None/omit (use all tools, default)")
    parser.add_argument("--tool-seed", type=int, default=None,
                        help="Random seed for distraction tool selection (for reproducibility)")
    parser.add_argument("--simulation-seed", type=int, default=300,
                        help="Random seed for simulation (same as tau2-bench --seed, default: 300)")
    
    # Tool compression configuration
    parser.add_argument("--compress-tools", action="store_true", default=False,
                        help="Compress tool descriptions to save tokens (description max 75 chars)")
    parser.add_argument("--tool-description-max-len", type=int, default=75,
                        help="Max length for tool description when --compress-tools is enabled (default: 75)")
    
    # GPT-5 compatibility mode (dynamic tool loading)
    parser.add_argument("--gpt5-mode", action="store_true", default=False,
                        help="Enable GPT-5 compatibility mode: dynamically limit tools to --max-tools per task. "
                             "Only affects mcpbench benchmark. Automatically calculates distraction count.")
    parser.add_argument("--max-tools", type=int, default=125,
                        help="Maximum tools per task in GPT-5 mode (default: 125, leaving 3 buffer for 128 limit)")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Log level")
    
    # Resume configuration
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from previous run, skip completed tasks (default: True)")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="Force re-run all tasks, do not skip completed tasks")
    
    args = parser.parse_args()
    
    # Handle resume parameters
    if args.no_resume:
        args.resume = False
    
    # Parse distraction-count parameter
    # Supports: None (no filtering), "all" (use all tools), or integer N (use N distraction tools)
    distraction_count_raw = args.distraction_count
    if distraction_count_raw is None:
        args.distraction_count = None  # No filtering, use all tools
        args.use_all_tools = False
    elif distraction_count_raw.lower() == "all":
        args.distraction_count = None  # Use all tools
        args.use_all_tools = True  # But mark as explicit "all" mode
    else:
        try:
            args.distraction_count = int(distraction_count_raw)
            args.use_all_tools = False
        except ValueError:
            raise ValueError(f"Invalid --distraction-count value: {distraction_count_raw}. "
                           "Use an integer, 'all', or omit for default.")
    
    # Apply configuration defaults
    if args.temperature is None:
        args.temperature = get_temperature()
    if args.max_steps is None:
        args.max_steps = get_max_execution_rounds()
    if args.task_timeout is None:
        args.task_timeout = get_task_timeout()
    
    # Configure logging
    logger.remove()
    logger.add(lambda msg: print(msg), level=args.log_level)
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Create LLM API
    if args.use_litellm:
        llm = LiteLLMAPI(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        llm = OpenAIAPI(
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    
    # LLM used by User Simulator (defaults to same as Agent)
    user_model = args.user_model or args.model
    
    # Generate simulation seed (same logic as tau2-bench)
    # tau2-bench: random.seed(seed); seeds = [random.randint(0, 1000000) for _ in range(num_trials)]
    import random
    random.seed(args.simulation_seed)
    simulation_seed = random.randint(0, 1000000)  # seed for trial 0
    logger.info(f"Simulation seed: {simulation_seed} (from base seed {args.simulation_seed})")
    
    user_llm_args = {
        "temperature": args.user_temperature,  # Use independent user temperature (default 0.0 deterministic)
        "seed": simulation_seed,  # simulation seed aligned with tau2-bench
    }
    
    # Determine tasks (but servers are always loaded in full)
    mcpbench_tasks = []
    domains = ["airline", "retail", "telecom"]  # default tau2 domains
    
    if args.task_file:
        # Determine needed domains from task file
        task_entries = load_tasks_from_file(Path(args.task_file))
        domains_needed = set(e["domain"] for e in task_entries)
        logger.info(f"Domains needed from task file: {domains_needed}")
    elif args.benchmark == "tau2":
        if args.domain:
            domains = [args.domain]
        else:
            domains = ["airline", "retail", "telecom"]
    elif args.benchmark == "mcpbench":
        # Load mcp-bench tasks
        mcpbench_tasks = load_mcpbench_tasks(
            combination_types=[args.mcpbench_type],
            server_names=[args.domain] if args.domain else None,
            max_tasks=args.num_tasks,
            include_distractors=args.include_distractors,
        )
        
        if not mcpbench_tasks:
            logger.error("No mcp-bench tasks found!")
            return
        
        logger.info(f"Loaded {len(mcpbench_tasks)} mcp-bench tasks")
    
    # Always load all servers (tau2 3 + mcp-bench 28 = 31)
    # GPT-5 mode: only load servers needed by task file domains
    if args.gpt5_mode and args.task_file:
        # GPT-5 mode: only load required servers
        logger.info("GPT-5 mode: Loading only required servers for task file...")
        task_entries = load_tasks_from_file(Path(args.task_file))
        
        # Collect all required mcpbench servers
        required_mcpbench_servers = set()
        for entry in task_entries:
            if entry.get("benchmark") == "mcpbench":
                domain = entry.get("domain", "")
                servers_for_domain = get_mcpbench_servers_for_domain(domain)
                required_mcpbench_servers.update(servers_for_domain)
        
        logger.info(f"Required MCP-Bench servers: {required_mcpbench_servers}")
        
        # Build server configuration: load tau2 servers + all mcpbench servers
        # tau2 servers as distraction source, mcpbench includes required and other distraction
        servers = {}
        servers.update(DEFAULT_SERVERS)  # tau2 servers (schema fixed)
        
        # Load all mcpbench servers (required + as distraction source)
        mcpbench_configs = get_mcpbench_server_configs()
        for name, raw_config in mcpbench_configs.items():
            host_config = convert_to_host_config(name, raw_config)
            if host_config:
                servers[name] = host_config
        
        logger.info(f"GPT-5 mode: Loading {len(servers)} servers (tau2 + all mcpbench)")
    else:
        # Normal mode: load all servers
        logger.info("Loading ALL servers (tau2 + mcp-bench)...")
        print("DEBUG: About to call get_all_servers()", flush=True)
        servers = get_all_servers()
    
    print(f"DEBUG: Got {len(servers)} servers", flush=True)
    logger.info(f"Total servers to load: {len(servers)}")
    
    # Run benchmark
    print("DEBUG: Entering BenchmarkHost context", flush=True)
    async with BenchmarkHost() as host:
        # Load all servers
        print("DEBUG: Inside BenchmarkHost, starting to load servers", flush=True)
        logger.info("Initializing ALL MCP servers...")
        
        # Load servers one by one, skip failed ones
        loaded_count = 0
        failed_servers = []
        for idx, (name, config) in enumerate(servers.items(), 1):
            try:
                print(f"DEBUG: Loading server {idx}/{len(servers)}: {name}", flush=True)
                logger.info(f"Loading server {idx}/{len(servers)}: {name}")
                await host.create_clients_from_config({name: config})
                loaded_count += 1
                print(f"DEBUG: Successfully loaded {name}", flush=True)
                logger.info(f"  -> Successfully loaded {name}")
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                # Some MCP servers can fail to initialize due to missing optional
                # dependencies (node, scipy, etc.) or internal cancellation.
                # Treat all such failures as non-fatal and continue loading.
                if isinstance(e, asyncio.CancelledError):
                    # anyio/stdio_client can cancel the current task when a server
                    # dies during startup; clear cancellation so we can continue.
                    task = asyncio.current_task()
                    if task is not None and hasattr(task, "uncancel"):
                        task.uncancel()

                logger.warning(f"Failed to load server '{name}': {type(e).__name__}: {e}")
                failed_servers.append(name)
        
        logger.info(f"Successfully loaded {loaded_count}/{len(servers)} servers")
        if failed_servers:
            logger.warning(f"Failed servers: {failed_servers}")
        
        # Display tool information
        tools = host.get_tools_schema()
        logger.info(f"Total tools available: {len(tools)}")
        
        # Tool compression configuration (controlled via command line arguments)
        compress_tools = args.compress_tools
        tool_description_max_len = args.tool_description_max_len
        if compress_tools:
            logger.info(f"Tool compression: ENABLED (description max {tool_description_max_len} chars, full param descriptions)")
        
        # Create ScalingConfig (if scaling mode is enabled)
        scaling_config = None
        checkpoint_store = None
        if args.token_budget:
            scaling_config = ScalingConfig(seed=args.scaling_seed)
            checkpoint_store = CheckpointStore(args.checkpoint_dir)
            logger.info(f"Scaling mode: ENABLED (budget={args.token_budget}, seed={args.scaling_seed}, checkpoint_dir={args.checkpoint_dir})")
        
        # Create Agent (using config parameters)
        agent = UniversalAgent(
            host=host,
            llm_client=llm,
            max_steps=args.max_steps,
            task_timeout=args.task_timeout,
            compress_tools=compress_tools,
            tool_description_max_len=tool_description_max_len,
            # Sequential Scaling parameters
            target_budget=args.token_budget,
            scaling_config=scaling_config,
            force_max_steps=args.force_max_steps,
        )
        
        logger.info(f"Agent config: max_steps={args.max_steps}, task_timeout={args.task_timeout}s, temperature={args.temperature}, user_temperature={args.user_temperature}")
        if args.token_budget:
            logger.info(f"Scaling mode: token_budget={args.token_budget}, seed={args.scaling_seed}, force_max_steps={args.force_max_steps}")
        if args.gpt5_mode:
            logger.info(f"GPT-5 mode: ENABLED (max_tools={args.max_tools}, dynamic distraction calculation)")
            logger.info(f"  -> Total loaded tools: {len(tools)} (should be < 128 for GPT-5 compatibility)")
        if args.distraction_count is not None:
            logger.info(f"Tool filtering: distraction_count={args.distraction_count}, seed={args.tool_seed}")
        elif args.use_all_tools:
            logger.info(f"Tool filtering: ALL (using all {len(tools)} tools explicitly)")
        else:
            logger.info(f"Tool filtering: DISABLED (using all {len(tools)} tools)")
        
        # Run
        if args.task_file:
            # Load tasks from JSON file (supports mixed tau2 + mcpbench)
            stats = await run_from_task_file(
                agent=agent,
                task_file=Path(args.task_file),
                user_llm=user_model,
                user_llm_args=user_llm_args,
                output_dir=output_dir,
                host=host,  # Pass BenchmarkHost instance
                distraction_count=args.distraction_count,
                tool_seed=args.tool_seed,
                simulation_seed=simulation_seed,
                resume=args.resume,
                # Scaling parameters
                checkpoint_store=checkpoint_store,
                token_budget=args.token_budget,
                sequential_reuse=args.sequential_reuse,
                # GPT-5 compatibility mode parameters
                gpt5_mode=args.gpt5_mode,
                max_tools=args.max_tools,
            )
        elif args.benchmark == "tau2":
            stats = await run_tau2_benchmark(
                agent=agent,
                domains=domains,
                user_llm=user_model,
                user_llm_args=user_llm_args,
                task_type=args.task_type,
                task_ids=args.task_ids,
                num_tasks=args.num_tasks,
                output_dir=output_dir,
                # Scaling parameters
                checkpoint_store=checkpoint_store,
                token_budget=args.token_budget,
            )
        elif args.benchmark == "mcpbench":
            stats = await run_mcpbench_benchmark(
                agent=agent,
                tasks=mcpbench_tasks,
                output_dir=output_dir,
            )
        elif args.benchmark == "swebench":
            # SWE-Bench: load tasks and run
            swebench_tasks = load_swebench_tasks(
                task_ids=args.task_ids,
                max_tasks=args.num_tasks,
            )
            if not swebench_tasks:
                logger.error("No SWE-Bench tasks found!")
                return
            
            # Construct task_entries format
            task_entries = []
            for task in swebench_tasks:
                task_entries.append({
                    "benchmark": "swebench",
                    "domain": "swebench",
                    **task
                })
            
            stats = await run_from_task_file(
                agent=agent,
                task_file=None,  # Directly pass task list
                task_entries=task_entries,
                user_llm=user_model,
                user_llm_args=user_llm_args,
                output_dir=output_dir,
                host=host,  # Pass BenchmarkHost instance
                distraction_count=args.distraction_count,
                tool_seed=args.tool_seed,
                simulation_seed=simulation_seed,
                resume=args.resume,
            )
        elif args.benchmark == "terminalbench":
            # TerminalBench: load tasks and run
            terminalbench_tasks = load_terminalbench_tasks(
                task_ids=args.task_ids,
                max_tasks=args.num_tasks,
            )
            if not terminalbench_tasks:
                logger.error("No TerminalBench tasks found!")
                return
            
            # Construct task_entries format
            task_entries = []
            for task in terminalbench_tasks:
                task_entries.append({
                    "benchmark": "terminalbench",
                    "domain": "terminalbench",
                    **task
                })
            
            stats = await run_from_task_file(
                agent=agent,
                task_file=None,  # Directly pass task list
                task_entries=task_entries,
                user_llm=user_model,
                user_llm_args=user_llm_args,
                output_dir=output_dir,
                host=host,  # Pass BenchmarkHost instance
                distraction_count=args.distraction_count,
                tool_seed=args.tool_seed,
                simulation_seed=simulation_seed,
                resume=args.resume,
            )
        else:
            # Default: run tau2
            stats = await run_tau2_benchmark(
                agent=agent,
                domains=domains,
                user_llm=user_model,
                user_llm_args=user_llm_args,
                task_type=args.task_type,
                task_ids=args.task_ids,
                num_tasks=args.num_tasks,
                output_dir=output_dir,
            )
        
        # Output results
        logger.info("=" * 50)
        logger.info("Benchmark Results:")
        logger.info(f"  Total tasks: {stats['total_tasks']}")
        logger.info(f"  Successful: {stats['successful_tasks']}")
        logger.info(f"  Success rate: {stats['success_rate']:.2%}")
        logger.info(f"  Average reward: {stats['average_reward']:.4f}")
        
        # Output statistics by benchmark
        if "benchmark_stats" in stats:
            for benchmark, bs in stats["benchmark_stats"].items():
                logger.info(f"  [{benchmark}] {bs['success']}/{bs['total']} ({bs['success_rate']:.2%}), avg_reward={bs['avg_reward']:.4f}")
                for domain, ds in bs["domains"].items():
                    logger.info(f"    - {domain}: {ds['success']}/{ds['total']} ({ds['success_rate']:.2%}), avg_reward={ds['avg_reward']:.4f}")
        elif "domain_stats" in stats:
            logger.info("  Per-domain stats:")
            for domain, ds in stats["domain_stats"].items():
                logger.info(f"    {domain}: {ds['success']}/{ds['total']} ({ds['success_rate']:.2%}), avg_reward={ds['avg_reward']:.4f}")
        elif "type_stats" in stats:
            logger.info("  Per-type stats (mcpbench):")
            for task_type, ts in stats["type_stats"].items():
                logger.info(f"    {task_type}: {ts['success']}/{ts['total']} ({ts['success_rate']:.2%}), avg_reward={ts['avg_reward']:.4f}")
        logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
