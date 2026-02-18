"""
MCP-Bench Task Loader

Load and process mcp-bench task files.

Task file format:
{
    "generation_info": {...},
    "server_tasks": [
        {
            "server_name": "OpenAPI Explorer",  # Primary server name (single-server task)
            "servers": ["OpenAPI Explorer"],     # List of required servers
            "combination_type": "single_server", # single_server, 2_server, 3_server
            "tasks": [
                {
                    "task_id": "openapi_explorer_000",
                    "task_description": "...",      # Specific task (used for evaluation)
                    "fuzzy_description": "...",     # Fuzzy task (seen by Agent)
                    "dependency_analysis": "...",   # Dependency analysis (used for evaluation)
                    "distraction_servers": [...]    # Distraction servers
                },
                ...
            ]
        },
        ...
    ],
    "total_tasks": 100
}
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

# mcp-bench path
MCP_BENCH_PATH = Path(__file__).parent.parent.parent / "benchmarks" / "mcp-bench"
TASKS_PATH = MCP_BENCH_PATH / "tasks"


@dataclass
class MCPBenchTask:
    """mcp-bench task data structure"""
    task_id: str
    task_description: str      # Specific task description (used as evaluation reference)
    fuzzy_description: str     # Fuzzy task description (seen by Agent)
    dependency_analysis: str   # Dependency analysis
    servers: list[str]         # Required servers
    distraction_servers: list[str] = field(default_factory=list)  # Distraction servers
    combination_type: str = "single_server"  # Task type
    server_name: str = ""      # Primary server name (single-server task)
    
    @property
    def all_servers(self) -> list[str]:
        """Get all servers to start (including distraction servers)"""
        return list(set(self.servers + self.distraction_servers))
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "fuzzy_description": self.fuzzy_description,
            "dependency_analysis": self.dependency_analysis,
            "servers": self.servers,
            "distraction_servers": self.distraction_servers,
            "combination_type": self.combination_type,
            "server_name": self.server_name,
        }


def load_mcpbench_tasks(
    task_file: Optional[str] = None,
    combination_types: Optional[list[str]] = None,
    server_names: Optional[list[str]] = None,
    max_tasks: Optional[int] = None,
    include_distractors: bool = True,
) -> list[MCPBenchTask]:
    """
    Load mcp-bench tasks
    
    Args:
        task_file: Task file path, defaults to single_runner_format
        combination_types: Task types to filter ["single_server", "2_server", "3_server"]
        server_names: Server names to filter
        max_tasks: Maximum number of tasks
        include_distractors: Whether to include distraction servers
        
    Returns:
        List of MCPBenchTask
    """
    if task_file is None:
        task_file = TASKS_PATH / "mcpbench_tasks_single_runner_format.json"
    else:
        task_file = Path(task_file)
    
    if not task_file.exists():
        logger.error(f"Task file not found: {task_file}")
        return []
    
    with open(task_file) as f:
        data = json.load(f)
    
    tasks = []
    
    for server_group in data.get("server_tasks", []):
        group_servers = server_group.get("servers", [])
        combination_type = server_group.get("combination_type", "single_server")
        server_name = server_group.get("server_name", "")
        
        # Filter by task type
        if combination_types and combination_type not in combination_types:
            continue
        
        # Filter by server
        if server_names:
            if not any(s in group_servers for s in server_names):
                continue
        
        for task_data in server_group.get("tasks", []):
            task = MCPBenchTask(
                task_id=task_data.get("task_id", ""),
                task_description=task_data.get("task_description", ""),
                fuzzy_description=task_data.get("fuzzy_description", ""),
                dependency_analysis=task_data.get("dependency_analysis", ""),
                servers=group_servers,
                distraction_servers=task_data.get("distraction_servers", []) if include_distractors else [],
                combination_type=combination_type,
                server_name=server_name,
            )
            tasks.append(task)
            
            if max_tasks and len(tasks) >= max_tasks:
                return tasks
    
    logger.info(f"Loaded {len(tasks)} tasks from {task_file}")
    return tasks


def load_single_server_tasks(
    server_name: Optional[str] = None,
    max_tasks: Optional[int] = None,
    include_distractors: bool = True,
) -> list[MCPBenchTask]:
    """
    Load single-server tasks
    
    Args:
        server_name: Specific server name (optional)
        max_tasks: Maximum number of tasks
        include_distractors: Whether to include distraction servers
        
    Returns:
        List of MCPBenchTask
    """
    server_names = [server_name] if server_name else None
    return load_mcpbench_tasks(
        combination_types=["single_server"],
        server_names=server_names,
        max_tasks=max_tasks,
        include_distractors=include_distractors,
    )


def load_multi_server_tasks(
    combination_type: str = "2_server",
    max_tasks: Optional[int] = None,
    include_distractors: bool = True,
) -> list[MCPBenchTask]:
    """
    Load multi-server tasks
    
    Args:
        combination_type: "2_server" or "3_server"
        max_tasks: Maximum number of tasks
        include_distractors: Whether to include distraction servers
        
    Returns:
        List of MCPBenchTask
    """
    return load_mcpbench_tasks(
        combination_types=[combination_type],
        max_tasks=max_tasks,
        include_distractors=include_distractors,
    )


def get_task_by_id(task_id: str) -> Optional[MCPBenchTask]:
    """
    Get a single task by ID
    
    Args:
        task_id: Task ID
        
    Returns:
        MCPBenchTask or None
    """
    tasks = load_mcpbench_tasks()
    for task in tasks:
        if task.task_id == task_id:
            return task
    return None


def list_available_servers_in_tasks() -> list[str]:
    """
    List all servers referenced in the task file
    
    Returns:
        List of server names
    """
    tasks = load_mcpbench_tasks(include_distractors=True)
    servers = set()
    for task in tasks:
        servers.update(task.all_servers)
    return sorted(servers)


def get_tasks_for_server(server_name: str) -> list[MCPBenchTask]:
    """
    Get all tasks related to a specific server
    
    Args:
        server_name: Server name
        
    Returns:
        List of tasks
    """
    all_tasks = load_mcpbench_tasks()
    return [t for t in all_tasks if server_name in t.servers]


# Create omni-compatible task format
def create_mixed_task_entry(task: MCPBenchTask) -> dict:
    """
    Create a task entry compatible with the omni task file
    
    Used for mixing with tau2 tasks in the same task file
    
    Args:
        task: MCPBenchTask
        
    Returns:
        Task entry dictionary
    """
    return {
        "benchmark": "mcpbench",
        "domain": task.server_name or "mcpbench",
        "task": task.to_dict(),
    }


def export_for_general_agent(
    output_file: str,
    tasks: list[MCPBenchTask],
) -> None:
    """
    Export tasks in omni format
    
    Args:
        output_file: Output file path
        tasks: List of tasks
    """
    entries = [create_mixed_task_entry(t) for t in tasks]
    
    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(entries)} tasks to {output_file}")
