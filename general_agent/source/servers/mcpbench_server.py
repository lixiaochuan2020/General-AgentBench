"""
MCP-Bench Server Configurations

Convert mcp-bench MCP server configurations to a format usable by omni BenchmarkHost.

mcp-bench has 30+ independent MCP servers (Google Maps, Wikipedia, Math, etc.),
each with its own startup command and environment configuration. These servers are independent processes communicating via stdio.

This module is responsible for:
1. Loading mcp-bench server configurations (commands.json)
2. Converting configurations to the format needed by BenchmarkHost.create_client()
3. Filtering servers based on task requirements
"""

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

# mcp-bench path
MCP_BENCH_PATH = Path(__file__).parent.parent.parent.parent / "benchmarks" / "mcp-bench"
MCP_SERVERS_PATH = MCP_BENCH_PATH / "mcp_servers"


# ============================================================================
# Server name mapping (mcp-bench name -> normalized name)
# ============================================================================
SERVER_NAME_MAP = {
    "OpenAPI Explorer": "openapi_explorer",
    "Unit Converter": "unit_converter",
    "Wikipedia": "wikipedia",
    "Google Maps": "google_maps",
    "Bibliomantic": "bibliomantic",
    "BioMCP": "biomcp",
    "Call for Papers": "call_for_papers",
    "Car Price Evaluator": "car_price",
    "Context7": "context7",
    "DEX Paprika": "dex_paprika",
    "FruityVice": "fruityvice",
    "Game Trends": "game_trends",
    "Huge Icons": "huge_icons",
    "Hugging Face": "huggingface",
    "Math MCP": "math",
    "NixOS": "nixos",
    "OSINT Intelligence": "osint",
    "Reddit": "reddit",
    "National Parks": "national_parks",
    "Medical Calculator": "medcalc",
    "Metropolitan Museum": "metmuseum",
    "Movie Recommender": "movie_recommender",
    "NASA Data": "nasa",
    "OKX Exchange": "okx",
    "Paper Search": "paper_search",
    "Scientific Computing": "scientific_computing",
    "Weather Data": "weather",
    "Time MCP": "time_mcp",
    # New benchmark tool servers
    "SWEBench": "swebench",
    "TerminalBench": "terminalbench",
}


def get_mcpbench_server_configs() -> dict[str, dict]:
    """
    Load mcp-bench server configurations
    
    Returns:
        Server configuration dict, format: {name: {cmd, env, cwd, transport?, port?, endpoint?}}
    """
    commands_file = MCP_SERVERS_PATH / "commands.json"
    
    if not commands_file.exists():
        logger.error(f"mcp-bench commands.json not found at {commands_file}")
        return {}
    
    with open(commands_file) as f:
        raw_configs = json.load(f)
    
    return raw_configs


def _resolve_cwd(cwd_relative: str) -> str:
    """Resolve relative working directory to absolute path
    
    The cwd format in commands.json is "../xxx", relative to the parent of the mcp_servers directory.
    But these servers are actually in the mcp_servers directory.
    So we need to convert "../xxx" to "xxx".
    """
    if not cwd_relative:
        return str(MCP_SERVERS_PATH)
    
    # Handle "../xxx" format - these servers are actually in the mcp_servers directory
    if cwd_relative.startswith("../"):
        # Strip the "../" prefix
        clean_path = cwd_relative[3:]
        resolved = MCP_SERVERS_PATH / clean_path
    else:
        resolved = MCP_SERVERS_PATH / cwd_relative
    
    resolved = resolved.resolve()
    
    # Log warning if path does not exist
    if not resolved.exists():
        logger.warning(f"Server cwd does not exist: {resolved}")
    
    return str(resolved)


def _load_api_keys() -> dict[str, str]:
    """Load mcp-bench API key file"""
    api_key_file = MCP_SERVERS_PATH / "api_key"
    api_keys = {}
    
    if api_key_file.exists():
        with open(api_key_file) as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    api_keys[key.strip()] = value.strip()
    
    return api_keys


# Cache API keys
_API_KEYS_CACHE = None


def _get_api_keys() -> dict[str, str]:
    """Get API keys (with cache)"""
    global _API_KEYS_CACHE
    if _API_KEYS_CACHE is None:
        _API_KEYS_CACHE = _load_api_keys()
    return _API_KEYS_CACHE


def _build_env(env_keys: list[str]) -> dict[str, str]:
    """Build environment variable dict from current environment and API key file"""
    env = os.environ.copy()
    
    # Load mcp-bench API keys
    api_keys = _get_api_keys()
    
    # Add specific environment variables needed by mcp-bench
    for key in env_keys:
        if key in os.environ:
            env[key] = os.environ[key]
        elif key in api_keys:
            env[key] = api_keys[key]
            logger.debug(f"Loaded {key} from api_key file")
        else:
            # Some API keys may be missing, log warning
            logger.debug(f"Environment variable {key} not set")
    
    return env


def convert_to_host_config(server_name: str, raw_config: dict) -> Optional[dict]:
    """
    Convert mcp-bench server configuration to a format usable by BenchmarkHost
    
    Args:
        server_name: Server name in mcp-bench
        raw_config: mcp-bench raw configuration
        
    Returns:
        Configuration needed by BenchmarkHost.create_client(), format:
        {
            "command": str,      # Startup command
            "args": list[str],   # Command arguments
            "env": dict,         # Environment variables
            "cwd": str,          # Working directory
            "transport": str,    # Transport type ("stdio" or "http")
            "port": int,         # HTTP port (HTTP transport only)
            "endpoint": str,     # HTTP endpoint (HTTP transport only)
        }
    """
    transport = raw_config.get("transport", "stdio")
    
    # Parse command
    cmd_str = raw_config.get("cmd", "")
    if not cmd_str:
        logger.warning(f"No command specified for {server_name}")
        return None
    
    cmd_parts = cmd_str.split()
    if not cmd_parts:
        return None
    
    command = cmd_parts[0]
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    
    # Handle working directory
    cwd = _resolve_cwd(raw_config.get("cwd", ""))
    
    # Build environment variables
    env = _build_env(raw_config.get("env", []))
    
    if transport == "http":
        # HTTP transport
        return {
            "command": command,
            "args": args,
            "command_list": cmd_parts,  # Full command list
            "env": env,
            "cwd": cwd,
            "transport": "http",
            "port": raw_config.get("port", 3001),
            "endpoint": raw_config.get("endpoint", "/mcp"),
        }
    else:
        # STDIO transport
        # If there is a working directory, use bash -c to execute commands with cd
        if cwd and cwd != str(MCP_SERVERS_PATH):
            # Use bash -c to execute commands with cd
            full_cmd = f"cd {cwd} && {cmd_str}"
            return {
                "command": "bash",
                "args": ["-c", full_cmd],
                "env": env,
                "cwd": cwd,
                "transport": "stdio",
            }
        else:
            return {
                "command": command,
                "args": args,
                "env": env,
                "cwd": cwd,
                "transport": "stdio",
            }


def get_server_config(server_name: str) -> Optional[dict]:
    """
    Get configuration for a single server
    
    Args:
        server_name: mcp-bench server name
        
    Returns:
        BenchmarkHost-compatible configuration
    """
    all_configs = get_mcpbench_server_configs()
    
    if server_name not in all_configs:
        logger.warning(f"Server '{server_name}' not found in mcp-bench configs")
        return None
    
    return convert_to_host_config(server_name, all_configs[server_name])


def get_servers_for_task(
    task_servers: list[str], 
    distraction_servers: list[str] | None = None,
    prefix: str = "mcpbench"
) -> dict[str, dict]:
    """
    Get server configurations needed for a task
    
    Args:
        task_servers: List of servers needed by the task (mcp-bench names)
        distraction_servers: List of distraction servers (optional)
        prefix: Server name prefix
        
    Returns:
        Server configuration dict, format: {prefixed_name: config}
    """
    all_configs = get_mcpbench_server_configs()
    
    # Merge needed servers
    needed_servers = set(task_servers)
    if distraction_servers:
        needed_servers.update(distraction_servers)
    
    result = {}
    for server_name in needed_servers:
        if server_name not in all_configs:
            logger.warning(f"Server '{server_name}' not found in mcp-bench configs")
            continue
            
        config = convert_to_host_config(server_name, all_configs[server_name])
        if config:
            # Use normalized name
            safe_name = SERVER_NAME_MAP.get(server_name, server_name.lower().replace(" ", "_"))
            key = f"{prefix}_{safe_name}" if prefix else safe_name
            result[key] = config
    
    return result


# ============================================================================
# Predefined server groups (for quick testing and batch loading)
# ============================================================================

# Simple test servers (fast startup, no API key needed)
SIMPLE_SERVERS = ["Math MCP", "FruityVice"]

# Medium complexity servers
MEDIUM_SERVERS = ["Math MCP", "Wikipedia", "Weather Data"]

# All servers involved in single-server tasks
ALL_SINGLE_SERVERS = [
    "Bibliomantic", "BioMCP", "Call for Papers", "Car Price Evaluator",
    "Context7", "DEX Paprika", "FruityVice", "Game Trends", 
    "Huge Icons", "Hugging Face", "Math MCP", "Medical Calculator",
    "Metropolitan Museum", "Movie Recommender", "NASA Data", "National Parks",
    "NixOS", "OKX Exchange", "OpenAPI Explorer", "OSINT Intelligence",
    "Paper Search", "Reddit", "Scientific Computing", "Time MCP",
    "Unit Converter", "Weather Data", "Wikipedia"
]

# Servers that do not need API keys (can be tested directly)
NO_API_KEY_SERVERS = [
    "Math MCP", "FruityVice", "Bibliomantic", "Call for Papers",
    "Car Price Evaluator", "DEX Paprika", "Huge Icons", "Medical Calculator",
    "Metropolitan Museum", "Movie Recommender", "OKX Exchange", 
    "OpenAPI Explorer", "Scientific Computing", "Unit Converter"
]


def get_simple_test_servers() -> dict[str, dict]:
    """Get server configs for simple testing"""
    return get_servers_for_task(SIMPLE_SERVERS)


def get_no_api_key_servers() -> dict[str, dict]:
    """Get server configs that do not require API keys"""
    return get_servers_for_task(NO_API_KEY_SERVERS)


def get_all_single_servers() -> dict[str, dict]:
    """Get server configs needed by all single-server tasks"""
    return get_servers_for_task(ALL_SINGLE_SERVERS)


def list_available_servers() -> list[str]:
    """List all available mcp-bench servers"""
    configs = get_mcpbench_server_configs()
    return list(configs.keys())
