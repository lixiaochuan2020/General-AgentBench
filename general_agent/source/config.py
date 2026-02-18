"""Configuration Loader Module for MCP-Benchmark.

This module provides unified management of all MCP-Bench configurations,
supporting YAML file configuration with environment variable overrides.

Copied from mcp-bench/config/config_loader.py to maintain consistency with the original framework.
"""

import yaml
import os
from typing import Dict, Any, Union, Optional, List
from pathlib import Path


class BenchmarkConfig:
    """Singleton configuration manager for benchmark settings.
    
    This class manages all benchmark configurations using a singleton pattern,
    loading settings from YAML files and allowing environment variable overrides.
    Configuration priority: environment variables > YAML file > default values.
    """
    
    _instance: Optional['BenchmarkConfig'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls) -> 'BenchmarkConfig':
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the configuration if not already loaded."""
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file with environment overrides."""
        # First try to load the original mcp-bench config file
        mcp_bench_config = Path(__file__).parent.parent.parent / 'benchmarks' / 'mcp-bench' / 'config' / 'benchmark_config.yaml'
        # Fallback: local config file
        local_config = Path(__file__).parent / 'benchmark_config.yaml'
        
        config_path = mcp_bench_config if mcp_bench_config.exists() else local_config
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_path}: {e}")
                self._config = self._get_default_config()
        else:
            print(f"Config file not found, using default configuration")
            self._config = self._get_default_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration matching mcp-bench benchmark_config.yaml."""
        return {
            'mcp': {
                'connection': {
                    'http_timeout': 60,
                    'tool_discovery_timeout': 10,
                    'server_startup_timeout': 30,
                    'health_check_timeout': 2,
                    'process_wait_timeout': 5,
                    'batch_timeout': 60
                },
                'ports': {
                    'default_port': 3001,
                    'port_search_attempts': 100,
                    'random_port_min': 10000,
                    'random_port_max': 50000
                }
            },
            'execution': {
                'task_timeout': 5000,
                'task_retry_max': 1,
                'retry_delay': 5,
                'compression_retries': 1,
                'max_execution_rounds': 200,
                'server_semaphore_limit': 20,
                'context_budget': None,
                'sequential_only_tools': [
                    "Google_Maps__search_nearby",
                    "Google_Maps__get_place_details",
                    "Google_Maps__maps_geocode",
                    "Google_Maps__maps_reverse_geocode",
                    "Google_Maps__maps_distance_matrix",
                    "Google_Maps__maps_directions",
                    "Google_Maps__maps_elevation",
                ],
                'problematic_tools': [
                    "Paper_Search__search_semantic",
                    "Paper_Search__download_semantic",
                    "Paper_Search__read_semantic_paper",
                    "OSINT_Intelligence__osint_overview",
                    "Paper_Search__search_iacr",
                    "Paper_Search__read_iacr_paper",
                    "Paper_Search__download_iacr",
                    "BioMCP__alphagenome_predictor",
                ],
                'content_summary_threshold': 1000,
                'content_truncate_length': 4000,
                'error_truncate_length': 1000,
                'error_display_prefix': 200
            },
            'benchmark': {
                'distraction_servers_default': 10,
                'resident_servers': ["Time MCP"],
                'task_delay': 1,
                'all_task_files': [
                    "./tasks/mcpbench_tasks_single_runner_format.json",
                    "./tasks/mcpbench_tasks_multi_2server_runner_format.json",
                    "./tasks/mcpbench_tasks_multi_3server_runner_format.json",
                ],
                'tasks_file': None,
                'enable_judge_stability': True,
                'enable_server_preselection': False,
                'filter_problematic_tools': True,
                'concurrent_summarization': True,
                'use_fuzzy_descriptions': True,
                'enable_concrete_description_ref_for_eval': True,
                'enable_dependency_analysis_ref_for_eval': True
            },
            'cache': {
                'enabled': False,
                'cache_dir': "./cache",
                'ttl_hours': 0,
                'max_size_mb': 1000,
                'key_strategy': "hash",
                'log_stats': True,
                'cleanup_interval': 0,
                'persistent': True,
                'server_whitelist': []
            },
            'evaluation': {
                'judge_stability_runs': 5,
                'judge_model': {
                    'name': "bedrock/openai.gpt-oss-120b-1:0",
                    'provider': "litellm"
                }
            },
            'llm': {
                'temperature': 0.7,
                'json_retry_groups': 5,
                'token_reduction_factors': [0.9, 0.8, 0.7],
                'min_tokens': 1000,
                'token_increment': 1000,
                'format_conversion_tokens': 8000,
                'planning_tokens': 12000,
                'summarization_max_tokens': 10000,
                'user_prompt_max_length': 30000
            },
            'data_collection': {
                'individual_timeout': 30,
                'batch_timeout': 60,
                'max_retries': 20,
                'retry_delay_base': 2,
                'retry_delay_multiplier': 2,
                'batch_retry_delay_base': 5,
                'batch_retry_delay_multiplier': 3,
                'default_http_port': 3000,
                'tool_description_truncate': 150
            },
            'task_generation': {
                'tasks_per_combination': 1,
                'generation_max_retries': 1
            },
            'dependency_extraction': {
                'required_support_count': 3,
                'min_support_count': 2
            },
            'server_selection': {
                'selection_tokens': 8000,
                'tool_sample_count': 3
            },
            'azure': {
                'api_version': "2024-12-01-preview"
            }
        }
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable configuration overrides."""
        env_mapping = {
            'EXECUTION_TASK_TIMEOUT': 'execution.task_timeout',
            'MCP_CONNECTION_HTTP_TIMEOUT': 'mcp.connection.http_timeout',
            'EXECUTION_MAX_EXECUTION_ROUNDS': 'execution.max_execution_rounds',
            'EXECUTION_COMPRESSION_RETRIES': 'execution.compression_retries',
            'LLM_TEMPERATURE': 'llm.temperature',
            'BENCHMARK_FILTER_PROBLEMATIC_TOOLS': 'benchmark.filter_problematic_tools',
        }
        
        for key, value in os.environ.items():
            if key.startswith('BENCHMARK_'):
                env_suffix = key[10:]  # Remove BENCHMARK_ prefix
                
                if env_suffix in env_mapping:
                    config_path = env_mapping[env_suffix]
                else:
                    config_path = env_suffix.lower().replace('_', '.')
                
                try:
                    converted_value = self._convert_env_value(value)
                    self._set_nested_value(self._config, config_path, converted_value)
                except Exception as e:
                    print(f"Warning: Failed to apply environment override {key}={value}: {e}")
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable values to appropriate types."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        try:
            return float(value)
        except ValueError:
            pass
        
        return value
    
    def _set_nested_value(
        self,
        config: Dict[str, Any],
        path: str,
        value: Any
    ) -> None:
        """Set value in nested dictionary using dot-separated path."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value through dot-separated path."""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})
    
    def reload(self) -> None:
        """Reload configuration from file and environment."""
        self._config = None
        self._load_config()


# Create global configuration instance
config = BenchmarkConfig()


# ============ Convenience Functions ============

def get_config(key_path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value."""
    return config.get(key_path, default)


# === MCP Connection ===
def get_mcp_timeout() -> int:
    """Get MCP HTTP timeout."""
    return config.get('mcp.connection.http_timeout', 60)

def get_tool_discovery_timeout() -> int:
    """Get tool discovery timeout."""
    return config.get('mcp.connection.tool_discovery_timeout', 10)

def get_server_startup_timeout() -> int:
    """Get server startup timeout."""
    return config.get('mcp.connection.server_startup_timeout', 30)


# === Execution ===
def get_task_timeout() -> int:
    """Get task execution timeout (seconds)."""
    return config.get('execution.task_timeout', 5000)

def get_max_retries() -> int:
    """Get maximum retry count."""
    return config.get('execution.task_retry_max', 1)

def get_retry_delay() -> int:
    """Get retry delay time (seconds)."""
    return config.get('execution.retry_delay', 5)

def get_max_execution_rounds() -> int:
    """Get maximum execution rounds."""
    return config.get('execution.max_execution_rounds', 200)

def get_server_semaphore_limit() -> int:
    """Get server concurrency semaphore limit."""
    return config.get('execution.server_semaphore_limit', 20)

def get_content_summary_threshold() -> int:
    """Get content summary token threshold."""
    return config.get('execution.content_summary_threshold', 1000)

def get_content_truncate_length() -> int:
    """Get content truncation length."""
    return config.get('execution.content_truncate_length', 4000)

def get_error_truncate_length() -> int:
    """Get error message truncation length."""
    return config.get('execution.error_truncate_length', 1000)

def get_error_display_prefix() -> int:
    """Get error display prefix length."""
    return config.get('execution.error_display_prefix', 200)

def get_sequential_only_tools() -> List[str]:
    """Get list of tools that must be called sequentially."""
    return config.get('execution.sequential_only_tools', [])

def get_problematic_tools() -> List[str]:
    """Get list of problematic tools to filter out."""
    return config.get('execution.problematic_tools', [])


# === Benchmark ===
def get_distraction_servers_count() -> int:
    """Get default distraction server count."""
    return config.get('benchmark.distraction_servers_default', 10)

def get_resident_servers() -> List[str]:
    """Get list of resident servers (always available)."""
    return config.get('benchmark.resident_servers', ["Time MCP"])

def get_task_delay() -> int:
    """Get inter-task delay time (seconds)."""
    return config.get('benchmark.task_delay', 1)

def filter_problematic_tools_enabled() -> bool:
    """Check if problematic tool filtering is enabled."""
    return config.get('benchmark.filter_problematic_tools', True)

def use_fuzzy_descriptions() -> bool:
    """Check if fuzzy descriptions should be used."""
    return config.get('benchmark.use_fuzzy_descriptions', True)


# === LLM ===
def get_temperature() -> float:
    """Get LLM temperature."""
    return config.get('llm.temperature', 0.7)

def get_min_tokens() -> int:
    """Get minimum token count."""
    return config.get('llm.min_tokens', 1000)

def get_token_increment() -> int:
    """Get token increment."""
    return config.get('llm.token_increment', 1000)

def get_planning_tokens() -> int:
    """Get planning token limit."""
    return config.get('llm.planning_tokens', 12000)

def get_summarization_max_tokens() -> int:
    """Get summarization maximum tokens."""
    return config.get('llm.summarization_max_tokens', 10000)

def get_user_prompt_max_length() -> int:
    """Get user prompt maximum length."""
    return config.get('llm.user_prompt_max_length', 30000)


# === Evaluation ===
def get_judge_model() -> str:
    """Get judge model name."""
    return config.get('evaluation.judge_model.name', "bedrock/openai.gpt-oss-120b-1:0")

def get_judge_stability_runs() -> int:
    """Get judge stability test run count."""
    return config.get('evaluation.judge_stability_runs', 5)


# === Tool Filtering ===

# Bedrock-compatible tool name separator
MCPBENCH_SEPARATOR = "__"

def _sanitize_name(name: str) -> str:
    """Sanitize a name to comply with Bedrock API requirements [a-zA-Z0-9_-]+"""
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def _convert_old_format_to_new(tool_identifier: str) -> str:
    """
    Convert old format (colon-separated) to new format (double-underscore-separated).
    
    Old format: "Paper Search:search_semantic"
    New format: "Paper_Search__search_semantic"
    
    Args:
        tool_identifier: Tool identifier (may be in old or new format)
        
    Returns:
        Tool identifier in new format
    """
    # If already in new format (contains __), return as-is
    if MCPBENCH_SEPARATOR in tool_identifier:
        return tool_identifier
    
    # Old format uses colon separator
    if ':' in tool_identifier:
        parts = tool_identifier.split(':', 1)
        if len(parts) == 2:
            server_name, tool_name = parts
            safe_server = _sanitize_name(server_name)
            safe_tool = _sanitize_name(tool_name)
            return f"{safe_server}{MCPBENCH_SEPARATOR}{safe_tool}"
    
    # Unrecognized format, return as-is
    return tool_identifier


def _get_normalized_problematic_tools() -> set:
    """Get normalized set of problematic tools (converted to Bedrock-compatible format)"""
    raw_tools = get_problematic_tools()
    return {_convert_old_format_to_new(t) for t in raw_tools}


def _get_normalized_sequential_tools() -> set:
    """Get normalized set of sequential-only tools (converted to Bedrock-compatible format)"""
    raw_tools = get_sequential_only_tools()
    return {_convert_old_format_to_new(t) for t in raw_tools}


def should_filter_tool(server_name: str, tool_name: str) -> bool:
    """
    Check if a tool should be filtered out.
    
    Args:
        server_name: Name of the server
        tool_name: Name of the tool
        
    Returns:
        True if the tool should be filtered out
    """
    if not filter_problematic_tools_enabled():
        return False
    
    # Use Bedrock-compatible format for matching
    safe_server = _sanitize_name(server_name)
    safe_tool = _sanitize_name(tool_name)
    tool_identifier = f"{safe_server}{MCPBENCH_SEPARATOR}{safe_tool}"
    
    # Get normalized set of problematic tools
    problematic = _get_normalized_problematic_tools()
    
    return tool_identifier in problematic


def is_sequential_only_tool(server_name: str, tool_name: str) -> bool:
    """
    Check if a tool must be called sequentially.
    
    Args:
        server_name: Name of the server
        tool_name: Name of the tool
        
    Returns:
        True if the tool must be called sequentially
    """
    # Use Bedrock-compatible format for matching
    safe_server = _sanitize_name(server_name)
    safe_tool = _sanitize_name(tool_name)
    tool_identifier = f"{safe_server}{MCPBENCH_SEPARATOR}{safe_tool}"
    
    # Get normalized set of sequential-only tools
    sequential = _get_normalized_sequential_tools()
    
    return tool_identifier in sequential


def filter_tools_by_config(
    tools: Dict[str, Any],
    server_name: str = None
) -> Dict[str, Any]:
    """
    Filter tools based on configuration.
    
    Args:
        tools: Dictionary of tools {tool_name: tool_info}
        server_name: Optional server name for filtering
        
    Returns:
        Filtered tools dictionary
    """
    if not filter_problematic_tools_enabled():
        return tools
    
    filtered = {}
    for tool_name, tool_info in tools.items():
        # If server_name is not provided, try to get it from tool_info
        srv_name = server_name or tool_info.get('server', 'unknown')
        
        if not should_filter_tool(srv_name, tool_name):
            filtered[tool_name] = tool_info
    
    return filtered
