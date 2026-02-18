"""
MCP-Benchmark Host

Following the official MCP architecture design:
- Host:  BenchmarkHost (coordinator, manages multiple Clients)
- Client: Each ClientSession instance (1:1 connection with a Server)
- Server: Individual Benchmark Server processes

Extended support:
- OpenHands Runtime Client: Uses OpenHands Docker Runtime directly in-process
  (for SWE-Bench tasks, fully consistent with the original benchmark)
"""

import asyncio
import json
import logging
import os
import random
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional, Callable

import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import (
    filter_problematic_tools_enabled,
    should_filter_tool,
    is_sequential_only_tool,
)

# Add OpenHands to path
OPENHANDS_PATH = Path(__file__).parent.parent.parent / "benchmarks" / "OpenHands"
if str(OPENHANDS_PATH) not in sys.path:
    sys.path.insert(0, str(OPENHANDS_PATH))

logger = logging.getLogger(__name__)


def convert_schema_for_openai(schema: dict) -> dict:
    """
    Convert JSON Schema 2020-12 format to OpenAI-compatible format
    
    Main conversions:
    - prefixItems -> items (OpenAI does not support prefixItems)
    - Empty items: {} -> items: {"type": "object"} (GPT-5 requires items to have a type)
    
    This conversion is backwards-compatible and does not affect other models.
    
    Args:
        schema: JSON Schema (may contain prefixItems)
        
    Returns:
        Converted schema (using items)
    """
    if not isinstance(schema, dict):
        return schema
    
    result = {}
    for key, value in schema.items():
        if key == "prefixItems":
            # Convert prefixItems to items
            # prefixItems: [{"type": "integer"}, {"type": "integer"}]
            # -> items: {"type": "integer"} (take the first element's type, usually the same)
            if isinstance(value, list) and len(value) > 0:
                # Check if all element types are the same
                types = [item.get("type") for item in value if isinstance(item, dict)]
                if types and all(t == types[0] for t in types):
                    result["items"] = {"type": types[0]}
                else:
                    # Types are different, use anyOf
                    result["items"] = {"anyOf": value}
        elif key == "items" and isinstance(value, dict) and not value:
            # Fix empty items: {} -> {"type": "object"}
            # GPT-5 requires array items to have a complete type definition
            result["items"] = {"type": "object"}
        elif isinstance(value, dict):
            # Recursively handle nested schema
            result[key] = convert_schema_for_openai(value)
        elif isinstance(value, list):
            # Recursively handle elements in lists
            result[key] = [
                convert_schema_for_openai(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result


@dataclass
class HTTPTool:
    """Simple representation of an HTTP transport tool"""
    name: str
    description: str
    inputSchema: dict


@dataclass
class MCPClient:
    """
    Represents an MCP Client, maintaining a 1:1 connection with a Server
    
    Per MCP specification:  "Each client is created by the host and maintains 
    an isolated server connection"
    
    Extended support for OpenHands Runtime Client (transport_type="openhands")
    """
    name: str
    session: ClientSession | None = None  # None for HTTP/OpenHands transport
    tools: list = field(default_factory=list)
    transport_type: str = "stdio"  # "stdio", "http", or "openhands"
    
    # HTTP transport specific fields
    port: int | None = None
    endpoint: str = "/mcp"
    session_id: str | None = None
    server_process: subprocess.Popen | None = None
    
    # NOTE: OpenHands Runtime fields removed - SWE-Bench now handled via Terminal-Bench
    
    def get_tool_names(self) -> list[str]:
        """Get all tool names provided by the server connected to this client"""
        return [tool.name for tool in self.tools]


class BenchmarkHost:
    """
    MCP Host - Following the official architecture design
    
    Responsibilities (per MCP specification):
    - Creates and manages multiple client instances
    - Controls client connection permissions and lifecycle
    - Coordinates AI/LLM integration and sampling
    - Manages context aggregation across clients
    """
    
    # Internal tool name list (these tools are not exposed to the Agent, only for internal framework use)
    # Format: original tool name (the name used when registering with the MCP server)
    INTERNAL_TOOL_NAMES = {
        "reset_state",      # Reset server state
        "get_answer",       # Get the submitted answer
        "set_answer",       # Set answer (used internally by search server)
    }
    
    # Tools with __ prefix are also internal tools (used for swebench/terminalbench Docker bridge mode)
    # These tools are called by run.py (orchestrator) and not exposed to the Agent
    INTERNAL_TOOL_PREFIX = "__"
    
    def __init__(self):
        self.clients: dict[str, MCPClient] = {}
        self.tool_to_client: dict[str, str] = {}  # key: tool_key (server:tool or tool_name)
        self._internal_tools: set[str] = set()    # Set of tool_keys for internal tools
        self._exit_stack = AsyncExitStack()
        self._initialized = False
        self._http_session: aiohttp.ClientSession | None = None
    
    # mcp-bench tool name separator (used to distinguish server and tool)
    # Using __ because Bedrock API only allows [a-zA-Z0-9_-]+
    MCPBENCH_SEPARATOR = "__"
    
    def _is_internal_tool(self, tool_name: str) -> bool:
        """
        Determine whether a tool is an internal tool (not exposed to the Agent)
        
        Internal tools include:
        1. Tools in the INTERNAL_TOOL_NAMES set (e.g., reset_state, get_answer)
        2. Tools with __ prefix (e.g., __swebench_switch_container)
        """
        # Check if it is in the internal tool name list
        if tool_name in self.INTERNAL_TOOL_NAMES:
            return True
        
        # Check if it starts with __ prefix
        if tool_name.startswith(self.INTERNAL_TOOL_PREFIX):
            return True
        
        return False
    
    def _is_tau2_server(self, client_name: str) -> bool:
        """Determine if this is a tau2 server (tau2 uses domain_tool format, others use server__tool format)"""
        return client_name.startswith("tau2-")
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize name to comply with Bedrock API requirements [a-zA-Z0-9_-]+
        
        Conversion rules:
        - Space -> _
        - Colon -> _
        - Other illegal characters -> _
        """
        import re
        # Replace illegal characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        return sanitized
    
    def _get_tool_key(self, client_name: str, tool_name: str) -> str:
        """
        Get the unique key for a tool (Bedrock-compatible format)
        
        Format:
        - tau2: tool_name (e.g., telecom_get_customer)
        - mcp-bench: ServerName__tool_name (e.g., BioMCP__think, OpenAPI_Explorer__getApiOverview)
        
        Note: Using __ as separator because Bedrock API only allows [a-zA-Z0-9_-]+
        """
        if self._is_tau2_server(client_name):
            return tool_name  # tau2: telecom_get_customer
        else:
            # mcp-bench: sanitize server name, use __ as separator
            safe_server_name = self._sanitize_name(client_name)
            safe_tool_name = self._sanitize_name(tool_name)
            return f"{safe_server_name}{self.MCPBENCH_SEPARATOR}{safe_tool_name}"  # BioMCP__think
    
    async def __aenter__(self):
        await self._exit_stack.__aenter__()
        self._http_session = aiohttp.ClientSession()
        self._initialized = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    async def create_client(
        self, 
        name: str, 
        command: str, 
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> MCPClient:
        """
        Create a new MCP Client and connect to the corresponding Server (STDIO transport)
        
        Args:
            name: Client/Server name
            command: Command to start the Server
            args: Command arguments
            env: Environment variables
            cwd: Working directory
            
        Returns:
            MCPClient: The created client instance
        """
        logger.info(f"Creating MCP Client for server: {name}")

        # IMPORTANT:
        # We must not leave partially-initialized stdio/session contexts registered
        # in the host-level AsyncExitStack when startup fails. Doing so can trigger
        # cascading cancellations (anyio cancel scopes) later during shutdown or
        # subsequent tool calls.
        #
        # Use a per-client AsyncExitStack and only attach it to the host on success.
        client_stack = AsyncExitStack()
        try:
            params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
            transport = await client_stack.enter_async_context(stdio_client(params))
            read, write = transport

            session = await client_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await client_stack.aclose()
            raise

        # Attach this client's cleanup to the host stack.
        self._exit_stack.push_async_exit(client_stack)
        
        # Get the tools exposed by this Server
        tools_response = await session.list_tools()
        tools = []
        
        # Handle different response format versions
        if hasattr(tools_response, 'tools'):
            tools = list(tools_response.tools)
        else:
            # Legacy version compatibility
            for item in tools_response: 
                if isinstance(item, tuple) and item[0] == "tools":
                    tools = list(item[1])
                    break
        
        # Create MCPClient object
        client = MCPClient(name=name, session=session, tools=tools, transport_type="stdio")
        self.clients[name] = client
        
        # Build tool -> client routing map (filter problematic tools)
        filtered_count = 0
        for tool in tools:
            # Check if this tool needs to be filtered
            if should_filter_tool(name, tool.name):
                logger.debug(f"Filtering problematic tool: {name}:{tool.name}")
                filtered_count += 1
                continue
            
            # Generate tool key (tau2 uses tool_name, others use server:tool format)
            tool_key = self._get_tool_key(name, tool.name)
            
            if tool_key in self.tool_to_client:
                logger.warning(
                    f"Tool '{tool_key}' already registered by client "
                    f"'{self.tool_to_client[tool_key]}', overwriting with '{name}'"
                )
            self.tool_to_client[tool_key] = name
            
            # Mark internal tools (not exposed to Agent)
            if self._is_internal_tool(tool.name):
                self._internal_tools.add(tool_key)
            
            # Mark tools that require serial invocation
            if is_sequential_only_tool(name, tool.name):
                if not hasattr(self, '_sequential_only_tools'):
                    self._sequential_only_tools = set()
                self._sequential_only_tools.add(tool_key)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} problematic tools from '{name}'")
        
        logger.info(f"Client '{name}' created with {len(tools) - filtered_count} tools: {client.get_tool_names()}")
        return client

    async def create_http_client(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        port: int = 3001,
        endpoint: str = "/mcp",
    ) -> MCPClient:
        """
        Create an HTTP transport MCP Client
        
        Following the mcp-bench approach: start an HTTP server process, communicate via JSON-RPC over HTTP
        """
        logger.info(f"Creating HTTP MCP Client for server: {name}")
        
        # Find available port
        actual_port = await self._find_available_port(port)
        
        # Update port in command
        updated_command = self._update_command_port(command, port, actual_port)
        
        # Prepare environment variables
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        full_env['MCP_SERVER_PORT'] = str(actual_port)
        
        # Start server process
        logger.info(f"Starting HTTP server for {name} on port {actual_port}")
        logger.info(f"Command: {' '.join(updated_command)}")
        
        server_process = subprocess.Popen(
            updated_command,
            cwd=cwd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        await asyncio.sleep(3)
        
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            raise Exception(f"HTTP server failed to start for {name}: {stderr}")
        
        # Initialize and discover tools
        base_url = f"http://localhost:{actual_port}{endpoint}"
        session_id = None
        
        try:
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "all-in-one-benchmark", "version": "1.0.0"}
                }
            }
            
            async with self._http_session.post(
                base_url,
                json=init_request,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json, text/event-stream'
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Initialization failed: {error_text}")
                
                session_id = response.headers.get('mcp-session-id')
                
                # Handle SSE response
                content_type = response.headers.get('content-type', '')
                if 'text/event-stream' in content_type:
                    response_text = await response.text()
                    lines = response_text.strip().split('\n')
                    for line in lines:
                        if line.startswith('data: '):
                            try:
                                json.loads(line[6:])
                                break
                            except json.JSONDecodeError:
                                continue
            
            # Get tool list
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream'
            }
            if session_id:
                headers['mcp-session-id'] = session_id
            
            async with self._http_session.post(
                base_url,
                json=tools_request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Tools list failed: {error_text}")
                
                content_type = response.headers.get('content-type', '')
                if 'text/event-stream' in content_type:
                    response_text = await response.text()
                    lines = response_text.strip().split('\n')
                    for line in lines:
                        if line.startswith('data: '):
                            try:
                                result = json.loads(line[6:])
                                break
                            except json.JSONDecodeError:
                                continue
                else:
                    result = await response.json()
                
                tools_data = result.get('result', {}).get('tools', [])
            
            # Create Tool objects (using simple dataclass to simulate)
            tools = [HTTPTool(
                name=t['name'],
                description=t.get('description', ''),
                inputSchema=t.get('inputSchema', {})
            ) for t in tools_data]
            
            # Create MCPClient object
            client = MCPClient(
                name=name,
                session=None,
                tools=tools,
                transport_type="http",
                port=actual_port,
                endpoint=endpoint,
                session_id=session_id,
                server_process=server_process
            )
            self.clients[name] = client
            
            # Build tool -> client routing map
            filtered_count = 0
            for tool in tools:
                if should_filter_tool(name, tool.name):
                    logger.debug(f"Filtering problematic tool: {name}:{tool.name}")
                    filtered_count += 1
                    continue
                
                # Generate tool key (tau2 uses tool_name, others use server:tool format)
                tool_key = self._get_tool_key(name, tool.name)
                
                if tool_key in self.tool_to_client:
                    logger.warning(
                        f"Tool '{tool_key}' already registered by client "
                        f"'{self.tool_to_client[tool_key]}', overwriting with '{name}'"
                    )
                self.tool_to_client[tool_key] = name
                
                # Mark internal tools (not exposed to Agent)
                if self._is_internal_tool(tool.name):
                    self._internal_tools.add(tool_key)
            
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} problematic tools from '{name}'")
            
            logger.info(f"HTTP Client '{name}' created with {len(tools) - filtered_count} tools")
            return client
            
        except Exception as e:
            # Clean up server process
            if server_process:
                server_process.terminate()
                await asyncio.sleep(1)
                if server_process.poll() is None:
                    server_process.kill()
            raise

    async def _find_available_port(self, start_port: int = 3001, max_attempts: int = 100) -> int:
        """Find an available port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        # If configured port range is unavailable, use a random port
        for _ in range(max_attempts):
            port = random.randint(10000, 50000)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        raise RuntimeError(f"Could not find available port")

    def _update_command_port(self, command: list[str], original_port: int, new_port: int) -> list[str]:
        """Update port number in command"""
        if original_port == new_port:
            return command
        
        updated = []
        i = 0
        while i < len(command):
            arg = command[i]
            if f"--port {original_port}" in arg:
                updated.append(arg.replace(f"--port {original_port}", f"--port {new_port}"))
            elif f"--port={original_port}" in arg:
                updated.append(arg.replace(f"--port={original_port}", f"--port={new_port}"))
            elif arg == "--port":
                updated.append(arg)
                i += 1
                if i < len(command) and command[i] == str(original_port):
                    updated.append(str(new_port))
                elif i < len(command):
                    updated.append(command[i])
            else:
                updated.append(arg)
            i += 1
        return updated
    
    async def create_clients_from_config(self, config: dict[str, dict]) -> None:
        """
        Batch create Clients from a config dictionary
        
        Args:
            config: Config dictionary, format: {name: {command, args, env?, transport?, port?, endpoint?, cwd?}}
        """
        for name, server_config in config.items():
            transport = server_config.get("transport", "stdio")
            
            if transport == "http":
                # HTTP transport
                command = server_config.get("command_list") or [server_config["command"]] + server_config.get("args", [])
                await self.create_http_client(
                    name=name,
                    command=command,
                    env=server_config.get("env"),
                    cwd=server_config.get("cwd"),
                    port=server_config.get("port", 3001),
                    endpoint=server_config.get("endpoint", "/mcp"),
                )
            # NOTE: openhands transport removed - SWE-Bench now handled via Terminal-Bench
            else:
                # STDIO transport (default)
                await self.create_client(
                    name=name,
                    command=server_config["command"],
                    args=server_config.get("args", []),
                    env=server_config.get("env"),
                    cwd=server_config.get("cwd"),
                )
    
    # NOTE: create_openhands_client method removed - SWE-Bench now handled via Terminal-Bench
    
    def list_all_tools(self) -> list:
        """
        Aggregate tools from all Servers connected via Clients
        
        Per MCP specification: Host "Manages context aggregation across clients"
        """
        all_tools = []
        for client in self.clients.values():
            all_tools.extend(client.tools)
        return all_tools
    
    def get_tools_for_agent(self) -> list[dict]:
        """
        Get tools visible to the Agent (filtering out internal tools)
        
        Return format is consistent with get_tools_schema() (OpenAI-compatible format)
        
        Filter rules:
        - Filter tools in INTERNAL_TOOL_NAMES
        - Filter tools with __ prefix (e.g., __swebench_switch_container)
        """
        schema = []
        for tool_key, client_name in self.tool_to_client.items():
            # Skip internal tools (not exposed to Agent)
            if tool_key in self._internal_tools:
                continue
            
            # Check if tool_key starts with __ (for cases not using server:tool format)
            if tool_key.startswith(self.INTERNAL_TOOL_PREFIX):
                continue
            
            # Check if tool_key contains a tool name with __ prefix (e.g., swebench__swebench_switch_container)
            if self.MCPBENCH_SEPARATOR in tool_key:
                tool_name_part = tool_key.split(self.MCPBENCH_SEPARATOR, 1)[1]
                if tool_name_part.startswith(self.INTERNAL_TOOL_PREFIX):
                    continue
            
            client = self.clients[client_name]
            
            # Extract original tool_name from tool_key
            if self.MCPBENCH_SEPARATOR in tool_key:
                sanitized_tool_name = tool_key.split(self.MCPBENCH_SEPARATOR, 1)[1]
                found_tool = None
                for tool in client.tools:
                    if self._sanitize_name(tool.name) == sanitized_tool_name:
                        found_tool = tool
                        break
                if found_tool:
                    schema.append({
                        "type": "function",
                        "function": {
                            "name": tool_key,
                            "description": found_tool.description or "",
                            "parameters": convert_schema_for_openai(found_tool.inputSchema or {"type": "object", "properties": {}})
                        }
                    })
            else:
                for tool in client.tools:
                    if tool.name == tool_key:
                        schema.append({
                            "type": "function",
                            "function": {
                                "name": tool_key,
                                "description": tool.description or "",
                                "parameters": convert_schema_for_openai(tool.inputSchema or {"type": "object", "properties": {}})
                            }
                        })
                        break
        
        return schema
    
    async def call_internal_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> Any:
        """
        Call an internal tool (for use by run.py orchestrator)
        
        These tools start with __ prefix and are not exposed to the Agent.
        For example: __swebench_switch_container, __swebench_run_tests, etc.
        
        Args:
            tool_name: Internal tool name (e.g., __swebench_switch_container)
            arguments: Tool parameters
            
        Returns:
            Call result
            
        Raises:
            ValueError: If the tool does not exist or is not an internal tool
        """
        # Verify it is an internal tool
        if not tool_name.startswith(self.INTERNAL_TOOL_PREFIX):
            raise ValueError(
                f"Tool '{tool_name}' is not an internal tool. "
                f"Internal tools must start with '{self.INTERNAL_TOOL_PREFIX}'."
            )
        
        # Find the client that owns this tool
        if tool_name not in self.tool_to_client:
            # Try looking up with server__tool format
            found = False
            for client_name, client in self.clients.items():
                for tool in client.tools:
                    if tool.name == tool_name:
                        # Direct call
                        logger.debug(f"Calling internal tool '{tool_name}' via client '{client_name}'")
                        if client.transport_type == "http":
                            result = await self._call_tool_http(client, tool_name, arguments)
                        else:
                            result = await client.session.call_tool(tool_name, arguments)
                        return result
            
            if not found:
                raise ValueError(f"Internal tool '{tool_name}' not found in any client.")
        
        # Use standard routing
        result, _ = await self.route_tool_call(tool_name, arguments)
        return result
    
    def get_client_names(self) -> list[str]:
        """Get all client names"""
        return list(self.clients.keys())
    
    def get_tools_by_clients(self, client_names: list[str]) -> list[str]:
        """
        Get all tool names for specified clients (returns tool_key format)
        
        Args:
            client_names: List of client names
            
        Returns:
            List of tool keys (e.g., search__web_search)
        """
        tool_keys = []
        for name in client_names:
            if name in self.clients:
                client = self.clients[name]
                for tool in client.tools:
                    # Use tool_key format (consistent with keys in tool_to_client)
                    tool_key = self._get_tool_key(name, tool.name)
                    if tool_key in self.tool_to_client:
                        tool_keys.append(tool_key)
        return tool_keys
    
    def select_distraction_tools(
        self, 
        required_client_names: list[str],
        distraction_count: int = 10,
        seed: int | None = None,
        excluded_client_names: list[str] | None = None,
    ) -> list[str]:
        """
        Randomly select distraction tools from non-required server tools (tool level)
        
        Args:
            required_client_names: Required client names (task-related servers)
            distraction_count: Number of distraction **tools** to select
            seed: Random seed (for reproducibility)
            excluded_client_names: Client names to completely exclude (e.g., exclude terminalbench when running swebench)
            
        Returns:
            List of selected distraction tool names
        """
        import random as rand
        if seed is not None:
            rand.seed(seed)
        
        # Get required tools (exclude these)
        required_tools = set(self.get_tools_by_clients(required_client_names))
        
        # Get tools to exclude
        excluded_tools = set()
        if excluded_client_names:
            excluded_tools = set(self.get_tools_by_clients(excluded_client_names))
            logger.info(f"Excluding {len(excluded_tools)} tools from clients: {excluded_client_names}")
        
        # Get all non-required tools (also exclude excluded)
        all_tools = set(self.tool_to_client.keys())
        available_tools = list(all_tools - required_tools - excluded_tools)
        
        if not available_tools:
            return []
        
        # distraction_count == -1 means select all distraction tools
        if distraction_count < 0:
            logger.info(f"Selected all {len(available_tools)} distraction tools")
            return available_tools
        
        # Randomly select distraction_count tools
        selected_count = min(distraction_count, len(available_tools))
        selected_tools = rand.sample(available_tools, selected_count)
        
        logger.info(f"Selected {selected_count} distraction tools from {len(available_tools)} available")
        
        return selected_tools
    
    def get_filtered_tools_schema(
        self,
        required_client_names: list[str],
        distraction_count: int = 10,
        seed: int | None = None,
        compress: bool = False,
        max_description_len: int = 50,
        excluded_client_names: list[str] | None = None,
        minimal: bool = False,
    ) -> list[dict]:
        """
        Get filtered tool schema (required tools + distraction tools)
        
        Args:
            required_client_names: Required client names (task-related servers)
            distraction_count: Number of distraction **tools** (not servers)
            seed: Random seed
            compress: Whether to compress tool definitions (for long context scenarios like Claude)
            max_description_len: Max description length (chars) when compressing, default 50 chars ~10 words
            excluded_client_names: Client names to completely exclude (e.g., exclude terminalbench when running swebench)
            minimal: Minimal mode - only keep name, description, parameter names
            
        Returns:
            List of tool schemas in OpenAI-compatible format
        """
        # Get required tools
        required_tools = set(self.get_tools_by_clients(required_client_names))
        
        # Get distraction tools (tool level, also excludes excluded clients)
        distraction_tools = set(self.select_distraction_tools(
            required_client_names, distraction_count, seed, excluded_client_names
        ))
        
        # Merge
        active_tools = required_tools | distraction_tools
        
        logger.info(f"Active tools: {len(required_tools)} required + {len(distraction_tools)} distraction = {len(active_tools)} total")
        
        # Generate schema (filter out internal tools)
        schema = []
        for tool_key in active_tools:
            # Skip internal tools (not exposed to LLM)
            if tool_key in self._internal_tools:
                continue
            if tool_key not in self.tool_to_client:
                continue
            client_name = self.tool_to_client[tool_key]
            client = self.clients[client_name]
            for tool in client.tools:
                # Comparison needs to use tool_key format
                expected_key = self._get_tool_key(client_name, tool.name)
                if expected_key == tool_key:
                    description = tool.description or ""
                    parameters = convert_schema_for_openai(tool.inputSchema or {"type": "object", "properties": {}})
                    
                    if minimal:
                        # Minimal mode: only keep name, truncated description, parameter names
                        description = self._compress_description(description, max_description_len)
                        parameters = self._compress_parameters_minimal(parameters)
                    elif compress:
                        description = self._compress_description(description, max_description_len)
                        parameters = self._compress_parameters(parameters)
                    
                    schema.append({
                        "type": "function",
                        "function": {
                            "name": tool_key,  # Use tool_key as tool name (e.g., search__web_search)
                            "description": description,
                            "parameters": parameters
                        }
                    })
                    break
        return schema
    
    def get_tools_schema(
        self, 
        compress: bool = False, 
        max_description_len: int = 50,
        excluded_client_names: list[str] | None = None,
        minimal: bool = False,
    ) -> list[dict]:
        """
        Get OpenAI-compatible format schema for all tools
        (only returns non-filtered tools, excluding internal tools)
        
        Tool name format (Bedrock-compatible):
        - tau2: tool_name (e.g., telecom_get_customer)
        - mcp-bench: Server__tool (e.g., BioMCP__think, OpenAPI_Explorer__getApiOverview)
        
        Note: tool_key is already in Bedrock-compatible format [a-zA-Z0-9_-]+
        
        Args:
            compress: Whether to compress tool definitions (for long context scenarios like Claude)
            max_description_len: Max description length (chars) when compressing, default 50 chars ~10 words
            excluded_client_names: Client names to completely exclude (e.g., exclude terminalbench when running swebench)
            minimal: Minimal mode - only keep name, description, parameter names
        """
        # Get tools to exclude
        excluded_tools = set()
        if excluded_client_names:
            excluded_tools = set(self.get_tools_by_clients(excluded_client_names))
            logger.info(f"Excluding {len(excluded_tools)} tools from clients: {excluded_client_names}")
        
        schema = []
        for tool_key, client_name in self.tool_to_client.items():
            # Skip internal tools (not exposed to LLM)
            if tool_key in self._internal_tools:
                continue
            
            # Skip excluded tools
            if tool_key in excluded_tools:
                continue
                
            client = self.clients[client_name]
            
            # Extract original tool_name from tool_key (for finding the tool object)
            if self.MCPBENCH_SEPARATOR in tool_key:
                # mcp-bench format: Server__tool -> tool
                original_tool_name = tool_key.split(self.MCPBENCH_SEPARATOR, 1)[1]
                # Need to reverse-lookup original tool name (since tool_name may have been sanitized)
                # Iterate through tools to find one that matches after sanitization
                found_tool = None
                for tool in client.tools:
                    if self._sanitize_name(tool.name) == original_tool_name:
                        found_tool = tool
                        break
                if found_tool:
                    description = found_tool.description or ""
                    parameters = convert_schema_for_openai(found_tool.inputSchema or {"type": "object", "properties": {}})
                    
                    if minimal:
                        # Minimal mode: only keep name, truncated description, parameter names
                        description = self._compress_description(description, max_description_len)
                        parameters = self._compress_parameters_minimal(parameters)
                    elif compress:
                        description = self._compress_description(description, max_description_len)
                        parameters = self._compress_parameters(parameters)
                    
                    schema.append({
                        "type": "function",
                        "function": {
                            "name": tool_key,  # Bedrock-compatible tool_key
                            "description": description,
                            "parameters": parameters
                        }
                    })
            else:
                # tau2 format or other: tool_name is directly the tool_key
                for tool in client.tools:
                    if tool.name == tool_key:
                        description = tool.description or ""
                        parameters = convert_schema_for_openai(tool.inputSchema or {"type": "object", "properties": {}})
                        
                        if minimal:
                            # Minimal mode: only keep name, truncated description, parameter names
                            description = self._compress_description(description, max_description_len)
                            parameters = self._compress_parameters_minimal(parameters)
                        elif compress:
                            description = self._compress_description(description, max_description_len)
                            parameters = self._compress_parameters(parameters)
                        
                        schema.append({
                            "type": "function",
                            "function": {
                                "name": tool_key,
                                "description": description,
                                "parameters": parameters
                            }
                        })
                        break
        return schema
    
    def _compress_description(self, description: str, max_len: int = 50) -> str:
        """
        Compress tool description, preserving core information
        
        Strategy:
        1. Take the first sentence (usually the core functionality description)
        2. If still too long, truncate and add ellipsis
        
        Default max_len=50 chars (~10 words), aggressive compression for large tool set scenarios
        """
        if not description or len(description) <= max_len:
            return description
        
        # Take the first sentence
        first_sentence_end = description.find('. ')
        if first_sentence_end > 0 and first_sentence_end <= max_len:
            return description[:first_sentence_end + 1]
        
        # Truncate
        return description[:max_len - 3].rstrip() + "..."
    
    def _compress_parameters(self, parameters: dict, max_prop_desc_len: int = 50) -> dict:
        """
        Compress parameter schema, removing redundant information
        
        Strategy:
        1. Remove default values (can be handled in implementation)
        2. Truncate property descriptions (default 50 chars)
        3. Keep type, required, enum, items (required for array types)
        """
        if not parameters or not isinstance(parameters, dict):
            return parameters
        
        compressed = {"type": parameters.get("type", "object")}
        
        if "properties" in parameters:
            compressed["properties"] = {}
            for prop_name, prop_schema in parameters.get("properties", {}).items():
                if isinstance(prop_schema, dict):
                    # Keep type and truncated description
                    new_prop = {"type": prop_schema.get("type", "string")}
                    if "description" in prop_schema:
                        desc = prop_schema["description"]
                        # Truncate parameter description
                        if len(desc) > max_prop_desc_len:
                            desc = desc[:max_prop_desc_len - 3].rstrip() + "..."
                        new_prop["description"] = desc
                    # Keep enum (important constraint)
                    if "enum" in prop_schema:
                        new_prop["enum"] = prop_schema["enum"]
                    # Keep items (array type must have items, required by GPT-5)
                    if "items" in prop_schema:
                        new_prop["items"] = prop_schema["items"]
                    compressed["properties"][prop_name] = new_prop
                else:
                    compressed["properties"][prop_name] = prop_schema
        
        if "required" in parameters:
            compressed["required"] = parameters["required"]
        
        return compressed
    
    def _compress_parameters_minimal(self, parameters: dict) -> dict:
        """
        Minimal compression of parameter schema - only keep parameter name list
        
        Output format: only a required list (containing all parameter names)
        This is the most aggressive compression, saving the most tokens
        """
        if not parameters or not isinstance(parameters, dict):
            return {"type": "object", "properties": {}}
        
        # Collect all parameter names
        all_params = list(parameters.get("properties", {}).keys())
        
        # Return minimal format
        return {
            "type": "object",
            "properties": {name: {"type": "string"} for name in all_params},
            "required": parameters.get("required", all_params)
        }
    
    def get_tools_text(
        self,
        max_description_len: int = 100,
        excluded_client_names: list[str] | None = None,
    ) -> str:
        """
        Get plain text format tool descriptions (minimal mode, for bump sort and other evaluation scenarios)
        
        Format:
        - tool_name(param1, param2, ...): description
        
        More compact than JSON format, saving more tokens
        
        Args:
            max_description_len: Max description length
            excluded_client_names: Client names to exclude
        
        Returns:
            Plain text format tool list
        """
        # Get tools to exclude
        excluded_tools = set()
        if excluded_client_names:
            excluded_tools = set(self.get_tools_by_clients(excluded_client_names))
        
        lines = []
        for tool_key, client_name in self.tool_to_client.items():
            # Skip internal tools
            if tool_key in self._internal_tools:
                continue
            # Skip excluded tools
            if tool_key in excluded_tools:
                continue
            
            client = self.clients[client_name]
            
            # Find tool
            tool = None
            if self.MCPBENCH_SEPARATOR in tool_key:
                original_tool_name = tool_key.split(self.MCPBENCH_SEPARATOR, 1)[1]
                for t in client.tools:
                    if self._sanitize_name(t.name) == original_tool_name:
                        tool = t
                        break
            else:
                for t in client.tools:
                    if t.name == tool_key:
                        tool = t
                        break
            
            if tool:
                # Get parameter name list
                params = []
                if tool.inputSchema and "properties" in tool.inputSchema:
                    params = list(tool.inputSchema["properties"].keys())
                
                # Compress description
                desc = self._compress_description(tool.description or "", max_description_len)
                
                # Format: tool_name(p1, p2): desc
                params_str = ", ".join(params) if params else ""
                lines.append(f"- {tool_key}({params_str}): {desc}")
        
        return "\n".join(lines)
    
    def get_filtered_tools_text(
        self,
        required_client_names: list[str],
        distraction_count: int = 10,
        seed: int | None = None,
        max_description_len: int = 100,
        excluded_client_names: list[str] | None = None,
    ) -> str:
        """
        Get filtered plain text format tool descriptions
        
        Args:
            required_client_names: Required client names
            distraction_count: Number of distraction tools
            seed: Random seed
            max_description_len: Max description length
            excluded_client_names: Client names to exclude
            
        Returns:
            Plain text format tool list
        """
        # Get required tools
        required_tools = set(self.get_tools_by_clients(required_client_names))
        
        # Get distraction tools
        distraction_tools = set(self.select_distraction_tools(
            required_client_names, distraction_count, seed, excluded_client_names
        ))
        
        # Merge
        active_tools = required_tools | distraction_tools
        
        lines = []
        for tool_key in active_tools:
            # Skip internal tools
            if tool_key in self._internal_tools:
                continue
            if tool_key not in self.tool_to_client:
                continue
            
            client_name = self.tool_to_client[tool_key]
            client = self.clients[client_name]
            
            # Find tool
            tool = None
            for t in client.tools:
                expected_key = self._get_tool_key(client_name, t.name)
                if expected_key == tool_key:
                    tool = t
                    break
            
            if tool:
                # Get parameter name list
                params = []
                if tool.inputSchema and "properties" in tool.inputSchema:
                    params = list(tool.inputSchema["properties"].keys())
                
                # Compress description
                desc = self._compress_description(tool.description or "", max_description_len)
                
                # Format: tool_name(p1, p2): desc
                params_str = ", ".join(params) if params else ""
                lines.append(f"- {tool_key}({params_str}): {desc}")
        
        return "\n".join(lines)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> Any:
        """
        Call a tool (convenience method, returns only the result)
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool parameters
            
        Returns:
            Call result
        """
        result, _ = await self.route_tool_call(tool_name, arguments)
        return result

    async def route_tool_call(
        self, 
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> tuple[Any, str]:
        """
        Route a tool call to the correct Client (and then to the corresponding Server)
        
        Per MCP specification:
        - Host receives "User- or model-initiated action"
        - Host sends "Request (tools/resources)" to Server via Client
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool parameters
            
        Returns: 
            tuple: (call result, activated client name)
        """
        if tool_name not in self.tool_to_client:
            # Consistent with tau2 toolkit.py error format
            raise ValueError(f"Tool '{tool_name}' not found.")
        
        client_name = self.tool_to_client[tool_name]
        client = self.clients[client_name]
        
        logger.debug(f"Routing tool '{tool_name}' to client '{client_name}'")
        
        # If in Server__tool format (mcp-bench), extract actual tool name to pass to MCP server
        # Note: Need to distinguish internal tools (starting with __, e.g., __telecom_execute_user_tool)
        # from mcp-bench format (Server__tool, e.g., BioMCP__think)
        # Internal tools start with __, after split the first part is an empty string
        if self.MCPBENCH_SEPARATOR in tool_name and not tool_name.startswith(self.INTERNAL_TOOL_PREFIX):
            # tool_name is in sanitized format, need to find the original tool name
            sanitized_tool_name = tool_name.split(self.MCPBENCH_SEPARATOR, 1)[1]
            # Iterate through tools to find the original tool name that matches after sanitization
            actual_tool_name = None
            for tool in client.tools:
                if self._sanitize_name(tool.name) == sanitized_tool_name:
                    actual_tool_name = tool.name  # Use original tool name
                    break
            if actual_tool_name is None:
                raise ValueError(f"Tool '{tool_name}' not found in client '{client_name}'")
        else:
            actual_tool_name = tool_name
        
        if client.transport_type == "openhands":
            # OpenHands Runtime transport - execute directly in-process
            result = await self._call_tool_openhands(client, actual_tool_name, arguments)
        elif client.transport_type == "http":
            # HTTP transport - call via JSON-RPC
            result = await self._call_tool_http(client, actual_tool_name, arguments)
        else:
            # STDIO transport - call via ClientSession
            result = await client.session.call_tool(actual_tool_name, arguments)
        
        return result, client_name
    
    # NOTE: _call_tool_openhands method removed - SWE-Bench now handled via Terminal-Bench

    async def _call_tool_http(
        self,
        client: MCPClient,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> Any:
        """Call a tool via HTTP transport"""
        base_url = f"http://localhost:{client.port}{client.endpoint}"
        
        tool_request = {
            "jsonrpc": "2.0",
            "id": random.randint(1, 10000),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'
        }
        if client.session_id:
            headers['mcp-session-id'] = client.session_id
        
        async with self._http_session.post(
            base_url,
            json=tool_request,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Tool call failed: {error_text}")
            
            content_type = response.headers.get('content-type', '')
            if 'text/event-stream' in content_type:
                response_text = await response.text()
                lines = response_text.strip().split('\n')
                for line in lines:
                    if line.startswith('data: '):
                        try:
                            result = json.loads(line[6:])
                            break
                        except json.JSONDecodeError:
                            continue
            else:
                result = await response.json()
            
            if 'error' in result:
                raise Exception(f"Tool error: {result['error']}")
            
            return result.get('result', result)
    
    async def shutdown(self) -> None:
        """
        Terminate all Client connections
        
        Per MCP specification: Host -> Client: "Terminate", Client -> Server: "End session"
        """
        logger.info("Shutting down BenchmarkHost...")
        
        # Close HTTP server processes
        for name, client in self.clients.items():
            if client.transport_type == "http" and client.server_process:
                logger.info(f"Stopping HTTP server for {name} (PID: {client.server_process.pid})")
                try:
                    client.server_process.terminate()
                    await asyncio.sleep(1)
                    if client.server_process.poll() is None:
                        client.server_process.kill()
                except Exception as e:
                    logger.error(f"Error stopping HTTP server {name}: {e}")
            
            # NOTE: OpenHands Runtime cleanup removed - SWE-Bench now handled via Terminal-Bench
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        
        # Close STDIO connections
        await self._exit_stack.aclose()
        
        self.clients.clear()
        self.tool_to_client.clear()
        self._initialized = False
        logger.info("BenchmarkHost shutdown complete")