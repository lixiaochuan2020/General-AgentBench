"""
Tau2-Bench Server

Directly expose tau2-bench native tools to MCP Host
"""

import json
import sys
import inspect
import logging
from typing import Any, Optional, get_type_hints
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import Field, create_model

# Set up logger
logger = logging.getLogger(__name__)

# Add tau2-bench to Python path
TAU2_BENCH_PATH = Path(__file__).parent.parent.parent.parent / "benchmarks" / "tau2-bench"
if TAU2_BENCH_PATH.exists():
    sys.path.insert(0, str(TAU2_BENCH_PATH / "src"))


class Tau2BenchServer:
    """
    Tau2-Bench Benchmark Server
    
    Directly expose tau2-bench native tools for each domain
    """
    
    def __init__(self, domain: str = "airline", add_domain_prefix: bool = True):
        """
        Initialize Tau2-Bench Server
        
        Args:
            domain: Domain to load (airline, retail, telecom, mock)
            add_domain_prefix: Whether to add domain prefix to tool names (to avoid name conflicts across domains)
        """
        self.name = f"tau2-{domain}"
        self.domain = domain
        self.add_domain_prefix = add_domain_prefix
        self.mcp = FastMCP(self.name)
        self.environment = None
        self.toolkit = None
        self.user_toolkit = None  # User tools (e.g. telecom domain)
        
        # Initialize environment and tools
        self._init_environment()
        self._register_native_tools()
    
    def _init_environment(self) -> None:
        """Initialize tau2 environment"""
        try:
            from tau2.registry import registry
            
            # Get environment constructor
            env_constructor = registry.get_env_constructor(self.domain)
            self.environment = env_constructor()
            self.toolkit = self.environment.tools
            self.user_toolkit = self.environment.user_tools  # Load user tools
            
        except ImportError as e:
            print(f"Warning: tau2-bench not available: {e}")
            print("Running in mock mode with limited functionality")
            self.environment = None
            self.toolkit = None
            self.user_toolkit = None
        except Exception as e:
            print(f"Error initializing tau2 environment: {e}")
            self.environment = None
            self.toolkit = None
            self.user_toolkit = None
    
    def apply_initial_state(self, initial_state) -> None:
        """
        Apply the task's initial state to the environment
        
        This is critical for correct evaluation: the evaluator creates its own environment and applies initial_state,
        so our MCP server environment must also apply the same initial_state to stay consistent.
        
        Args:
            initial_state: tau2 Task.initial_state object
        """
        if self.environment is None:
            return
            
        if initial_state is None:
            return
        
        initialization_data = None
        if initial_state.initialization_data is not None:
            initialization_data = initial_state.initialization_data
        
        initialization_actions = None
        if initial_state.initialization_actions is not None:
            initialization_actions = initial_state.initialization_actions
        
        # Apply initial state
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=[],
        )
    
    def reset_environment(self) -> None:
        """Reset environment to initial state (recreate environment)"""
        self._init_environment()
    
    def _register_native_tools(self) -> None:
        """Register tau2-bench native tools (agent tools only, excluding user tools)
        
        Note: User tools are for the User Simulator and should not be exposed to the Agent.
        User Simulator calls user_toolkit directly via the tau2 native interface.
        """
        
        if self.toolkit is None:
            # If tau2-bench is not available, register some basic tools
            self._register_fallback_tools()
            return
        
        # Register internal management tools (for setting task initial state, not exposed to Agent)
        self._register_internal_tools()
        
        # Get agent native tools only (excluding user tools)
        native_tools = self.toolkit.get_tools()
        
        for tool_name, tool in native_tools.items():
            # Optionally add domain prefix
            mcp_tool_name = f"{self.domain}_{tool_name}" if self.add_domain_prefix else tool_name
            self._register_single_tool(mcp_tool_name, tool_name, tool, tool_type="agent")
        
        # Note: No longer registering user tools to Agent
        # User tools (e.g. telecom's get_status_bar, run_speed_test, etc.) are used directly by User Simulator
    
    def _register_internal_tools(self) -> None:
        """Register internal management tools (for setting task initial state, etc.)"""
        
        server_ref = self  # Closure reference
        
        @self.mcp.tool(name=f"__{self.domain}_apply_initial_state")
        def apply_initial_state(initialization_actions_json: str) -> str:
            """
            [Internal] Apply initial state to the environment.
            This is called before each task to set up the correct initial state.
            
            Args:
                initialization_actions_json: JSON string of initialization actions
            """
            import json
            from tau2.data_model.tasks import EnvFunctionCall
            try:
                actions_data = json.loads(initialization_actions_json)
                # Convert to EnvFunctionCall object list
                # EnvFunctionCall uses func_name instead of name, env_type is Literal["user", "assistant"]
                actions = []
                for action_dict in actions_data:
                    env_type_str = action_dict.get("env_type", "assistant")
                    # env_type uses string "user" or "assistant" directly
                    actions.append(EnvFunctionCall(
                        func_name=action_dict["name"],
                        arguments=action_dict.get("arguments", {}),
                        env_type=env_type_str,
                    ))
                
                if server_ref.environment is not None:
                    logger.info(f"[DEBUG apply_initial_state] Before set_state: toolkit id={id(server_ref.toolkit)}, env.tools id={id(server_ref.environment.tools)}")
                    logger.info(f"[DEBUG apply_initial_state] toolkit is env.tools: {server_ref.toolkit is server_ref.environment.tools}")
                    server_ref.environment.set_state(
                        initialization_data=None,
                        initialization_actions=actions,
                        message_history=[],
                    )
                    logger.info(f"[DEBUG apply_initial_state] After set_state: toolkit id={id(server_ref.toolkit)}, env.tools id={id(server_ref.environment.tools)}")
                return json.dumps({"status": "success", "actions_applied": len(actions)})
            except Exception as e:
                import traceback
                return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
        
        @self.mcp.tool(name=f"__{self.domain}_reset_environment")
        def reset_environment() -> str:
            """
            [Internal] Reset the environment to default state.
            """
            import json
            try:
                old_toolkit_id = id(server_ref.toolkit) if server_ref.toolkit else None
                server_ref.reset_environment()
                new_toolkit_id = id(server_ref.toolkit) if server_ref.toolkit else None
                logger.info(f"[DEBUG reset_environment] toolkit changed: {old_toolkit_id} -> {new_toolkit_id}")
                logger.info(f"[DEBUG reset_environment] toolkit is environment.tools: {server_ref.toolkit is server_ref.environment.tools if server_ref.environment else 'N/A'}")
                return json.dumps({"status": "success"})
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})
        
        @self.mcp.tool(name=f"__{self.domain}_execute_user_tool")
        def execute_user_tool(tool_name: str, arguments_json: str) -> str:
            """
            [Internal] Execute a user tool on the environment.
            This is called by the User Simulator to execute user actions
            that affect the environment state.
            
            Args:
                tool_name: Name of the user tool to execute
                arguments_json: JSON string of tool arguments
            """
            import json
            try:
                if server_ref.user_toolkit is None:
                    return f"Error: No user toolkit available for domain {server_ref.domain}"
                
                arguments = json.loads(arguments_json) if arguments_json else {}
                result = server_ref.user_toolkit.use_tool(tool_name, **arguments)
                
                # Sync environment state
                if server_ref.environment is not None:
                    server_ref.environment.sync_tools()
                
                # Return format must match tau2 native: return result string directly
                return str(result) if result is not None else ""
            except Exception as e:
                return f"Error: {str(e)}"
        
        @self.mcp.tool(name=f"__{self.domain}_replay_message_history")
        def replay_message_history(messages_json: str) -> str:
            """
            [Internal] Replay message history to restore environment state from checkpoint.
            
            This is used by Sequential Scaling to restore the environment state
            when resuming from a checkpoint. It parses the message history and
            replays all tool calls to bring the environment to the same state.
            
            Args:
                messages_json: JSON string of the message history from checkpoint
                
            Returns:
                JSON status with replayed message count and db_hash (if available)
            """
            import json
            try:
                messages_data = json.loads(messages_json)
                
                if server_ref.environment is None:
                    return json.dumps({
                        "status": "error",
                        "message": "Environment not initialized"
                    })
                
                # Convert Omni message format to tau2 Message format
                from tau2.data_model.message import (
                    AssistantMessage, ToolMessage, ToolCall, UserMessage
                )
                
                tau2_messages = []
                for msg in messages_data:
                    role = msg.get("role", "")
                    
                    if role == "assistant":
                        tool_calls = []
                        raw_tool_calls = msg.get("tool_calls", [])
                        if raw_tool_calls:
                            for tc in raw_tool_calls:
                                # Omni format: {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
                                # Or simplified format: {"id": ..., "name": ..., "arguments": ...}
                                if "function" in tc:
                                    func = tc["function"]
                                    tc_name = func.get("name", "")
                                    tc_args = func.get("arguments", {})
                                    if isinstance(tc_args, str):
                                        tc_args = json.loads(tc_args)
                                else:
                                    tc_name = tc.get("name", "")
                                    tc_args = tc.get("arguments", {})
                                    if isinstance(tc_args, str):
                                        tc_args = json.loads(tc_args)
                                
                                tool_calls.append(ToolCall(
                                    id=tc.get("id", ""),
                                    name=tc_name,
                                    arguments=tc_args,
                                    requestor="assistant",
                                ))
                        
                        tau2_messages.append(AssistantMessage(
                            role="assistant",
                            content=msg.get("content") or "",
                            tool_calls=tool_calls if tool_calls else None,
                        ))
                    
                    elif role == "tool":
                        tau2_messages.append(ToolMessage(
                            role="tool",
                            id=msg.get("tool_call_id", ""),
                            content=msg.get("content", ""),
                        ))
                    
                    elif role == "user":
                        # User messages (including User Simulator's tool_calls)
                        user_tool_calls = []
                        raw_tool_calls = msg.get("tool_calls", [])
                        if raw_tool_calls:
                            for tc in raw_tool_calls:
                                if "function" in tc:
                                    func = tc["function"]
                                    tc_name = func.get("name", "")
                                    tc_args = func.get("arguments", {})
                                    if isinstance(tc_args, str):
                                        tc_args = json.loads(tc_args)
                                else:
                                    tc_name = tc.get("name", "")
                                    tc_args = tc.get("arguments", {})
                                    if isinstance(tc_args, str):
                                        tc_args = json.loads(tc_args)
                                
                                user_tool_calls.append(ToolCall(
                                    id=tc.get("id", ""),
                                    name=tc_name,
                                    arguments=tc_args,
                                    requestor="user",
                                ))
                        
                        tau2_messages.append(UserMessage(
                            role="user",
                            content=msg.get("content") or "",
                            tool_calls=user_tool_calls if user_tool_calls else None,
                        ))
                    
                    # system messages are skipped (do not affect environment state)
                
                # Call environment.set_state() to replay message history
                server_ref.environment.set_state(
                    initialization_data=None,
                    initialization_actions=None,  # Initialization actions should have been executed earlier
                    message_history=tau2_messages,
                )
                
                # Get db_hash (if available) for verifying state consistency
                db_hash = None
                if hasattr(server_ref.environment, 'get_db_hash'):
                    db_hash = server_ref.environment.get_db_hash()
                
                return json.dumps({
                    "status": "success",
                    "replayed_messages": len(tau2_messages),
                    "db_hash": db_hash,
                })
                
            except Exception as e:
                import traceback
                return json.dumps({
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                })
    
    def _register_single_tool(self, mcp_tool_name: str, original_name: str, tool, tool_type: str = "agent") -> None:
        """
        Register a single tau2 Tool as an MCP tool
        
        Args:
            mcp_tool_name: Tool name displayed in MCP
            original_name: Original tool name in tau2-bench
            tool: tau2 Tool object
            tool_type: Tool type ("agent" or "user")
        """
        # Get tool schema
        schema = tool.openai_schema
        func_schema = schema.get("function", {})
        description = func_schema.get("description", mcp_tool_name)
        
        # Add description prefix for user tools
        if tool_type == "user":
            description = f"[User Tool] {description}"
        
        parameters = func_schema.get("parameters", {})
        param_props = parameters.get("properties", {})
        required_params = parameters.get("required", [])
        
        # Choose correct toolkit getter based on tool type
        # Important: Cannot directly capture toolkit reference because reset_environment rebuilds the toolkit object
        # Use closure referencing self so the latest toolkit is dynamically obtained on each call
        server_ref_for_tools = self
        is_user_tool = (tool_type == "user")
        
        def get_current_toolkit():
            """Dynamically get the current toolkit (agent or user)"""
            if is_user_tool:
                return server_ref_for_tools.user_toolkit
            else:
                return server_ref_for_tools.toolkit
        
        # Map JSON Schema types to Python types
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        # Build function parameter list (using closures to store tool info)
        orig_name = original_name
        t_name = mcp_tool_name
        
        # Serialization function fully consistent with tau2 Environment.to_json_str
        # Note: tau2's to_json_str has a bug - BaseModel is not processed recursively, causing datetime to be serialized with str()
        # We must replicate this buggy behavior to ensure evaluator comparison passes
        def _process_for_json(resp):
            """Serialization fully consistent with tau2 Environment.to_json_str._process
            
            Note: This replicates tau2's buggy behavior:
            1. BaseModel.model_dump() results are not recursively processed
            2. Only top-level int/float/bool are converted to str, nested ones are not
            3. datetime relies on json.dumps(default=str), producing space-separated format
            """
            from pydantic import BaseModel
            from datetime import datetime, date
            
            if isinstance(resp, BaseModel):
                # tau2 bug: directly returns model_dump() without recursive processing
                return resp.model_dump()
            elif isinstance(resp, str):
                return resp
            elif resp is None:
                return resp
            elif isinstance(resp, (int, float, bool)):
                return str(resp)
            elif isinstance(resp, list):
                return [_process_for_json(item) for item in resp]
            elif isinstance(resp, tuple):
                return tuple(_process_for_json(item) for item in resp)
            elif isinstance(resp, dict):
                return {k: _process_for_json(v) for k, v in resp.items()}
            elif isinstance(resp, (datetime, date)):
                return resp.isoformat()
            else:
                raise ValueError(f"Unsupported type: {type(resp)}")
        
        # Create an executor function
        def make_executor(tool_name, get_toolkit, param_properties):
            def executor(**kwargs):
                try:
                    # Dynamically get current toolkit (will be a new object after reset_environment)
                    toolkit = get_toolkit()
                    logger.info(f"[DEBUG executor] tool={tool_name}, toolkit id={id(toolkit)}")
                    
                    # Critical: Perform type conversion based on schema
                    # LLM may pass parameters with wrong types, we need to correct based on schema
                    converted_kwargs = {}
                    for param_name, param_value in kwargs.items():
                        if param_name in param_properties:
                            expected_type = param_properties[param_name].get("type", "string")
                            original_value = param_value
                            if expected_type == "number" and isinstance(param_value, int):
                                # Force convert int -> float (e.g. gb_amount: 2 -> 2.0)
                                param_value = float(param_value)
                                logger.info(f"[DEBUG executor] Converted {param_name}: {original_value} (int) -> {param_value} (float)")
                            elif expected_type == "integer" and isinstance(param_value, float):
                                # Force convert float -> int
                                param_value = int(param_value)
                            elif expected_type == "string" and not isinstance(param_value, str):
                                # Force convert other types -> str (e.g. product_id: 123 -> "123")
                                param_value = str(param_value)
                        converted_kwargs[param_name] = param_value
                    
                    result = toolkit.use_tool(tool_name, **converted_kwargs)
                    # Convert result to JSON string, fully consistent with tau2 Environment.to_json_str
                    # Critical: If result is a string, return directly without JSON wrapping
                    if isinstance(result, str):
                        return result
                    else:
                        # Use serialization logic consistent with tau2
                        processed = _process_for_json(result)
                        return json.dumps(processed, default=str)
                except Exception as e:
                    # Consistent with native tau2 error format: "Error: <message>"
                    raise ValueError(f"Error: {str(e)}")
            return executor
        
        executor = make_executor(orig_name, get_current_toolkit, param_props)
        
        # Dynamically create a function with correct signature
        # Build parameter signature
        sig_params = []
        for param_name, param_info in param_props.items():
            param_type = type_mapping.get(param_info.get("type", "string"), str)
            param_desc = param_info.get("description", "")
            
            # Create parameter
            if param_name in required_params:
                # Required parameter
                param = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=param_type
                )
            else:
                # Optional parameter
                param = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=Optional[param_type]
                )
            sig_params.append(param)
        
        # Create new function signature
        new_sig = inspect.Signature(sig_params, return_annotation=str)
        
        # Create wrapper function
        def make_wrapper(exec_fn, signature, name, doc):
            def wrapper(*args, **kwargs):
                # Convert positional arguments to keyword arguments
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                # Filter out None-valued optional parameters
                filtered_kwargs = {k: v for k, v in bound.arguments.items() if v is not None}
                return exec_fn(**filtered_kwargs)
            
            wrapper.__signature__ = signature
            wrapper.__name__ = name
            wrapper.__doc__ = doc
            wrapper.__annotations__ = {p.name: p.annotation for p in signature.parameters.values()}
            wrapper.__annotations__['return'] = str
            return wrapper
        
        handler = make_wrapper(executor, new_sig, mcp_tool_name, description)
        
        # Register using MCP's tool decorator
        self.mcp.tool(name=mcp_tool_name, description=description)(handler)
    
    def _register_fallback_tools(self) -> None:
        """Register fallback tools (when tau2-bench is not available)"""
        
        @self.mcp.tool()
        def get_server_status() -> str:
            """Get server status"""
            return json.dumps({
                "status": "running",
                "domain": self.domain,
                "tau2_available": False,
                "message": "tau2-bench not installed. Please run: cd ../tau2-bench && pip install -e ."
            }, indent=2)
        
        @self.mcp.tool()
        def list_available_domains() -> str:
            """List all available tau2 domains"""
            return json.dumps({
                "domains": ["airline", "retail", "telecom", "mock"],
                "current": self.domain,
                "note": "tau2-bench not installed, tools not available"
            }, indent=2)


def create_server(domain: str = "airline") -> Tau2BenchServer:
    """Create a tau2-bench server for the specified domain"""
    return Tau2BenchServer(domain=domain)


# Create independent entry points for each domain
def create_airline_server():
    return Tau2BenchServer(domain="airline")

def create_retail_server():
    return Tau2BenchServer(domain="retail")

def create_telecom_server():
    return Tau2BenchServer(domain="telecom")

def create_mock_server():
    return Tau2BenchServer(domain="mock")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run Tau2-Bench MCP Server")
    parser.add_argument(
        "--domain", 
        type=str, 
        default=os.environ.get("TAU2_DOMAIN", "airline"),
        choices=["airline", "retail", "telecom", "mock"],
        help="Domain to load (default: airline, or TAU2_DOMAIN env var)"
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Don't add domain prefix to tool names"
    )
    args = parser.parse_args()
    
    add_prefix = not args.no_prefix
    print(f"Starting Tau2-Bench Server for domain: {args.domain}")
    print(f"Add domain prefix: {add_prefix}")
    server = Tau2BenchServer(domain=args.domain, add_domain_prefix=add_prefix)
    
    total_tools = 0
    if server.toolkit:
        agent_tools = server.toolkit.get_tools()
        print(f"\nLoaded {len(agent_tools)} agent tools (available to Agent):")
        for name in agent_tools.keys():
            mcp_name = f"{args.domain}_{name}" if add_prefix else name
            print(f"  - {mcp_name} (original: {name})")
        total_tools += len(agent_tools)
    
    if server.user_toolkit:
        user_tools = server.user_toolkit.get_tools()
        print(f"\n[INFO] {len(user_tools)} user tools detected (for User Simulator only, NOT exposed to Agent):")
        for name in list(user_tools.keys())[:5]:  # Only show first 5
            print(f"  - {name}")
        if len(user_tools) > 5:
            print(f"  ... and {len(user_tools) - 5} more")
    
    if total_tools == 0:
        print("Warning: Running in fallback mode (tau2-bench not available)")
    else:
        print(f"\nTotal: {total_tools} tools registered as MCP tools")
    
    server.mcp.run()
