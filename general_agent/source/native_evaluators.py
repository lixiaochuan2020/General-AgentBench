"""
Native Evaluators - Native evaluators for each benchmark

Calls native evaluation logic for each benchmark to ensure comparable results.
"""

import json
import copy
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class NativeEvalResult:
    """Native evaluation result"""
    task_id: str
    domain: str
    benchmark: str  # "tau2", "webarena", etc.
    reward: float  # 0.0 - 1.0
    success: bool
    info: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def strip_domain_prefix(tool_name: str, domain: str) -> str:
    """
    Remove domain prefix from tool name
    
    Args:
        tool_name: MCP tool name, e.g. "airline_get_user_details"
        domain: Domain name, e.g. "airline"
    
    Returns:
        Original tool name, e.g. "get_user_details"
    """
    prefix = f"{domain}_"
    user_prefix = f"{domain}_user_"
    
    if tool_name.startswith(user_prefix):
        return tool_name[len(user_prefix):]
    elif tool_name.startswith(prefix):
        return tool_name[len(prefix):]
    return tool_name


def strip_domain_prefix_in_content(content: str, domain: str) -> str:
    """
    Remove tool name domain prefix from ToolMessage content
    
    This is used to handle error messages etc., ensuring tool names in content
    are consistent with the converted format.
    For example:
    - "Error: Tool 'telecom_set_network_mode_preference' not found."
    - Becomes "Error: Tool 'set_network_mode_preference' not found."
    
    Args:
        content: ToolMessage content
        domain: Domain name, e.g. "telecom"
    
    Returns:
        Content with replacements applied
    """
    import re
    
    # Match '{domain}_xxx' or "{domain}_xxx" pattern tool names
    # Also handle {domain}_user_ prefix
    user_prefix = f"{domain}_user_"
    prefix = f"{domain}_"
    
    # Handle user_ prefix first (longer pattern takes priority)
    content = content.replace(f"'{user_prefix}", "'")
    content = content.replace(f'"{user_prefix}', '"')
    
    # Then handle normal prefix
    content = content.replace(f"'{prefix}", "'")
    content = content.replace(f'"{prefix}', '"')
    
    return content


class Tau2Evaluator:
    """
    tau2-bench native evaluator wrapper
    
    Uses tau2.evaluator.evaluator.evaluate_simulation for evaluation
    """
    
    def __init__(self, solo_mode: bool = False):
        self.solo_mode = solo_mode
    
    def convert_trace_to_simulation(self, trace, task, domain: str):
        """
        Convert AgentTrace to tau2 SimulationRun format
        
        Args:
            trace: AgentTrace produced by UniversalAgent
            task: tau2 Task object
            domain: Domain name
            
        Returns:
            SimulationRun object
        """
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.data_model.message import (
            AssistantMessage,
            ToolCall,
            ToolMessage,
            UserMessage,
        )
        from tau2.registry import registry
        
        logger.info(f"[EVAL CONVERT] Converting trace to tau2 simulation format")
        logger.info(f"[EVAL CONVERT] Input trace messages: {len(trace.messages)}")
        
        # Get tool schema for type conversion
        tool_schemas = {}
        try:
            env_constructor = registry.get_env_constructor(domain)
            temp_env = env_constructor()
            for tool_name, tool in temp_env.tools.get_tools().items():
                schema = tool.openai_schema.get("function", {}).get("parameters", {}).get("properties", {})
                tool_schemas[tool_name] = schema
            if temp_env.user_tools:
                for tool_name, tool in temp_env.user_tools.get_tools().items():
                    schema = tool.openai_schema.get("function", {}).get("parameters", {}).get("properties", {})
                    tool_schemas[tool_name] = schema
        except Exception as e:
            logger.warning(f"[EVAL CONVERT] Failed to load tool schemas: {e}")
        
        def convert_arguments(tool_name: str, arguments: dict) -> dict:
            """Convert argument types based on tool schema"""
            if tool_name not in tool_schemas:
                return arguments
            schema = tool_schemas[tool_name]
            converted = {}
            for param_name, param_value in arguments.items():
                if param_name in schema:
                    expected_type = schema[param_name].get("type", "string")
                    if expected_type == "number" and isinstance(param_value, int):
                        param_value = float(param_value)
                    elif expected_type == "integer" and isinstance(param_value, float):
                        param_value = int(param_value)
                converted[param_name] = param_value
            return converted
        
        messages = []
        
        # Build tool_call_id -> requestor mapping table
        # So when encountering a ToolMessage, we can correctly determine if it responds to a user or assistant tool call
        tool_call_id_to_requestor: dict[str, str] = {}
        
        # First pass: collect all tool_call id and requestor mappings
        for msg in trace.messages:
            if msg.role == "user" and msg.tool_calls:
                # User-initiated tool calls
                for tc in msg.tool_calls:
                    tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                    if tc_id:
                        tool_call_id_to_requestor[tc_id] = "user"
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant-initiated tool calls
                for tc in msg.tool_calls:
                    tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                    if tc_id:
                        tool_call_id_to_requestor[tc_id] = "assistant"
        
        logger.info(f"[EVAL CONVERT] Built tool_call_id->requestor map with {len(tool_call_id_to_requestor)} entries")
        for tc_id, requestor in list(tool_call_id_to_requestor.items())[:10]:  # Only show first 10
            logger.info(f"[EVAL CONVERT]   {tc_id} -> {requestor}")
        
        def get_tc_attr(tc, attr, default=None):
            """Get tool_call attribute, compatible with dict and object, and OpenAI nested format"""
            if isinstance(tc, dict):
                # First try direct access
                if attr in tc:
                    return tc.get(attr, default)
                # Try OpenAI nested format {"function": {"name": ..., "arguments": ...}}
                if "function" in tc:
                    func = tc["function"]
                    if attr == "name":
                        return func.get("name", default)
                    elif attr == "arguments":
                        args = func.get("arguments", default)
                        # arguments may be a JSON string
                        if isinstance(args, str):
                            try:
                                import json
                                return json.loads(args)
                            except:
                                return {}
                        return args
                return default
            return getattr(tc, attr, default)
        
        for msg in trace.messages:
            if msg.role == "system":
                continue
            elif msg.role == "user":
                # User message may also contain tool_calls (from User Simulator)
                if msg.tool_calls:
                    tool_calls = [
                        ToolCall(
                            id=get_tc_attr(tc, 'id') or f"user_{get_tc_attr(tc, 'name')}",
                            name=strip_domain_prefix(get_tc_attr(tc, 'name'), domain),
                            arguments=convert_arguments(
                                strip_domain_prefix(get_tc_attr(tc, 'name'), domain),
                                get_tc_attr(tc, 'arguments', {})
                            ),
                            requestor="user",  # User-initiated tool call
                        )
                        for tc in msg.tool_calls
                    ]
                    logger.info(f"[EVAL CONVERT]   UserMessage with {len(tool_calls)} tool_calls (requestor=user)")
                    for tc in tool_calls:
                        logger.info(f"[EVAL CONVERT]     - {tc.name}({tc.arguments})")
                    messages.append(UserMessage(
                        role="user",
                        content=msg.content or "",
                        tool_calls=tool_calls,
                    ))
                else:
                    messages.append(UserMessage(
                        role="user",
                        content=msg.content or "",
                    ))
            elif msg.role == "assistant":
                if msg.tool_calls:
                    # Assistant message with tool calls
                    tool_calls = [
                        ToolCall(
                            id=get_tc_attr(tc, 'id'),
                            name=strip_domain_prefix(get_tc_attr(tc, 'name'), domain),
                            arguments=convert_arguments(
                                strip_domain_prefix(get_tc_attr(tc, 'name'), domain),
                                get_tc_attr(tc, 'arguments', {})
                            ),
                            requestor="assistant",
                        )
                        for tc in msg.tool_calls
                    ]
                    logger.info(f"[EVAL CONVERT]   AssistantMessage with {len(tool_calls)} tool_calls (requestor=assistant)")
                    for tc in tool_calls:
                        logger.info(f"[EVAL CONVERT]     - {tc.name}({tc.arguments})")
                    messages.append(AssistantMessage(
                        role="assistant",
                        content=msg.content,
                        tool_calls=tool_calls,
                    ))
                else:
                    messages.append(AssistantMessage(
                        role="assistant",
                        content=msg.content or "",
                    ))
            elif msg.role == "tool":
                # Use pre-built tool_call_id -> requestor mapping table to determine
                tool_call_id = msg.tool_call_id or ""
                # Look up requestor from mapping table, default to assistant if not found
                requestor = tool_call_id_to_requestor.get(tool_call_id, "assistant")
                # Handle tool name prefix in content, ensure consistency with ToolCall name format
                content = strip_domain_prefix_in_content(msg.content or "", domain)
                logger.info(f"[EVAL CONVERT]   ToolMessage id={tool_call_id}, requestor={requestor}")
                messages.append(ToolMessage(
                    role="tool",
                    id=tool_call_id,
                    content=content,
                    requestor=requestor,
                    error=False,
                ))
        
        # Determine termination reason
        if trace.error:
            if "max steps" in trace.error.lower():
                termination_reason = TerminationReason.MAX_STEPS
            else:
                termination_reason = TerminationReason.TOO_MANY_ERRORS
        else:
            termination_reason = TerminationReason.USER_STOP
        
        # Generate necessary time and ID information
        now = datetime.now()
        start_time = trace.start_time if hasattr(trace, 'start_time') and trace.start_time else now.isoformat()
        end_time = trace.end_time if hasattr(trace, 'end_time') and trace.end_time else now.isoformat()
        duration = trace.duration if hasattr(trace, 'duration') and trace.duration else 0.0
        run_id = str(uuid.uuid4())
        
        return SimulationRun(
            id=run_id,
            task_id=task.id,
            start_time=start_time if isinstance(start_time, str) else start_time.isoformat(),
            end_time=end_time if isinstance(end_time, str) else end_time.isoformat(),
            duration=duration,
            messages=messages,
            termination_reason=termination_reason,
        )
    
    def evaluate(
        self,
        trace,
        task,
        domain: str,
    ) -> NativeEvalResult:
        """
        Evaluate using tau2 native evaluator
        
        Args:
            trace: AgentTrace
            task: tau2 Task object
            domain: Domain name
            
        Returns:
            NativeEvalResult
        """
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        
        try:
            # Convert to tau2 format
            simulation = self.convert_trace_to_simulation(trace, task, domain)
            
            # Call native evaluator
            logger.info(f"[EVAL NATIVE] Calling tau2 evaluate_simulation...")
            logger.info(f"[EVAL NATIVE] Simulation messages: {len(simulation.messages)}")
            logger.info(f"[EVAL NATIVE] Task.id: {task.id}")
            logger.info(f"[EVAL NATIVE] Domain: {domain}")
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=self.solo_mode,
                domain=domain,
            )
            logger.info(f"########################### {reward_info}")
            logger.info(f"[EVAL NATIVE] evaluate_simulation completed")
            # Debug logging: reward_info fields
            try:
                logger.debug(
                    "[Tau2Evaluator] reward=%.2f | breakdown=%s | db_check=%s | env_assertions=%s | action_checks=%s | communicate_checks=%s",
                    getattr(reward_info, "reward", 0.0),
                    getattr(reward_info, "reward_breakdown", None),
                    getattr(reward_info, "db_check", None),
                    getattr(reward_info, "env_assertions", None),
                    getattr(reward_info, "action_checks", None),
                    getattr(reward_info, "communicate_checks", None),
                )
            except Exception as _:
                pass
            
            # Use model_dump to convert Pydantic objects to dict, ensuring correct JSON serialization
            def serialize_pydantic(obj):
                """Convert Pydantic object to JSON-serializable dict"""
                if obj is None:
                    return None
                if hasattr(obj, 'model_dump'):
                    return obj.model_dump()
                if isinstance(obj, list):
                    return [serialize_pydantic(item) for item in obj]
                if isinstance(obj, dict):
                    return {k: serialize_pydantic(v) for k, v in obj.items()}
                return obj
            
            return NativeEvalResult(
                task_id=task.id,
                domain=domain,
                benchmark="tau2",
                reward=reward_info.reward,
                success=reward_info.reward==1.0,
                info={
                    "reward_breakdown": serialize_pydantic(reward_info.reward_breakdown),
                    "db_check": serialize_pydantic(reward_info.db_check),
                    "env_assertions": serialize_pydantic(reward_info.env_assertions),
                    "action_checks": serialize_pydantic(reward_info.action_checks),
                    "communicate_checks": serialize_pydantic(reward_info.communicate_checks),
                },
            )
            
        except Exception as e:
            logger.info(f"Tau2 evaluation failed for task {task.id} with error: {e}")
            logger.exception(f"Tau2 evaluation failed for task {task.id}")
            return NativeEvalResult(
                task_id=task.id,
                domain=domain,
                benchmark="tau2",
                reward=0.0,
                success=False,
                error=str(e),
            )


class MCPBenchEvaluator:
    """
    mcp-bench native evaluator - fully consistent with original version
    
    Uses 6-dimension 1-10 scoring method:
    1. Task Fulfillment: Task completion degree
    2. Grounding: Whether information is well-grounded
    3. Tool Appropriateness: Whether tool selection is appropriate
    4. Parameter Accuracy: Parameter accuracy
    5. Dependency Awareness: Dependency awareness
    6. Parallelism and Efficiency: Parallelism and efficiency
    
    Supports Judge Stability (average of multiple evaluations) and Prompt randomization
    """
    
    def __init__(self, llm_model: str = "bedrock/openai.gpt-oss-120b-1:0", 
                 enable_judge_stability: bool = True,
                 judge_stability_runs: int = 5):
        """
        Initialize evaluator
        
        Args:
            llm_model: Model used for LLM-as-judge
            enable_judge_stability: Whether to enable Judge Stability (average of multiple evaluations)
            judge_stability_runs: Number of Judge Stability evaluation runs
        """
        self.llm_model = llm_model
        self.enable_judge_stability = enable_judge_stability
        self.judge_stability_runs = judge_stability_runs
        
        # 6-dimension scoring criteria (consistent with original version)
        self.evaluation_dimensions = {
            "Task Completion": {
                "Task Fulfillment": {
                    "1-3": "Perfectly completes 10-30% of requirements.",
                    "4-6": "Perfectly completes 40-60% of requirements.",
                    "7-8": "Perfectly completes 70-80% of requirements.",
                    "9-10": "Perfectly completes 90-100% of requirements."
                },
                "Grounding": {
                    "1-3": "10-30% of claims are perfectly grounded in tool outputs.",
                    "4-6": "40-60% of claims are perfectly grounded in tool outputs.",
                    "7-8": "70-80% of claims are perfectly grounded in tool outputs.",
                    "9-10": "90-100% of claims are perfectly grounded in tool outputs."
                }
            },
            "Tool Usage": {
                "Tool Appropriateness": {
                    "1-3": "10-30% of tools were perfectly selected for their subtasks.",
                    "4-6": "40-60% of tools were perfectly selected for their subtasks.",
                    "7-8": "70-80% of tools were perfectly selected for their subtasks.",
                    "9-10": "90-100% of tools were perfectly selected for their subtasks."
                },
                "Parameter Accuracy": {
                    "1-3": "10-30% of tool calls have perfectly accurate and complete parameters.",
                    "4-6": "40-60% of tool calls have perfectly accurate and complete parameters.",
                    "7-8": "70-80% of tool calls have perfectly accurate and complete parameters.",
                    "9-10": "90-100% of tool calls have perfectly accurate and complete parameters."
                }
            },
            "Planning Effectiveness and Efficiency": {
                "Dependency Awareness": {
                    "1-3": "10-30% of dependency chains are perfectly executed.",
                    "4-6": "40-60% of dependency chains are perfectly executed.",
                    "7-8": "70-80% of dependency chains are perfectly executed.",
                    "9-10": "90-100% of dependency chains are perfectly executed."
                },
                "Parallelism and Efficiency": {
                    "1-3": "More than 70% redundant calls OR less than 30% of parallelizable tasks were executed in parallel.",
                    "4-6": "40-60% redundant calls OR 40-60% of parallelizable tasks were executed in parallel.",
                    "7-8": "20-30% redundant calls AND 70-80% of parallelizable tasks were executed in parallel.",
                    "9-10": "Less than 10% redundant calls AND 90-100% of parallelizable tasks were executed in parallel."
                }
            }
        }
    
    async def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        """Call LLM for evaluation"""
        import litellm
        
        response = await litellm.acompletion(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        return response.choices[0].message.content
    
    def convert_trace_to_execution_results(self, trace) -> list[dict]:
        """Convert AgentTrace to execution_results format"""
        results = []
        
        for step in trace.steps:
            if step.message_type == "tool_call" and step.tool_name:
                tool_result = None
                tool_error = False
                
                for later_step in trace.steps:
                    if (later_step.message_type == "tool_result" and 
                        later_step.tool_name == step.tool_name):
                        tool_result = later_step.tool_result
                        tool_error = later_step.tool_error
                        break
                
                results.append({
                    "tool_name": step.tool_name,
                    "parameters": step.tool_arguments or {},
                    "result": tool_result or "",
                    "error": tool_error,
                })
        
        return results
    
    def build_accumulated_info(self, trace) -> str:
        """Build accumulated_information string"""
        lines = []
        
        for step in trace.steps:
            if step.message_type == "tool_call":
                lines.append(f"[Tool Call] {step.tool_name}")
                if step.tool_arguments:
                    import json
                    lines.append(f"  Parameters: {json.dumps(step.tool_arguments, ensure_ascii=False)}")
            elif step.message_type == "tool_result":
                result_preview = step.tool_result[:500] if step.tool_result else ""
                if step.tool_error:
                    lines.append(f"  Error: {result_preview}")
                else:
                    lines.append(f"  Result: {result_preview}")
            elif step.message_type == "llm_response":
                content_preview = step.content[:300] if step.content else ""
                lines.append(f"[Agent Response] {content_preview}")
        
        return "\n".join(lines)
    
    def build_available_tools(self, tools_schema: list[dict]) -> dict[str, dict]:
        """Convert tools_schema to available_tools format"""
        available_tools = {}
        
        for tool in tools_schema:
            func = tool.get("function", {})
            name = func.get("name", "")
            if name:
                parts = name.split("_", 1)
                server = parts[0] if len(parts) > 1 else "unknown"
                
                available_tools[name] = {
                    "server": server,
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                }
        
        return available_tools
    
    def _format_available_tools(self, available_tools: dict) -> str:
        """Format available tools list"""
        if not available_tools:
            return "No tools available"
        
        lines = []
        for name, info in list(available_tools.items())[:30]:
            desc = info.get("description", "")[:100]
            lines.append(f"- {name}: {desc}")
        
        if len(available_tools) > 30:
            lines.append(f"... and {len(available_tools) - 30} more tools")
        
        return "\n".join(lines)
    
    def _generate_randomized_prompt(self, task: str, final_solution: str, 
                                   execution_summary: str, total_rounds: int, 
                                   available_tools: dict = None,
                                   concrete_task_description: str = None,
                                   dependency_analysis: str = None) -> str:
        """Generate evaluation prompt with randomized structure (to prevent position bias)"""
        import random
        
        # Create dimension copies for randomization
        dimensions_copy = {}
        for main_dim, sub_dims in self.evaluation_dimensions.items():
            dimensions_copy[main_dim] = {}
            for sub_dim, criteria in sub_dims.items():
                dimensions_copy[main_dim][sub_dim] = dict(criteria)
        
        # Randomize main dimension order
        main_dimension_names = list(dimensions_copy.keys())
        random.shuffle(main_dimension_names)
        
        prompt_parts = []
        
        # Header
        prompt_parts.append("""You are an impartial evaluator judging the quality of an AI agent's multi-server tool-based task execution.

You must assign scores **only based on evidence** from the task, solution, and tool usage. Your evaluation should be:
- Objective (avoid being influenced by language fluency or formatting)
- Justified (include specific reasons tied to each score)
- Robust against bias (ignore narrative style, verbosity, or formatting polish)

---""")

        # Task description
        if concrete_task_description:
            prompt_parts.append(f"""
**TASK PRESENTED TO AGENT**: "{task}"

**CONCRETE TASK REFERENCE (For evaluation context only)**: 
Note: The agent did NOT see this concrete version. It only saw the task above. 
"{concrete_task_description}"
""")
        else:
            prompt_parts.append(f'**ORIGINAL TASK**: "{task}"')

        # Dependency analysis
        if dependency_analysis:
            prompt_parts.append(f"""
**DEPENDENCY ANALYSIS**:
Note: This analysis was generated during task creation to help understand tool dependencies.
The agent did NOT see this analysis.
{dependency_analysis[:1000]}
""")

        prompt_parts.append(f"""
**FINAL SOLUTION**: "{final_solution[:1500] if final_solution else 'No final solution provided'}"
**TOTAL ROUNDS**: {total_rounds}
**EXECUTION SUMMARY**:
{execution_summary[:3000]}

**AVAILABLE TOOLS** ({len(available_tools) if available_tools else 0} tools):
{self._format_available_tools(available_tools)}

---""")
        
        # Add randomized dimension scoring criteria
        for main_dim in main_dimension_names:
            sub_dims = dimensions_copy[main_dim]
            sub_dim_names = list(sub_dims.keys())
            random.shuffle(sub_dim_names)
            
            prompt_parts.append(f"\n### {main_dim} Rubric (1–10 per subdimension)\n")
            
            for i, sub_dim in enumerate(sub_dim_names, 1):
                criteria = sub_dims[sub_dim]
                criteria_items = list(criteria.items())
                random.shuffle(criteria_items)
                
                prompt_parts.append(f"{i}. **{sub_dim}**")
                for range_key, description in criteria_items:
                    prompt_parts.append(f"   - {range_key}: {description}")
                prompt_parts.append("")
        
        # Detailed scoring guidelines (consistent with original version)
        prompt_parts.extend([
            "---",
            "",
            "### PERCENTAGE-BASED SCORING SYSTEM:",
            "",
            "**How to Calculate Scores:**",
            "For each dimension, calculate the DEFECT RATE:",
            "- Defect Rate = (Number of Issues / Total Opportunities) × 100%",
            "",
            "Then map defect rate to score:",
            "- 0-10% defects → Score 9-10 (Excellent to Perfect)",
            "- 10-30% defects → Score 7-9 (Good performance)",
            "- 30-50% defects → Score 5-7 (Average performance)", 
            "- 50-70% defects → Score 3-5 (Poor performance)",
            "- 70-100% defects → Score 0-3 (Failed)",
            "",
            "**How to Score:**",
            '1. When evaluating percentages, be EXTREMELY STRICT about what counts as "perfectly executed"',
            '2. "Perfectly" means ALL of the following must be true:',
            "   - Correct tool selection (not just 'works' but OPTIMAL choice)",
            "   - Complete and accurate parameters (not just valid, but IDEAL)",
            "   - Zero redundancy (no repeated or unnecessary calls)",
            "   - Proper error handling (graceful recovery from ANY failure)",
            "   - Efficient execution (parallel when possible, minimal rounds)",
            "3. If ANY of the above is missing, that portion is NOT perfectly executed (counts as 0%)",
            "",
            "**KEY PRINCIPLES:**",
            "1. ALWAYS calculate as percentage, NOT absolute numbers",
            "2. 10 errors in 100 calls (10%) = same score as 1 error in 10 calls (10%)",
            "3. Consider the OPPORTUNITY COUNT for each dimension",
            "",
            "CRITICAL: Apply the STRICTEST interpretation of 'perfectly executed'. If there's ANY doubt, score lower.",
            "",
            "FORMAT NOTE: Text output when JSON not required = NO PENALTY (0% defect)",
            "FORMAT NOTE: Missing JSON when explicitly required = Count as failed requirement",
            "",
            "Remember: Most real-world executions should score 4-6. Scores of 8+ should be EXCEPTIONAL.",
            "",
            "FINAL REMINDER BEFORE SCORING:",
            "- Default to 4-5 unless you have strong evidence for higher",
            "- Count ONLY truly perfect executions toward the percentage",
            "- Be your most critical self - find flaws first, then acknowledge successes",
            "- If you're considering a score above 7, re-examine for ANY imperfection",
            "",
            "---",
            "",
            "Please return your evaluation in this exact JSON format:",
            "{",
            '  "task_fulfillment_reasoning": "Explain how well the agent fulfilled the task objectives and what percentage was completed.",',
            '  "grounding_reasoning": "Explain how well the agent\'s outputs were grounded in actual tool results.",',
            '  "tool_appropriateness_reasoning": "Explain whether the tools selected were appropriate for each subtask.",',
            '  "parameter_accuracy_reasoning": "Explain the accuracy and completeness of parameters used in tool calls.",',
            '  "dependency_awareness_reasoning": "Explain how well the agent understood and respected task dependencies.",',
            '  "parallelism_efficiency_reasoning": "Explain the efficiency of execution, including parallelism and avoiding redundancy.",',
            "",
            '  "task_fulfillment": X,',
            '  "grounding": X,',
            '  "tool_appropriateness": X,',
            '  "parameter_accuracy": X,',
            '  "dependency_awareness": X,',
            '  "parallelism_and_efficiency": X',
            "}",
            "",
            "Return **only** the JSON object."
        ])
        
        return "\n".join(prompt_parts)
    
    def _calculate_average_scores(self, all_scores: list[dict]) -> dict:
        """Calculate average scores from multiple evaluations"""
        if not all_scores:
            raise ValueError("No scores to average")
        
        score_fields = [
            "task_fulfillment", "grounding",
            "tool_appropriateness", "parameter_accuracy", 
            "dependency_awareness", "parallelism_and_efficiency"
        ]
        
        avg_scores = {}
        for field in score_fields:
            values = [s.get(field, 5) for s in all_scores]
            avg_scores[field] = sum(values) / len(values)
        
        return avg_scores
    
    def _is_token_limit_error(self, error_message: str) -> bool:
        """Check if this is a token limit error"""
        error_lower = str(error_message).lower()
        token_limit_indicators = [
            "maximum context length",
            "context length",
            "token limit",
            "too many tokens",
            "exceeds maximum",
            "requested too many tokens"
        ]
        return any(indicator in error_lower for indicator in token_limit_indicators)
    
    async def compress_for_judge(self, accumulated_info: str, target_tokens: int = 10000) -> str:
        """
        Compress execution history for Judge evaluation
        
        Retains more details than Agent compression, focusing on evidence needed for evaluation
        
        Args:
            accumulated_info: Execution history to compress
            target_tokens: Target token count (default 10000)
            
        Returns:
            Compressed execution history optimized for evaluation
        """
        if not accumulated_info:
            return ""
        
        current_tokens = len(accumulated_info) // 4
        if current_tokens <= target_tokens:
            return accumulated_info
        
        logger.info(f"Compressing execution history for judge: {current_tokens} -> {target_tokens} tokens")
        
        system_prompt = """You are compressing execution history for a judge model that needs to evaluate agent performance. Focus on preserving evidence of task completion and decision quality."""
        
        user_prompt = f"""Compress the following agent execution history for evaluation purposes.

CRITICAL REQUIREMENTS FOR JUDGE COMPRESSION:
1. Preserve ALL tool calls with their exact parameters and server names
2. Keep key results and error messages that show task progress
3. Maintain the chronological sequence of execution rounds
4. Retain evidence of success/failure for each operation
5. Keep decision reasoning that shows agent's problem-solving approach
6. Preserve numerical results, data values, and specific findings
7. Keep any information that proves task completion or failure

Target length: approximately {target_tokens} tokens

EXECUTION HISTORY TO COMPRESS:
{accumulated_info}

COMPRESSED VERSION FOR EVALUATION:"""
        
        try:
            compressed = await self._call_llm(system_prompt, user_prompt, target_tokens)
            
            compressed_tokens = len(compressed) // 4
            logger.info(f"Judge compression successful: {current_tokens} -> {compressed_tokens} tokens")
            return compressed.strip()
            
        except Exception as e:
            logger.error(f"Failed to compress for judge: {e}")
            # Fall back to simple truncation on compression failure
            target_chars = target_tokens * 4
            if len(accumulated_info) > target_chars:
                return accumulated_info[:target_chars] + "\n[Truncated for token limit]"
            return accumulated_info
    
    async def evaluate_async(
        self,
        trace,
        task_description: str,
        fuzzy_description: str,
        dependency_analysis: str = None,
        available_tools: dict = None,
    ) -> NativeEvalResult:
        """
        Async evaluation - 6-dimension 1-10 scoring
        
        Supports Judge Stability (average of multiple evaluations) and compression retry on token limit errors
        """
        import json
        import re
        
        # Track the accumulated_info currently in use (may be compressed)
        accumulated_info = self.build_accumulated_info(trace)
        accumulated_info_to_use = accumulated_info
        
        # Maximum 2 retries (original + compressed retry)
        max_retries = 2
        for retry_attempt in range(max_retries):
            try:
                final_solution = trace.final_response or ""
                total_rounds = trace.total_steps
                
                all_scores = []
                num_runs = self.judge_stability_runs if self.enable_judge_stability else 1
                
                for run_idx in range(num_runs):
                    # Generate randomized prompt each time
                    eval_prompt = self._generate_randomized_prompt(
                        task=fuzzy_description,
                        final_solution=final_solution,
                        execution_summary=accumulated_info_to_use,
                        total_rounds=total_rounds,
                        available_tools=available_tools,
                        concrete_task_description=task_description,
                        dependency_analysis=dependency_analysis,
                    )
                    
                    system_prompt = "You are an expert AI task execution evaluator. Provide detailed scoring with reasoning."
                    
                    response = await self._call_llm(system_prompt, eval_prompt, max_tokens=4096)
                    
                    # Parse JSON
                    json_match = re.search(r'\{[\s\S]*\}', response)
                    if json_match:
                        try:
                            eval_result = json.loads(json_match.group())
                            all_scores.append(eval_result)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON in run {run_idx + 1}")
                    else:
                        logger.warning(f"No JSON found in run {run_idx + 1}")
                
                # Calculate average scores
                if all_scores:
                    if len(all_scores) > 1:
                        avg_scores = self._calculate_average_scores(all_scores)
                    else:
                        avg_scores = all_scores[0]
                else:
                    avg_scores = {
                        "task_fulfillment": 5,
                        "grounding": 5,
                        "tool_appropriateness": 5,
                        "parameter_accuracy": 5,
                        "dependency_awareness": 5,
                        "parallelism_and_efficiency": 5,
                    }
                
                # Extract 6-dimension scores (1-10 range)
                task_fulfillment = max(1, min(10, float(avg_scores.get("task_fulfillment", 5))))
                grounding = max(1, min(10, float(avg_scores.get("grounding", 5))))
                tool_appropriateness = max(1, min(10, float(avg_scores.get("tool_appropriateness", 5))))
                parameter_accuracy = max(1, min(10, float(avg_scores.get("parameter_accuracy", 5))))
                dependency_awareness = max(1, min(10, float(avg_scores.get("dependency_awareness", 5))))
                parallelism_efficiency = max(1, min(10, float(avg_scores.get("parallelism_and_efficiency", 5))))
                
                # Calculate composite scores (consistent with original version)
                task_completion_score = (task_fulfillment + grounding) / 2
                tool_selection_score = (tool_appropriateness + parameter_accuracy) / 2
                planning_score = (dependency_awareness + parallelism_efficiency) / 2
                
                # Overall score = average of three composite scores (1-10 range)
                overall_score = (task_completion_score + tool_selection_score + planning_score) / 3
                
                # Convert reward to 0-1 range
                reward = overall_score / 10.0
                
                return NativeEvalResult(
                    task_id=trace.task_id,
                    domain="mcpbench",
                    benchmark="mcpbench",
                    reward=reward,
                    success=overall_score >= 5.0,  # Score 5 or above is considered successful
                    info={
                        # 6-dimension scores (1-10)
                        "task_fulfillment": task_fulfillment,
                        "grounding": grounding,
                        "tool_appropriateness": tool_appropriateness,
                        "parameter_accuracy": parameter_accuracy,
                        "dependency_awareness": dependency_awareness,
                        "parallelism_and_efficiency": parallelism_efficiency,
                        # Composite scores
                        "task_completion_score": task_completion_score,
                        "tool_selection_score": tool_selection_score,
                        "planning_effectiveness_and_efficiency_score": planning_score,
                        "overall_score": overall_score,
                        # Detailed info
                        "task_fulfillment_reasoning": avg_scores.get("task_fulfillment_reasoning", ""),
                        "grounding_reasoning": avg_scores.get("grounding_reasoning", ""),
                        "tool_appropriateness_reasoning": avg_scores.get("tool_appropriateness_reasoning", ""),
                        "parameter_accuracy_reasoning": avg_scores.get("parameter_accuracy_reasoning", ""),
                        "dependency_awareness_reasoning": avg_scores.get("dependency_awareness_reasoning", ""),
                        "parallelism_efficiency_reasoning": avg_scores.get("parallelism_efficiency_reasoning", ""),
                        # Meta info
                        "judge_stability_runs": num_runs,
                        "raw_response": response if not self.enable_judge_stability else f"[{num_runs} runs averaged]",
                    },
                )
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if this is a token limit error and hasn't been compressed yet
                if self._is_token_limit_error(error_msg) and retry_attempt == 0 and accumulated_info:
                    logger.info("Token limit error in judge evaluation, compressing execution history to 10000 tokens...")
                    accumulated_info_to_use = await self.compress_for_judge(accumulated_info, target_tokens=10000)
                    logger.info("Retrying evaluation with compressed execution history...")
                    continue
                
                # If not a token limit error or compression already attempted, raise
                logger.exception(f"MCPBench evaluation failed for task {trace.task_id}")
                return NativeEvalResult(
                    task_id=trace.task_id,
                    domain="mcpbench",
                    benchmark="mcpbench",
                    reward=0.0,
                    success=False,
                    error=str(e),
                )
        
        # Should not reach here
        raise Exception("Unexpected evaluation flow")
    
    def evaluate(
        self,
        trace,
        task_description: str,
        fuzzy_description: str,
        dependency_analysis: str = None,
        available_tools: dict = None,
    ) -> NativeEvalResult:
        """Synchronous evaluation wrapper"""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.evaluate_async(
                trace, task_description, fuzzy_description,
                dependency_analysis, available_tools
            )
        )


class SearchEvaluator:
    """
    Search benchmark native evaluator
    
    Directly calls original evaluation functions from deepresearch_llm_modeling/eval_scripts/
    
    Supports three evaluation modes:
    1. browsecomp: Uses LLM-as-judge to check if answer is semantically equivalent to golden_answer
    2. mind2web: Uses rubric tree evaluation (includes correctness and attribution checks)
    3. webvoyager: Uses LLM to evaluate if web navigation task was successfully completed
    """
    
    # Evaluation model
    JUDGE_MODEL = "gpt-4o"  # Use gpt-4o for better evaluation quality
    
    # Fallback prompts (used when original functions are not available)
    FAILURE_INDICATORS = [
        'did not find', 'could not find', 'unable to find', 'unable to identify',
        'cannot be identified', 'cannot be definitively identified', 'not found', 
        'no information', 'insufficient information', 'could not be determined', 
        'cannot determine', 'i don\'t know', 'i do not know', 'no answer', 
        'answer not available', 'information not available', 'not publicly available'
    ]
    
    BROWSECOMP_PROMPT = """You are evaluating if an agent's answer is correct.
**Question**: {question}
**Ground Truth Answer**: {correct_answer}
**Agent's Answer**: {agent_answer}

Respond with "CORRECT" or "INCORRECT" based on semantic equivalence.
"""
    
    MIND2WEB_CORRECTNESS_PROMPT = """Evaluate if the agent's answer satisfies the criteria.
**Question**: {question}
**Answer**: {answer}
**Criteria**: {description} {criteria}

Respond with "PASS" or "FAIL".
"""

    MIND2WEB_ATTRIBUTION_PROMPT = """Evaluate if the answer is supported by sources.
**Answer**: {answer}
**Sources**: {source_context}
**Criteria**: {description} {criteria}

Respond with "PASS" or "FAIL".
"""

    WEBVOYAGER_PROMPT = """Evaluate if the agent completed the web navigation task.
**Task**: {task}
**Answer**: {answer}

Respond with "SUCCESS" or "FAILURE".
"""

    def __init__(self):
        """Initialize evaluator, import original evaluation functions"""
        import os
        import re
        import sys
        from dotenv import load_dotenv
        
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._re = re
        
        # Add original evaluation script path
        eval_scripts_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "benchmarks",
            "deepresearch_llm_modeling",
            "eval_scripts"
        )
        if eval_scripts_path not in sys.path:
            sys.path.insert(0, eval_scripts_path)
        
        # Dynamically import original evaluation functions
        try:
            from browsecomp_eval_pass1 import evaluate_browsecomp_answer
            from mind2web2_eval_pass1 import (
                evaluate_rubric_node as _evaluate_rubric_node,
                extract_citations_from_answer,
            )
            from webvoyager_eval_pass1 import evaluate_single_result
            
            self._orig_browsecomp_eval = evaluate_browsecomp_answer
            self._orig_rubric_eval = _evaluate_rubric_node
            self._orig_webvoyager_eval = evaluate_single_result
            self._orig_extract_citations = extract_citations_from_answer
            self._use_original_functions = True
            logger.info("[SEARCH EVAL] Successfully imported original eval functions from deepresearch_llm_modeling")
        except ImportError as e:
            logger.warning(f"[SEARCH EVAL] Could not import original eval functions: {e}")
            logger.warning("[SEARCH EVAL] Falling back to built-in evaluation logic")
            self._use_original_functions = False
        
        # Create OpenAI client for fallback (may be needed even when using original functions)
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, timeout=180)
        
        # Cache sources extracted from trace
        self._cached_sources: list[str] = []
        self._cached_source_contents: dict[str, str] = {}
    
    def _extract_sources_from_trace(self, trace) -> tuple[list[str], dict[str, str]]:
        """
        Extract all sources (URLs) from AgentTrace
        
        Based on deepresearch_llm_modeling/eval_scripts/mind2web2_eval_pass1.py
        
        Returns:
            (sources_list, source_contents_dict)
        """
        sources = []
        source_contents = {}
        
        # URL regex pattern
        url_pattern = r'https?://[^\s<>"\'`|(){}[\]]+[^\s<>"\'`|(){}[\].,;:]'
        
        # Iterate through all steps in trace
        if hasattr(trace, 'steps'):
            for step in trace.steps:
                # Check <information> blocks in tool_result
                tool_result = step.tool_result if hasattr(step, 'tool_result') else None
                if tool_result:
                    info_matches = self._re.findall(
                        r'<information>(.*?)</information>', 
                        str(tool_result), 
                        self._re.DOTALL
                    )
                    for info_block in info_matches:
                        urls = self._re.findall(url_pattern, info_block)
                        for url in urls:
                            if url not in source_contents:
                                sources.append(url)
                                source_contents[url] = info_block
                    
                    # Also check URLs without <information> tags
                    if not info_matches:
                        urls = self._re.findall(url_pattern, str(tool_result))
                        for url in urls:
                            if url not in source_contents:
                                sources.append(url)
                                source_contents[url] = str(tool_result)[:1000]
        
        # Also check messages
        if hasattr(trace, 'messages'):
            for msg in trace.messages:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if content and msg.role if hasattr(msg, 'role') else True:
                    # Only check tool result messages
                    role = msg.role if hasattr(msg, 'role') else 'unknown'
                    if role in ['tool', 'function']:
                        info_matches = self._re.findall(
                            r'<information>(.*?)</information>', 
                            str(content), 
                            self._re.DOTALL
                        )
                        for info_block in info_matches:
                            urls = self._re.findall(url_pattern, info_block)
                            for url in urls:
                                if url not in source_contents:
                                    sources.append(url)
                                    source_contents[url] = info_block
        
        return sources, source_contents
    
    def _extract_citations_from_answer(self, answer: str) -> list[str]:
        """
        Extract cited URLs from answer
        
        Returns:
            list of cited URLs
        """
        url_pattern = r'https?://[^\s<>"\'`|(){}[\]]+[^\s<>"\'`|(){}[\].,;:]'
        return self._re.findall(url_pattern, answer)
    
    def _call_llm(self, prompt: str, max_tokens: int = 150, temperature: float = 0) -> str:
        """Call LLM for evaluation"""
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.JUDGE_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content.strip()
                if content:
                    return content
            except Exception as e:
                logger.warning(f"OpenAI API Error (attempt {attempt+1}/3): {e}")
        return ""
    
    def _check_failure_indicators(self, answer: str) -> bool:
        """Check if answer contains failure indicators"""
        answer_lower = answer.lower().strip()
        return any(indicator in answer_lower for indicator in self.FAILURE_INDICATORS)
    
    def _evaluate_browsecomp(
        self,
        question: str,
        correct_answer: str,
        agent_answer: str,
    ) -> tuple[bool, str]:
        """
        Evaluate BrowseComp task - directly calls original evaluation function
        
        Returns:
            (is_correct, reasoning)
        """
        # If original function available, call it directly
        if self._use_original_functions:
            try:
                is_correct = self._orig_browsecomp_eval(
                    question=question,
                    correct_answer=correct_answer,
                    agent_answer=agent_answer,
                    api_key=self.api_key,
                    model=self.JUDGE_MODEL
                )
                reasoning = "Evaluated using original browsecomp_eval_pass1.evaluate_browsecomp_answer()"
                return is_correct, reasoning
            except Exception as e:
                logger.warning(f"[SEARCH EVAL] Original browsecomp eval failed: {e}, using fallback")
        
        # Fallback: use built-in evaluation logic
        # Check answer length
        if len(agent_answer.strip()) < 3:
            return False, "Answer is too short (less than 3 characters)"
        
        # 3. Use LLM-as-judge for evaluation
        prompt = self.BROWSECOMP_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            agent_answer=agent_answer,
        )
        
        response = self._call_llm(prompt, max_tokens=100, temperature=0)
        
        if not response:
            return False, "LLM evaluation failed - no response"
        
        # Parse results
        lines = response.strip().splitlines()
        last_line = lines[-1].strip().upper() if lines else ""
        reasoning = "\n".join(lines[:-1]).strip() if len(lines) > 1 else response
        
        is_correct = last_line == "CORRECT"
        return is_correct, reasoning
    
    def _evaluate_rubric_node(
        self,
        node: dict,
        answer: str,
        question: str,
    ) -> tuple[bool, str]:
        """
        Recursively evaluate rubric node (from mind2web2_eval_pass1.py)
        
        Returns:
            (is_pass, reasoning)
        """
        node_type = node.get('type')
        
        if node_type in ('and', 'or'):
            children = node.get('children', [])
            if not children:
                return True, f"No criteria under this '{node_type}' node."
            
            all_reasons = []
            final_pass = (node_type == 'and')  # 'and' defaults to True, 'or' defaults to False
            
            for i, child in enumerate(children):
                is_pass, reason = self._evaluate_rubric_node(child, answer, question)
                status = 'PASS' if is_pass else 'FAIL'
                child_type = child.get('type')
                
                if child_type in ['correctness', 'attribution']:
                    formatted_reason = f"  - Criteria #{i+1} ({status}): {reason}"
                else:
                    indented_sub_reason = "    " + reason.replace("\n", "\n    ")
                    formatted_reason = f"  - Criteria #{i+1} ({status}):\n{indented_sub_reason}"
                
                all_reasons.append(formatted_reason)
                
                if node_type == 'and' and not is_pass:
                    final_pass = False
                if node_type == 'or' and is_pass:
                    final_pass = True
            
            return final_pass, "\n".join(all_reasons)
        
        elif node_type == 'correctness':
            return self._evaluate_correctness(node, answer, question)
        
        elif node_type == 'attribution':
            # Attribution: use LLM to evaluate if citations correctly support the answer
            return self._evaluate_attribution(node, answer, question)
        
        else:
            return False, f"Unknown node type: {node_type}"
    
    def _evaluate_correctness(
        self,
        node: dict,
        answer: str,
        question: str,
    ) -> tuple[bool, str]:
        """Evaluate correctness node"""
        description = node.get('description', '')
        criteria = node.get('criteria', '')
        
        prompt = self.MIND2WEB_CORRECTNESS_PROMPT.format(
            question=question,
            answer=answer,
            description=description,
            criteria=criteria,
        )
        
        response = self._call_llm(prompt, max_tokens=150, temperature=0)
        
        if not response:
            return False, "LLM evaluation failed - no response"
        
        lines = response.strip().splitlines()
        if not lines:
            return False, "LLM returned empty response"
        
        verdict_line = lines[-1].strip().upper()
        reasoning = "\n".join(lines[:-1]).strip() or "No reasoning provided."
        
        is_pass = "PASS" in verdict_line and "FAIL" not in verdict_line
        return is_pass, reasoning
    
    def _evaluate_attribution(
        self,
        node: dict,
        answer: str,
        question: str,
    ) -> tuple[bool, str]:
        """
        Evaluate attribution node - check if citations in answer correctly support content
        
        Based on deepresearch_llm_modeling/eval_scripts/mind2web2_eval_pass1.py
        
        Returns:
            (is_pass, reasoning)
        """
        description = node.get('description', '')
        criteria = node.get('criteria', '')
        
        # Extract cited URLs from answer
        cited_urls = self._extract_citations_from_answer(answer)
        
        # Build source context
        source_context = ""
        for url in cited_urls[:5]:  # Take at most 5 URLs
            content = self._cached_source_contents.get(url, "")
            if content:
                source_context += f"\n\n**Source: {url}**\n{content[:500]}..."
        
        # If no source was cited, may fail
        if not cited_urls:
            # Check if there are cached sources
            if self._cached_sources:
                # Sources available but answer didn't cite any
                return False, "Attribution Failure: The agent's answer did not cite any sources, but sources were available during search."
            else:
                # No sources available to cite, skip attribution
                return True, "Attribution skipped: No sources found in trajectory to cite."
        
        if not source_context:
            # Has citations but content is empty
            return False, "Attribution Failure: Cited URLs were not found in the search trajectory."
        
        # Use LLM to evaluate attribution
        prompt = self.MIND2WEB_ATTRIBUTION_PROMPT.format(
            answer=answer,
            source_context=source_context,
            description=description,
            criteria=criteria,
        )
        
        response = self._call_llm(prompt, max_tokens=150, temperature=0)
        
        if not response:
            return False, "LLM evaluation failed - no response"
        
        lines = response.strip().splitlines()
        if not lines:
            return False, "LLM returned empty response"
        
        verdict_line = lines[-1].strip().upper()
        reasoning = "\n".join(lines[:-1]).strip() or "No reasoning provided."
        
        is_pass = "PASS" in verdict_line and "FAIL" not in verdict_line
        return is_pass, reasoning
    
    def _evaluate_webvoyager(
        self,
        task: str,
        answer: str,
    ) -> tuple[bool, str]:
        """
        Evaluate WebVoyager task - directly calls original evaluation function
        
        Returns:
            (is_success, reasoning)
        """
        # If original function available, call it directly
        if self._use_original_functions:
            try:
                is_success, reasoning = self._orig_webvoyager_eval(
                    task=task,
                    answer=answer,
                    api_key=self.api_key,
                    model=self.JUDGE_MODEL
                )
                return is_success, reasoning
            except Exception as e:
                logger.warning(f"[SEARCH EVAL] Original webvoyager eval failed: {e}, using fallback")
        
        # Fallback: built-in evaluation logic
        prompt = self.WEBVOYAGER_PROMPT.format(
            task=task,
            answer=answer,
        )
        
        response = self._call_llm(prompt, max_tokens=150, temperature=0)
        
        if not response:
            return False, "LLM evaluation failed - no response"
        
        lines = response.strip().splitlines()
        if not lines:
            return False, "LLM returned empty response"
        
        verdict_line = lines[-1].strip().upper()
        reasoning = "\n".join(lines[:-1]).strip().replace("Reasoning: ", "") or "No reasoning provided."
        
        is_success = verdict_line == "SUCCESS"
        return is_success, reasoning

    def _evaluate_mind2web(
        self,
        question: str,
        answer: str,
        rubric: dict,
    ) -> tuple[bool, str]:
        """
        Evaluate Mind2Web task - directly calls original evaluation function
        
        Returns:
            (is_pass, reasoning)
        """
        if not rubric:
            return False, "No rubric provided for Mind2Web task"
        
        # If original function available, call it directly
        if self._use_original_functions:
            try:
                is_pass, reasoning = self._orig_rubric_eval(
                    node=rubric,
                    answer=answer,
                    sources=self._cached_sources,
                    source_contents=self._cached_source_contents,
                    question=question,
                    api_key=self.api_key,
                    model=self.JUDGE_MODEL
                )
                return is_pass, reasoning
            except Exception as e:
                logger.warning(f"[SEARCH EVAL] Original mind2web rubric eval failed: {e}, using fallback")
        
        # Fallback: built-in rubric evaluation logic
        return self._evaluate_rubric_node(rubric, answer, question)
    
    def evaluate(
        self,
        trace,
        task: dict,
        search_count: int = 0,
    ) -> NativeEvalResult:
        """
        Evaluate search task
        
        Automatically selects evaluation method based on task type:
        - browsecomp: Uses LLM-as-judge to check semantic equivalence
        - mind2web: Uses rubric tree evaluation (includes correctness and attribution)
        - webvoyager: Uses LLM to evaluate web navigation task
        - other: Uses simple semantic equivalence evaluation
        
        Args:
            trace: AgentTrace
            task: Task dict containing id, question, golden_answer, type, rubric, etc.
            search_count: Number of searches
            
        Returns:
            NativeEvalResult
        """
        question = task.get("question", "")
        golden_answer = task.get("golden_answer", task.get("answer", ""))
        predicted_answer = trace.final_response or ""
        task_id = str(task.get("id", trace.task_id))
        task_type = task.get("type", "").lower()
        rubric = task.get("rubric")
        
        # Extract sources for attribution evaluation
        self._cached_sources, self._cached_source_contents = self._extract_sources_from_trace(trace)
        
        try:
            # Select evaluation method based on task type
            if task_type == "webvoyager":
                # WebVoyager: use LLM to evaluate web navigation task
                is_correct, reasoning = self._evaluate_webvoyager(question, predicted_answer)
                score = 1 if is_correct else 0
                eval_method = "webvoyager"
            elif task_type == "mind2web" and rubric:
                # Mind2Web: use rubric evaluation
                is_correct, reasoning = self._evaluate_mind2web(question, predicted_answer, rubric)
                score = 1 if is_correct else 0
                eval_method = "rubric"
            elif task_type == "browsecomp" or golden_answer:
                # BrowseComp or tasks with answers: use LLM-as-judge evaluation
                is_correct, reasoning = self._evaluate_browsecomp(question, golden_answer, predicted_answer)
                score = 1 if is_correct else 0
                eval_method = "llm_judge"
            else:
                # No answer and no rubric: cannot evaluate
                return NativeEvalResult(
                    task_id=task_id,
                    domain="search",
                    benchmark="search",
                    reward=0.0,
                    success=False,
                    error="No golden_answer or rubric provided for evaluation",
                    info={
                        "question": question,
                        "predicted_answer": predicted_answer,
                        "search_count": search_count,
                    },
                )
            
            logger.info(f"[SEARCH EVAL] Task {task_id} ({task_type}): {'PASS' if score == 1 else 'FAIL'}")
            logger.debug(f"[SEARCH EVAL] Reasoning: {reasoning[:200]}...")
            
            return NativeEvalResult(
                task_id=task_id,
                domain="search",
                benchmark="search",
                reward=float(score),
                success=score == 1,
                info={
                    "score": score,
                    "reasoning": reasoning,
                    "eval_method": eval_method,
                    "task_type": task_type,
                    "question": question,
                    "golden_answer": golden_answer,
                    "predicted_answer": predicted_answer,
                    "search_count": search_count,
                    "sources_found": len(self._cached_sources),
                },
            )
            
        except Exception as e:
            logger.exception(f"Search evaluation failed for task {task_id}")
            return NativeEvalResult(
                task_id=task_id,
                domain="search",
                benchmark="search",
                reward=0.0,
                success=False,
                error=str(e),
            )


class MathHayEvaluator:
    """
    MathHay native evaluator
    
    Evaluates long-context math reasoning tasks. Uses two evaluation methods:
    1. Numerical comparison: math.isclose(expected, predicted, rel_tol=1e-9)
    2. LLM Verification: GPT-4o determines if two solutions are equivalent
    
    Final verdict: Numerical match OR LLM judges "Yes"
    
    Based on MathHay/evaluation/evaluation.py
    """
    
    # LLM Verification prompt (from original MathHay)
    LLM_VERIFICATION_PROMPT = """Your task is to determine if the two given solutions are equivalent in terms of reasoning and final answer.

Solution 1:
{solution1}

Solution 2:
{solution2}

Criteria for equivalence:
1. Both solutions should have the same reasoning steps leading to the final answer.
2. The final numerical answers should be identical.

Please analyze the two solutions and state whether they are the same or different. If different, provide a brief explanation of the discrepancies.

Example:
Solution 1:
def solve():
    current_value = 45e9  # $45 billion
    projected_value = 400e9  # $400 billion
    answer = projected_value - current_value
    return answer
Answer1: 355000000000.0

Solution 2:
The current value of the AI chip market is projected to be $45 billion, and it is expected to rise to $400 billion by 2027. To find the difference, we subtract the current value from the projected value: $400 billion - $45 billion = $355 billion.
Answer2: 355.0

Output: Yes

Respond with a JSON object:
{{"reasoning": "your verification reasoning", "output": "Yes or No"}}
"""
    
    # Evaluation model
    JUDGE_MODEL = "gpt-4o"
    
    def __init__(self, use_llm_verification: bool = True):
        """
        Initialize MathHay evaluator
        
        Args:
            use_llm_verification: Whether to use LLM verification (in addition to numerical comparison)
        """
        import os
        import math
        import re
        from dotenv import load_dotenv
        
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.use_llm_verification = use_llm_verification
        self._math = math
        self._re = re
        
        # Initialize OpenAI client
        if self.use_llm_verification and self.api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, timeout=180)
        else:
            self.client = None
            if self.use_llm_verification:
                logger.warning("[MATHHAY EVAL] OpenAI API key not set, LLM verification disabled")
    
    def _compare_answers(self, expected: float, predicted: float) -> bool:
        """
        Numerical comparison (consistent with original MathHay)
        
        Uses math.isclose with rel_tol=1e-9
        """
        try:
            if expected is None or predicted is None:
                return False
            return self._math.isclose(float(expected), float(predicted), rel_tol=1e-9)
        except (ValueError, TypeError):
            return False
    
    def _extract_answer_from_response(self, response: str) -> tuple[float, str]:
        """
        Extract answer and reasoning from model response
        
        Prioritizes using original MathHay extract_json_from_string function.
        
        Tries multiple formats:
        1. Original MathHay extract_json_from_string
        2. JSON format: {"reasoning": "...", "answer": 123.45}
        3. <answer> tag: <answer>123.45</answer>
        4. Direct number
        
        Returns:
            (answer, reasoning)
        """
        if not response:
            return None, ""
        
        answer = None
        reasoning = response
        
        # Prioritize using original MathHay extract_json_from_string
        try:
            import sys
            mathhay_path = str(Path(__file__).parent.parent.parent / "benchmarks" / "MathHay")
            if mathhay_path not in sys.path:
                sys.path.insert(0, mathhay_path)
            from bench_generation.utils.tools import extract_json_from_string
            
            parsed = extract_json_from_string(response)
            if parsed and "answer" in parsed:
                answer = float(parsed.get("answer", 0))
                reasoning = parsed.get("reasoning", response)
                return answer, reasoning
        except Exception as e:
            logger.debug(f"[MATHHAY EVAL] Failed to use MathHay extract_json_from_string: {e}")
        
        # Fallback method: try JSON format
        try:
            import json
            # Find JSON block
            json_match = self._re.search(r'\{[^{}]*"answer"[^{}]*\}', response, self._re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                answer = float(parsed.get("answer", 0))
                reasoning = parsed.get("reasoning", response)
                return answer, reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # Try <answer> tag
        answer_match = self._re.search(r'<answer>\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*</answer>', response)
        if answer_match:
            try:
                answer = float(answer_match.group(1))
                return answer, response
            except ValueError:
                pass
        
        # Try to find last number (may be the final answer)
        numbers = self._re.findall(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?', response)
        if numbers:
            try:
                # Take the last non-zero number as the answer
                for num_str in reversed(numbers):
                    num = float(num_str)
                    if num != 0:
                        answer = num
                        break
            except ValueError:
                pass
        
        return answer, reasoning
    
    def _llm_verification(self, solution1: str, solution2: str) -> tuple[str, str]:
        """
        Use LLM to verify if two solutions are equivalent
        
        Returns:
            (output: "Yes"/"No", reasoning)
        """
        if not self.client:
            return "None", "LLM verification not available"
        
        prompt = self.LLM_VERIFICATION_PROMPT.format(
            solution1=solution1,
            solution2=solution2
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                import json
                json_match = self._re.search(r'\{.*\}', raw_output, self._re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return parsed.get("output", "None"), parsed.get("reasoning", raw_output)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Simple check for Yes/No
            if "yes" in raw_output.lower():
                return "Yes", raw_output
            elif "no" in raw_output.lower():
                return "No", raw_output
            
            return "None", raw_output
            
        except Exception as e:
            logger.warning(f"[MATHHAY EVAL] LLM verification failed: {e}")
            return "None", str(e)
    
    def evaluate(
        self,
        trace,
        task: dict,
    ) -> NativeEvalResult:
        """
        Evaluate MathHay task
        
        Args:
            trace: AgentTrace
            task: Task dict
            
        Returns:
            NativeEvalResult
        """
        task_id = str(task.get("id", "unknown"))
        question = task.get("question", "")
        golden_answer = task.get("golden_answer")
        golden_solution = task.get("solution", "")
        task_type = task.get("task_type", "3s3d")
        
        # Extract predicted answer from trace
        predicted_response = trace.final_response or ""
        predicted_answer, predicted_reasoning = self._extract_answer_from_response(predicted_response)
        
        try:
            # 1. Numerical comparison
            numerical_match = self._compare_answers(golden_answer, predicted_answer)
            
            # 2. LLM Verification (if enabled)
            llm_judge = "None"
            judge_reasoning = ""
            
            if self.use_llm_verification and not numerical_match and predicted_answer is not None:
                # Build solution strings
                solution1 = f"{golden_solution}\nAnswer1: {golden_answer}"
                solution2 = f"{predicted_reasoning}\nAnswer2: {predicted_answer}"
                llm_judge, judge_reasoning = self._llm_verification(solution1, solution2)
            
            # 3. Final verdict: numerical match OR LLM judges Yes
            correct = numerical_match or ("yes" in str(llm_judge).lower())
            score = 1 if correct else 0
            
            logger.info(f"[MATHHAY EVAL] Task {task_id}: {'PASS' if correct else 'FAIL'}")
            logger.info(f"  Expected: {golden_answer}, Predicted: {predicted_answer}")
            logger.info(f"  Numerical match: {numerical_match}, LLM judge: {llm_judge}")
            
            return NativeEvalResult(
                task_id=task_id,
                domain="math_reasoning",
                benchmark="mathhay",
                reward=float(score),
                success=correct,
                info={
                    "score": score,
                    "task_type": task_type,
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "golden_answer": golden_answer,
                    "predicted_answer": predicted_answer,
                    "numerical_match": numerical_match,
                    "llm_judge": llm_judge,
                    "judge_reasoning": judge_reasoning[:500] if judge_reasoning else "",
                    "predicted_reasoning": predicted_reasoning[:500] if predicted_reasoning else "",
                },
            )
            
        except Exception as e:
            logger.exception(f"MathHay evaluation failed for task {task_id}")
            return NativeEvalResult(
                task_id=task_id,
                domain="math_reasoning",
                benchmark="mathhay",
                reward=0.0,
                success=False,
                error=str(e),
            )


class NativeEvaluatorRegistry:
    """Native evaluator registry"""
    
    _evaluators: dict[str, Any] = {
        "tau2": Tau2Evaluator,
        "mcpbench": MCPBenchEvaluator,
        "search": SearchEvaluator,
        "mathhay": MathHayEvaluator,
    }
    
    @classmethod
    def get_evaluator(cls, benchmark: str, **kwargs):
        """
        Get evaluator for the specified benchmark
        
        Args:
            benchmark: Benchmark name
            **kwargs: Evaluator parameters
            
        Returns:
            Evaluator instance
        """
        if benchmark not in cls._evaluators:
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(cls._evaluators.keys())}")
        return cls._evaluators[benchmark](**kwargs)
    
    @classmethod
    def register(cls, name: str, evaluator_class):
        """Register evaluator"""
        cls._evaluators[name] = evaluator_class


def evaluate_with_native(
    trace,
    task: Any,
    benchmark: str,
    domain: str,
    **kwargs
) -> NativeEvalResult:
    """
    Evaluate using native evaluator
    
    Args:
        trace: AgentTrace
        task: Task object (benchmark-specific format)
        benchmark: Benchmark name
        domain: Domain name
        **kwargs: Other parameters
        
    Returns:
        NativeEvalResult
    """
    evaluator = NativeEvaluatorRegistry.get_evaluator(benchmark, **kwargs)
    return evaluator.evaluate(trace, task, domain)
