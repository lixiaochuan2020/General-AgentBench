from typing import Callable

from loguru import logger

from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_base import EvaluatorBase


class EnvironmentEvaluator(EvaluatorBase):
    """
    Evaluator focuses on endstate of the simulation environment.
    """

    @classmethod
    def calculate_reward(
        cls,
        environment_constructor: Callable[[], Environment],
        task: Task,
        full_trajectory: list[
            Message
        ],  # FIXME: It would be better to be able to get only the messages that are after the initial state
        solo_mode: bool = False,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation.
        Args:
            environment_constructor: Callable[[], Environment]
            task: Task
            full_trajectory: list[Message] (Must include the message history from task initial state)
            solo_mode: bool
        Returns:
            RewardInfo
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"note": "No evaluation criteria"},
            )
        expected_actions = task.evaluation_criteria.actions
        env_assertions = task.evaluation_criteria.env_assertions
        if expected_actions is None and env_assertions is None:
            return RewardInfo(
                reward=1.0,
                db_check=DBCheck(db_match=True, db_reward=1.0),
                info={"note": "No expected actions or env assertions"},
            )

        initialization_data = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_data is not None
        ):
            initialization_data = task.initial_state.initialization_data

        initialization_actions = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_actions is not None
        ):
            initialization_actions = task.initial_state.initialization_actions

        message_history = []
        if (
            task.initial_state is not None
            and task.initial_state.message_history is not None
        ):
            message_history = task.initial_state.message_history

        # === DEBUG LOGGING ===
        logger.info("[ENV_EVAL DEBUG] EnvironmentEvaluator.calculate_reward called")
        logger.info(f"[ENV_EVAL DEBUG] task.id = {task.id}")
        logger.info(f"[ENV_EVAL DEBUG] full_trajectory messages = {len(full_trajectory)}")
        logger.info(f"[ENV_EVAL DEBUG] initialization_actions = {initialization_actions}")
        logger.info(f"[ENV_EVAL DEBUG] message_history (initial) = {len(message_history)} messages")
        for i, msg in enumerate(full_trajectory):
            msg_type = type(msg).__name__
            has_tc = hasattr(msg, 'tool_calls') and msg.tool_calls
            requestor = getattr(msg, 'requestor', None)
            tc_info = ""
            if has_tc:
                tc_names = [tc.name for tc in msg.tool_calls]
                tc_requestors = [tc.requestor for tc in msg.tool_calls]
                tc_info = f", tool_calls={tc_names}, requestors={tc_requestors}"
            logger.info(f"[ENV_EVAL DEBUG]   trajectory[{i}]: {msg_type}, requestor={requestor}{tc_info}")
        logger.info("[ENV_EVAL DEBUG] Now calling set_state on predicted_environment...")
        # === END DEBUG ===

        predicted_environment = environment_constructor(solo_mode=solo_mode)
        predicted_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=full_trajectory,
        )
        # Debug logging: predicted environment setup
        try:
            logger.info(
                "[ENV EVAL] Predicted env set_state completed | init_data=%s | init_actions=%s | trajectory_len=%d",
                bool(initialization_data),
                len(initialization_actions) if initialization_actions else 0,
                len(full_trajectory),
            )
        except Exception as _:
            pass
        predicted_tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if (
                isinstance(message, AssistantMessage)
                or isinstance(message, UserMessage)
            ) and message.is_tool_call():
                predicted_tool_calls.extend(message.tool_calls)

        # Debug logging: predicted tool calls
        try:
            logger.info(
                "[ENV EVAL] Predicted tool calls count=%d | ids=%s",
                len(predicted_tool_calls),
                [tc.id for tc in predicted_tool_calls],
            )
        except Exception as _:
            pass

        # Setting up gold environment
        gold_environment = environment_constructor()
        gold_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        try:
            logger.info(
                "[ENV EVAL] Gold env set_state completed | msg_history_len=%d",
                len(message_history),
            )
        except Exception as _:
            pass
        golden_actions = task.evaluation_criteria.actions or []
        for action in golden_actions:
            try:
                gold_environment.make_tool_call(
                    tool_name=action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
            except Exception as e:
                logger.warning(
                    f"Error in golden actions {action.name}({action.arguments}): {e}"
                )

        # Comparing the environments
        agent_db_hash = gold_environment.get_db_hash()
        user_db_hash = gold_environment.get_user_db_hash()
        predicted_agent_db_hash = predicted_environment.get_db_hash()
        predicted_user_db_hash = predicted_environment.get_user_db_hash()
        agent_db_match = agent_db_hash == predicted_agent_db_hash
        user_db_match = user_db_hash == predicted_user_db_hash
        if agent_db_match and user_db_match:
            db_reward = 1.0
            db_match = True
        else:
            db_reward = 0.0
            db_match = False

        db_check = DBCheck(db_match=db_match, db_reward=db_reward)

        try:
            logger.info(
                "[ENV EVAL] DB compare | agent_match=%s user_match=%s | agent_hashes=(gold=%s, pred=%s) | user_hashes=(gold=%s, pred=%s) | db_reward=%.2f",
                agent_db_match,
                user_db_match,
                agent_db_hash,
                predicted_agent_db_hash,
                user_db_hash,
                predicted_user_db_hash,
                db_reward,
            )
        except Exception as _:
            pass

        # Run env assertions
        env_assertions = task.evaluation_criteria.env_assertions or []
        env_assertion_checks = []
        env_assertion_reward = 1.0
        for env_assertion in env_assertions:
            success = predicted_environment.run_env_assertion(
                env_assertion,
                raise_assertion_error=False,
            )
            res = EnvAssertionCheck(
                env_assertion=env_assertion,
                met=success,
                reward=1.0 if success else 0.0,
            )
            env_assertion_checks.append(res)
            env_assertion_reward *= res.reward

        try:
            logger.info(
                "[ENV EVAL] Env assertions run | count=%d | checks=%s | reward=%.2f",
                len(env_assertions),
                [(c.env_assertion, c.met, c.reward) for c in env_assertion_checks],
                env_assertion_reward,
            )
        except Exception as _:
            pass

        reward = 1.0
        reward_breakdown = {}
        if RewardType.DB in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.DB] = db_reward
            reward *= db_reward
        if RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.ENV_ASSERTION] = env_assertion_reward
            reward *= env_assertion_reward

        try:
            logger.info(
                "[ENV EVAL] Final reward=%.2f | breakdown=%s | basis=%s",
                reward,
                reward_breakdown,
                task.evaluation_criteria.reward_basis,
            )
        except Exception as _:
            pass

        return RewardInfo(
            reward=reward,
            db_check=db_check,
            env_assertions=env_assertion_checks,
            reward_basis=task.evaluation_criteria.reward_basis,
            reward_breakdown=reward_breakdown,
        )
