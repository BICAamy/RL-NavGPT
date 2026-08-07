"""Validate stage-three RL environment mechanics without loading an LLM."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import random
import sys

from gymnasium.utils.env_checker import check_env


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from action_plan_cache import attach_action_plans  # noqa: E402
from data_utils import construct_instrs  # noqa: E402
from env import ERROR_MARGIN, R2RNavBatch  # noqa: E402
from navigation_state import NavigationPromptConfig  # noqa: E402
from policy_output import (  # noqa: E402
    format_backtrack_output,
    format_finish_output,
    format_move_output,
)
from rl_env import (  # noqa: E402
    NavGPTEnvironmentFactory,
    NavGPTGymEnv,
    NavGPTTRLEnvironment,
    trl_environment_reward,
)
from utils.data import ImageObservationsDB  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Gymnasium R2R wrapper without an LLM"
    )
    parser.add_argument("--root-dir", default="../datasets")
    parser.add_argument("--split", default="R2R_train_enc")
    parser.add_argument("--action-plan-cache", required=True)
    parser.add_argument("--instr-id", default="")
    parser.add_argument("--max-steps", type=int, default=20)
    return parser


class TransitionRecorder:
    def __init__(self):
        self.transitions = []

    def __call__(self, transition):
        self.transitions.append(transition)
        return {"stage3_placeholder": 0.0}


class VisualFeatureRecorder:
    def __init__(self):
        self.calls = 0

    def __call__(self, observation):
        self.calls += 1
        return (
            str(observation["scan"]),
            str(observation["viewpoint"]),
            float(observation["heading"]),
            float(observation["elevation"]),
        )


class ConstantReward:
    def __call__(self, transition):
        return {"validation_reward": 1.25}


def choose_item(items, instr_id, max_steps):
    if instr_id:
        for item in items:
            if str(item["instr_id"]) == str(instr_id):
                if len(item["path"]) > max_steps:
                    raise ValueError(
                        "Selected path needs "
                        f'{len(item["path"])} actions including Finish, but '
                        f"--max-steps is {max_steps}"
                    )
                return item
        raise KeyError(f"instr_id not found in split: {instr_id}")
    for item in items:
        if (
            1 < len(item["path"]) <= max_steps
            and float(item.get("distance", 0)) > ERROR_MARGIN
        ):
            return item
    raise ValueError("No non-trivial navigation item found")


def build_base_env(
    root_dir: Path,
    split: str,
    cache_path: str,
    instr_id: str,
    max_steps: int,
):
    annotation_dir = root_dir / "R2R" / "annotations"
    items = construct_instrs(str(annotation_dir), "r2r", [split])
    selected = choose_item(items, instr_id, max_steps)
    selected = attach_action_plans([selected], cache_path)[0]

    observation_db = ImageObservationsDB(
        str(root_dir / "R2R" / "observations_list_summarized"),
        str(root_dir / "R2R" / "observations_summarized"),
        str(root_dir / "R2R" / "objects_list"),
    )
    base_env = R2RNavBatch(
        observation_db,
        [selected],
        str(root_dir / "R2R" / "connectivity"),
        str(root_dir / "R2R" / "navigable"),
        batch_size=1,
        seed=0,
        name=f"{split}_rl_smoke",
    )
    return base_env, selected, observation_db


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps < 2:
        raise ValueError("--max-steps must be at least 2")

    root_dir = Path(args.root_dir)
    base_env, item, observation_db = build_base_env(
        root_dir,
        args.split,
        args.action_plan_cache,
        args.instr_id,
        args.max_steps,
    )
    instr_id = str(item["instr_id"])
    transition_recorder = TransitionRecorder()
    visual_feature_recorder = VisualFeatureRecorder()
    single_action_env = NavGPTGymEnv(
        base_env,
        prompt_config=NavigationPromptConfig(use_single_action=True),
        max_steps=args.max_steps,
        reward_calculator=transition_recorder,
        visual_feature_provider=visual_feature_recorder,
    )

    prompt_a, info_a = single_action_env.reset(options={"instr_id": instr_id})
    prompt_b, info_b = single_action_env.reset(options={"instr_id": instr_id})
    require(prompt_a == prompt_b, "Repeated reset changed the initial prompt")
    require(
        info_a["viewpoint_id"] == info_b["viewpoint_id"],
        "Repeated reset changed the start viewpoint",
    )
    require(
        info_a["action_plan"] == info_b["action_plan"],
        "Repeated reset changed the cached action plan",
    )
    require(
        info_a["chat_messages"][1]["content"] == prompt_a,
        "Chat messages do not contain the exact policy prompt",
    )
    require(
        single_action_env.observation_space.contains(prompt_a),
        "Initial prompt is outside observation_space",
    )
    require(
        info_a["visual_feature"][1] == info_a["viewpoint_id"],
        "Visual feature provider did not receive the initial observation",
    )
    initial_pose = (
        info_a["viewpoint_id"],
        info_a["heading"],
        info_a["elevation"],
    )

    first_target = str(item["path"][1])
    _, _, terminated, truncated, move_info = single_action_env.step(
        format_move_output("Follow the first route segment.", first_target)
    )
    require(move_info["action_valid"], "Ground-truth adjacent move was rejected")
    require(move_info["moved"], "Valid move did not change viewpoint")
    require(move_info["viewpoint_id"] == first_target, "Moved to wrong viewpoint")
    require(not terminated and not truncated, "Valid move ended episode early")
    require(
        transition_recorder.transitions[-1].previous_visual_feature[1]
        == initial_pose[0],
        "Transition lost the previous visual feature",
    )
    require(
        transition_recorder.transitions[-1].current_visual_feature[1]
        == first_target,
        "Transition lost the current visual feature",
    )

    invalid_target = next(
        character * 32
        for character in "0123456789abcdef"
        if character * 32 not in move_info["candidate_viewpoint_ids"]
    )
    before_invalid = move_info["viewpoint_id"]
    _, _, terminated, truncated, invalid_info = single_action_env.step(
        format_move_output("Try an invalid candidate for validation.", invalid_target)
    )
    require(not invalid_info["action_valid"], "Invalid viewpoint was accepted")
    require(not invalid_info["moved"], "Invalid viewpoint moved the simulator")
    require(
        invalid_info["viewpoint_id"] == before_invalid,
        "Invalid action polluted simulator state",
    )
    require(not terminated and not truncated, "Recoverable invalid action ended episode")

    _, _, terminated, truncated, finish_info = single_action_env.step(
        format_finish_output("Stop early to validate termination semantics.")
    )
    require(terminated and not truncated, "Finish did not terminate the episode")
    require(not finish_info["success"], "Premature Finish was marked successful")
    require(
        finish_info["termination_reason"] == "premature_finish",
        "Wrong premature-Finish reason",
    )

    reset_prompt, reset_info = single_action_env.reset(
        options={"instr_id": instr_id}
    )
    require(reset_prompt == prompt_a, "Reset after termination changed prompt")
    require(
        (
            reset_info["viewpoint_id"],
            reset_info["heading"],
            reset_info["elevation"],
        )
        == initial_pose,
        "Reset after termination did not restore the initial pose",
    )
    for index, target in enumerate(item["path"][1:], start=1):
        _, _, terminated, truncated, path_info = single_action_env.step(
            format_move_output(f"Follow ground-truth edge {index}.", str(target))
        )
        require(path_info["action_valid"], f"Path edge {index} was rejected")
        require(not terminated and not truncated, f"Path edge {index} ended episode")
    _, _, terminated, truncated, success_info = single_action_env.step(
        format_finish_output("The annotated destination has been reached.")
    )
    require(terminated and not truncated, "Successful Finish did not terminate")
    require(success_info["success"], "Goal Finish was not marked successful")
    require(success_info["distance_to_goal"] < ERROR_MARGIN, "Goal is outside margin")
    json.dumps(single_action_env.trajectory)

    multi_action_env = NavGPTGymEnv(
        base_env,
        prompt_config=NavigationPromptConfig(use_single_action=False),
        max_steps=args.max_steps,
    )
    _, start_info = multi_action_env.reset(options={"instr_id": instr_id})
    start_viewpoint = start_info["viewpoint_id"]
    _, _, _, _, moved_info = multi_action_env.step(
        format_move_output("Move once before testing backtracking.", first_target)
    )
    require(moved_info["viewpoint_id"] == first_target, "Backtrack setup failed")
    _, _, terminated, truncated, backtrack_info = multi_action_env.step(
        format_backtrack_output("Return to the visited start.", start_viewpoint)
    )
    require(backtrack_info["action_valid"], "Valid backtrack was rejected")
    require(backtrack_info["revisited"], "Backtrack was not marked revisited")
    require(
        backtrack_info["viewpoint_id"] == start_viewpoint,
        "Backtrack did not return to start",
    )
    require(not terminated and not truncated, "Backtrack ended episode unexpectedly")

    timeout_env = NavGPTGymEnv(base_env, max_steps=1)
    timeout_env.reset(options={"instr_id": instr_id})
    try:
        timeout_env.step("x" * (timeout_env.action_space.max_length + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Oversized policy output was accepted")
    require(
        timeout_env.render().find("steps=0/1") >= 0,
        "Rejected oversized output consumed an environment step",
    )
    _, _, terminated, truncated, timeout_info = timeout_env.step(
        format_move_output("Exercise the step limit.", invalid_target)
    )
    require(not terminated and truncated, "Step limit did not set truncated=True")
    require(timeout_info["termination_reason"] == "max_steps", "Wrong timeout reason")

    reward_env = NavGPTGymEnv(
        base_env,
        max_steps=2,
        reward_calculator=ConstantReward(),
    )
    reward_env.reset(options={"instr_id": instr_id})
    _, first_reward, _, _, first_reward_info = reward_env.step(
        format_move_output("Validate reward accumulation.", invalid_target)
    )
    _, second_reward, _, truncated, second_reward_info = reward_env.step(
        format_move_output("Validate reward accumulation again.", invalid_target)
    )
    require(first_reward == second_reward == 1.25, "Wrong per-step reward")
    require(truncated, "Reward validation episode did not truncate")
    require(
        first_reward_info["reward_components"]["validation_reward"] == 1.25,
        "Reward component missing from info",
    )
    require(
        second_reward_info["episode_return"] == reward_env.get_reward() == 2.5,
        "get_reward() did not return the accumulated episode reward",
    )

    check_env(
        NavGPTGymEnv(base_env, max_steps=2),
        skip_render_check=True,
    )

    global_random_state = random.getstate()
    factory = NavGPTEnvironmentFactory(
        view_db=observation_db,
        instr_data=[item],
        connectivity_dir=str(root_dir / "R2R" / "connectivity"),
        navigable_dir=str(root_dir / "R2R" / "navigable"),
        max_steps=args.max_steps,
    )
    rollout_group = factory.create_group(instr_id, num_rollouts=2)
    require(
        random.getstate() == global_random_state,
        "Creating rollout environments changed Python's global RNG state",
    )
    require(
        rollout_group[0].base_env.env.sims[0]
        is not rollout_group[1].base_env.env.sims[0],
        "Rollout group shares a mutable simulator",
    )
    require(
        rollout_group[0].base_env.graphs[item["scan"]]
        is rollout_group[1].base_env.graphs[item["scan"]],
        "Rollout group did not share the read-only navigation graph cache",
    )
    require(
        rollout_group[0].reward_calculator
        is not rollout_group[1].reward_calculator,
        "Rollout group shares mutable reward state",
    )
    require(
        rollout_group[0].render() == rollout_group[1].render(),
        "Rollout group did not start from identical state",
    )

    trl_environment = factory.as_trl_factory()()
    require(
        isinstance(trl_environment, NavGPTTRLEnvironment),
        "TRL factory returned the wrong adapter type",
    )
    public_methods = {
        name
        for name, member in inspect.getmembers(
            trl_environment,
            predicate=inspect.ismethod,
        )
        if not name.startswith("_")
    }
    require(
        public_methods
        == {"reset", "submit_navigation_decision"},
        f"Unexpected TRL-exposed methods: {sorted(public_methods)}",
    )
    trl_tool_methods = public_methods.difference({"reset"})
    require(
        trl_tool_methods == {"submit_navigation_decision"},
        f"TRL would expose unexpected tools: {sorted(trl_tool_methods)}",
    )
    trl_prompt = trl_environment.reset(instr_id=instr_id)
    require(prompt_a in trl_prompt, "TRL reset dropped the policy prompt")
    require(
        "Call `submit_navigation_decision` exactly once" in trl_prompt,
        "TRL reset omitted the native tool protocol",
    )
    tool_observation = trl_environment.submit_navigation_decision(
        format_move_output("Validate the TRL adapter.", invalid_target)
    )
    require(
        trl_environment.last_info["action_valid"] is False,
        "TRL adapter bypassed canonical action validation",
    )
    require(
        "not an adjacent candidate" in tool_observation,
        "TRL tool did not return the resulting environment observation",
    )
    require(
        trl_environment_reward([trl_environment]) == [0.0],
        "TRL reward function returned the wrong stage-three reward",
    )

    print(f"PASS instr_id={instr_id}")
    print("- deterministic reset and cached action plan")
    print("- adjacent move and invalid-action no-op")
    print("- premature Finish and successful Finish semantics")
    print("- optional backtrack to a visited viewpoint")
    print("- max-step truncation and serializable trajectory records")
    print("- reward components and episode-return accumulation")
    print("- Gymnasium API checked by gymnasium.utils.env_checker")
    print("- isolated rollout simulators, reward state, and shared graph cache")
    print("- identical group state without perturbing Python's global RNG")
    print("- TRL-native adapter with one tool and explicit reward function")


if __name__ == "__main__":
    main()
