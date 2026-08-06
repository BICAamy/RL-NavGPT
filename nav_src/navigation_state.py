"""Shared navigation-state and policy-prompt construction.

The inference agent and the Gymnasium RL environment both use this module so
that a cached action plan, panoramic observation, candidate list, and history
are rendered by one implementation.
"""

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from prompt.chat_prompt import build_chat_messages
from prompt.planner_prompt import (
    BACK_TRACE_TOOL_DESCRIPTION,
    BACK_TRACE_TOOL_NAME,
    MAKE_ACTION_TOOL_DESCRIPTION,
    MAKE_ACTION_TOOL_NAME,
    get_prompt_set,
)


@dataclass(frozen=True)
class NavigationPromptConfig:
    """Formatting choices that define the policy state."""

    use_relative_angle: bool = True
    use_navigable: bool = False
    use_single_action: bool = True
    max_scratchpad_length: int = 7000

    @classmethod
    def from_namespace(cls, config: Any) -> "NavigationPromptConfig":
        return cls(
            use_relative_angle=bool(config.use_relative_angle),
            use_navigable=bool(config.use_navigable),
            use_single_action=bool(config.use_single_action),
            max_scratchpad_length=int(config.max_scratchpad_length),
        )


@dataclass(frozen=True)
class PromptTraceStep:
    """One prior model decision and the resulting environment text."""

    model_output: str
    action_name: str
    history: str
    observation: str


def build_policy_scratchpad(
    trace: Sequence[PromptTraceStep],
    max_length: int,
) -> str:
    """Build the exact decision/observation trace used by inference and RL."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    scratchpad = ""
    last_index = len(trace) - 1
    for index, step in enumerate(trace):
        scratchpad += step.model_output
        if index != last_index and step.action_name == MAKE_ACTION_TOOL_NAME:
            environment_text = step.history
        else:
            environment_text = step.observation
        scratchpad += f"\nObservation: {environment_text}\nDecision:"
    return scratchpad[-max_length:]


def normalize_angle(angle: float) -> float:
    while angle > 180:
        angle -= 360
    while angle <= -180:
        angle += 360
    return angle


def angle_to_left_right(angle: float) -> str:
    return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"


def describe_turn(previous_heading: float, current_heading: float) -> str:
    """Describe a heading change using the same wording as inference."""

    turned_angle = current_heading - previous_heading
    previous = angle_to_left_right(normalize_angle(previous_heading))
    current = angle_to_left_right(normalize_angle(current_heading))
    return (
        f"Turn heading direction {turned_angle:.2f} degrees from "
        f"{previous} to {current}."
    )


class NavigationStateBuilder:
    """Render observations, history, scratchpad, and complete policy prompts."""

    def __init__(self, config: NavigationPromptConfig):
        if config.max_scratchpad_length <= 0:
            raise ValueError("max_scratchpad_length must be positive")
        self.config = config
        self.prompt_set = get_prompt_set("plain")

    def get_navigable_str(
        self,
        cur_heading: float,
        cur_elevation: float,
        navigable: Mapping[str, Mapping[str, Any]],
    ) -> str:
        navigable_str = ""
        for viewpoint_id, item in navigable.items():
            heading = np.rad2deg(item["heading"])
            elevation = np.rad2deg(item["elevation"])
            distance = item["distance"]
            if self.config.use_relative_angle:
                heading -= cur_heading
                elevation -= cur_elevation
            navigable_str += (
                f"'{viewpoint_id}':\nheading: {heading:.2f}, "
                f"elevation: {elevation:.2f}, distance: {distance:.2f}\n"
            )
        return navigable_str

    def modify_heading_angles(
        self,
        heading_angle: float,
        observation_list: Sequence[str],
        candidate_dict: Mapping[str, Mapping[str, Any]],
        object_list: Sequence[Mapping[str, Mapping[str, Any]]],
    ) -> str:
        """Render the eight panoramic sectors relative to the current heading."""

        if len(observation_list) != 8 or len(object_list) != 8:
            raise ValueError("Panoramic observations and objects must have 8 sectors")

        directions = [
            "Front",
            "Front Right",
            "Right",
            "Rear Right",
            "Rear",
            "Rear Left",
            "Left",
            "Front Left",
        ]
        range_index = int((heading_angle - 22.5) // 45) + 1
        observation_indices = [(index + range_index) % 8 for index in range(8)]

        candidate_ranges: Dict[int, Dict[str, str]] = {}
        if not self.config.use_navigable:
            for viewpoint_id, viewpoint_data in candidate_dict.items():
                viewpoint_heading = np.rad2deg(viewpoint_data["heading"])
                viewpoint_range = int((viewpoint_heading - 22.5) // 45) + 1
                relative_heading = angle_to_left_right(
                    normalize_angle(viewpoint_heading - heading_angle)
                )
                description = (
                    f'{relative_heading}, {viewpoint_data["distance"]:.2f}m'
                )
                candidate_ranges.setdefault(viewpoint_range, {})[viewpoint_id] = (
                    description
                )

        angle_ranges = [
            (angle - 22.5 - heading_angle, angle + 22.5 - heading_angle)
            for angle in range(0, 360, 45)
        ]
        formatted = []
        for direction, index in zip(directions, observation_indices):
            left = angle_to_left_right(normalize_angle(angle_ranges[index][0]))
            right = angle_to_left_right(normalize_angle(angle_ranges[index][1]))
            section = (
                f"{direction}, range ({left} to {right}): \n"
                f"'{observation_list[index]}'"
            )

            objects: Dict[str, str] = {}
            for name, object_data in object_list[index].items():
                relative_heading = angle_to_left_right(
                    normalize_angle(object_data["heading"] - heading_angle)
                )
                objects[name] = (
                    f'{relative_heading}, {object_data["distance"]:.2f}m'
                )
            section += (
                f"\n{direction} Objects in 3m: {objects}"
                if objects
                else f"\n{direction} Objects in 3m: None"
            )

            candidates = candidate_ranges.get(index)
            section += (
                f"\n{direction} Navigable Viewpoints:{candidates}"
                if candidates
                else f"\n{direction} Navigable Viewpoints: None"
            )
            formatted.append(section)
        return "\n".join(formatted)

    def format_feature(self, observation: Mapping[str, Any]) -> str:
        feature = observation["obs"]
        if not self.config.use_relative_angle:
            return feature
        heading = float(np.rad2deg(observation["heading"]))
        return self.modify_heading_angles(
            heading,
            feature,
            observation["candidate"],
            observation["objects"],
        )

    def format_initial_observation(self, observation: Mapping[str, Any]) -> str:
        feature = self.format_feature(observation)
        heading = float(np.rad2deg(observation["heading"]))
        elevation = float(np.rad2deg(observation["elevation"]))
        orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"

        navigable = observation["candidate"]
        if self.config.use_navigable:
            navigable = self.get_navigable_str(heading, elevation, navigable)

        if self.config.use_relative_angle:
            output = f"\n\tCurrent Viewpoint:\n{feature}"
        else:
            output = (
                f"\n\tCurrent Orientation:\n{orientation}"
                f"\n\tCurrent Viewpoint:\n{feature}"
            )
        if self.config.use_navigable:
            output += f"\n\tNavigable Viewpoints:\n{navigable}"
        return output

    def format_tool_observation(
        self,
        observation: Mapping[str, Any],
        viewpoint_id: str,
    ) -> str:
        feature = self.format_feature(observation)
        heading = float(np.rad2deg(observation["heading"]))
        elevation = float(np.rad2deg(observation["elevation"]))
        orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
        navigable = observation["candidate"]
        if self.config.use_navigable:
            navigable = self.get_navigable_str(heading, elevation, navigable)

        if self.config.use_relative_angle:
            if self.config.use_navigable:
                return (
                    f"\n\tCurrent Viewpoint:\n{feature}"
                    f"\n\tNavigable Viewpoints:\n{navigable}"
                )
            return f'\nCurrent Viewpoint "{viewpoint_id}":\n{feature}'

        output = (
            f"\n\tCurrent Orientation:\n{orientation}"
            f"\n\tCurrent Viewpoint:\n{feature}"
        )
        if self.config.use_navigable:
            output += f"\n\tNavigable Viewpoints:\n{navigable}"
        return output

    def format_invalid_observation(
        self,
        observation: Mapping[str, Any],
        message: str,
    ) -> str:
        valid_ids = list(observation["candidate"].keys())
        current = self.format_tool_observation(
            observation,
            str(observation["viewpoint"]),
        )
        return (
            f"\nInvalid navigation decision: {message} Agent not moved. "
            "Do not fabricate viewpoint IDs. Current adjacent candidates are "
            f"{valid_ids}.{current}"
        )

    @staticmethod
    def initial_history(observation: Mapping[str, Any]) -> str:
        return (
            "Navigation start, no actions taken yet.\n"
            f'Current viewpoint "{observation["viewpoint"]}": Scene from the '
            f'viewpoint is a {observation["obs_summary"]}'
        )

    @staticmethod
    def history_after_move(observation: Mapping[str, Any], action: str) -> str:
        return (
            f"{action}\nCurrent viewpoint \"{observation['viewpoint']}\": "
            f"Scene from the viewpoint is a {observation['obs_summary']}"
        )

    def build_scratchpad(self, trace: Sequence[PromptTraceStep]) -> str:
        return build_policy_scratchpad(
            trace,
            self.config.max_scratchpad_length,
        )

    def build_policy_prompt(
        self,
        action_plan: str,
        initial_observation: str,
        trace: Sequence[PromptTraceStep],
    ) -> str:
        if self.config.use_single_action:
            tool_names = [MAKE_ACTION_TOOL_NAME]
            tool_descriptions = [MAKE_ACTION_TOOL_DESCRIPTION]
            template = self.prompt_set["vln_gpt35"]
        else:
            tool_names = [MAKE_ACTION_TOOL_NAME, BACK_TRACE_TOOL_NAME]
            tool_descriptions = [
                MAKE_ACTION_TOOL_DESCRIPTION,
                BACK_TRACE_TOOL_DESCRIPTION,
            ]
            template = self.prompt_set["vln_orchestrator"]

        return template.format(
            action_plan=action_plan,
            init_observation=initial_observation,
            agent_scratchpad=self.build_scratchpad(trace),
            tool_names=", ".join(tool_names),
            tool_descriptions="\n".join(
                f"{name}: {description}"
                for name, description in zip(tool_names, tool_descriptions)
            ),
        )

    @staticmethod
    def build_chat_messages(policy_prompt: str) -> List[Dict[str, str]]:
        return build_chat_messages(policy_prompt)

    @staticmethod
    def prompt_sha256(policy_prompt: str) -> str:
        return hashlib.sha256(policy_prompt.encode("utf-8")).hexdigest()
