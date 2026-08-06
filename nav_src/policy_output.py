"""Canonical policy output protocol shared by inference and RL training."""

from dataclasses import dataclass
import re
from typing import Optional


MAKE_ACTION_NAME = "action_maker"
BACK_TRACE_NAME = "back_tracer"
FINISH_ACTION = "Finish!"

_THINK_PATTERN = re.compile(
    r"<Think>\s*(?P<thought>.*?)\s*</Think>",
    flags=re.DOTALL,
)
_ACTION_PATTERN = re.compile(
    r"<Action>\s*(?P<action>.*?)\s*</Action>",
    flags=re.DOTALL,
)
_TOOL_ACTION_PATTERN = re.compile(
    r"(?P<tool>action_maker|back_tracer)\s*\(\s*"
    r"(?P<quote>[\"'])(?P<viewpoint>[a-f0-9]{32})(?P=quote)\s*\)",
)
_FINISH_PATTERN = re.compile(r"Finish!")


class PolicyOutputParseError(ValueError):
    """Raised when model output does not follow the Think/Action protocol."""


@dataclass(frozen=True)
class PolicyOutput:
    """Normalized navigation decision emitted by any LLM backend."""

    thought: str
    action_type: str
    action_name: str
    viewpoint_id: Optional[str] = None

    @property
    def is_finish(self) -> bool:
        return self.action_type == "finish"


def parse_policy_output(text: str) -> PolicyOutput:
    """Parse one strict ``<Think>`` + ``<Action>`` navigation decision.

    Canonical move/backtrack examples::

        <Think>I should enter the hallway.</Think>
        <Action>action_maker("0123456789abcdef0123456789abcdef")</Action>

        <Think>I should return to the previous junction.</Think>
        <Action>back_tracer("0123456789abcdef0123456789abcdef")</Action>

    Canonical stop example::

        <Think>I have reached the destination.</Think>
        <Action>Finish!</Action>
    """

    if not isinstance(text, str) or not text.strip():
        raise PolicyOutputParseError("Policy output must be a non-empty string.")

    think_matches = list(_THINK_PATTERN.finditer(text))
    action_matches = list(_ACTION_PATTERN.finditer(text))
    if len(think_matches) != 1 or len(action_matches) != 1:
        raise PolicyOutputParseError(
            "Expected exactly one <Think>...</Think> block and one "
            "<Action>...</Action> block."
        )

    think_match = think_matches[0]
    action_match = action_matches[0]
    if think_match.start() > action_match.start():
        raise PolicyOutputParseError("<Think> must appear before <Action>.")

    remainder = (
        text[:think_match.start()]
        + text[think_match.end():action_match.start()]
        + text[action_match.end():]
    )
    if remainder.strip():
        raise PolicyOutputParseError(
            "Do not emit text outside the <Think> and <Action> blocks."
        )

    thought = think_match.group("thought").strip()
    if not thought:
        raise PolicyOutputParseError("The <Think> block must not be empty.")

    action_text = action_match.group("action").strip()
    if _FINISH_PATTERN.fullmatch(action_text):
        return PolicyOutput(
            thought=thought,
            action_type="finish",
            action_name=FINISH_ACTION,
        )

    tool_match = _TOOL_ACTION_PATTERN.fullmatch(action_text)
    if tool_match is None:
        raise PolicyOutputParseError(
            "Action must be action_maker(\"<32-char viewpoint id>\"), "
            "back_tracer(\"<32-char viewpoint id>\"), or Finish!."
        )

    action_name = tool_match.group("tool")
    action_type = "move" if action_name == MAKE_ACTION_NAME else "backtrack"
    return PolicyOutput(
        thought=thought,
        action_type=action_type,
        action_name=action_name,
        viewpoint_id=tool_match.group("viewpoint"),
    )


def format_move_output(thought: str, viewpoint_id: str) -> str:
    """Format a canonical adjacent-viewpoint action."""

    return (
        f"<Think>{thought.strip()}</Think>\n"
        f'<Action>{MAKE_ACTION_NAME}("{viewpoint_id}")</Action>'
    )


def format_backtrack_output(thought: str, viewpoint_id: str) -> str:
    """Format a canonical return-to-visited-viewpoint action."""

    return (
        f"<Think>{thought.strip()}</Think>\n"
        f'<Action>{BACK_TRACE_NAME}("{viewpoint_id}")</Action>'
    )


def format_finish_output(thought: str) -> str:
    """Format a canonical stop action."""

    return f"<Think>{thought.strip()}</Think>\n<Action>{FINISH_ACTION}</Action>"
