"""Shared high-level Planner construction and action-plan normalization."""

import re
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from prompt.planner_prompt import PLANNER_PROMPT


_NUMBERED_STEP_RE = re.compile(r"(?m)^\s*\d+[.)]\s+\S")
_ACTION_PLAN_HEADER_RE = re.compile(r"(?i)^action\s+plan\s*:\s*")


class ActionPlanFormatError(ValueError):
    """Raised when the Planner does not return a usable numbered plan."""


def build_planner_chain(llm: Any) -> LLMChain:
    """Build the same Planner chain used by online and cached navigation."""

    prompt = PromptTemplate(
        template=PLANNER_PROMPT,
        input_variables=["instruction"],
    )
    return LLMChain(llm=llm, prompt=prompt)


def normalize_action_plan(text: str) -> str:
    """Validate and canonicalize a Planner response without changing its steps."""

    plan = text.strip()
    if plan.startswith("```") and plan.endswith("```"):
        lines = plan.splitlines()
        if len(lines) >= 3:
            plan = "\n".join(lines[1:-1]).strip()

    if not plan:
        raise ActionPlanFormatError("Planner returned an empty action plan")
    if not _NUMBERED_STEP_RE.search(plan):
        raise ActionPlanFormatError(
            "Planner output must contain at least one numbered action step"
        )

    if _ACTION_PLAN_HEADER_RE.match(plan):
        plan = _ACTION_PLAN_HEADER_RE.sub("", plan, count=1).lstrip()

    return f"Action plan:\n{plan}"


def generate_action_plan(plan_chain: LLMChain, instruction: str) -> str:
    """Generate the canonical action plan for one raw instruction."""

    output = plan_chain.run(instruction=instruction)
    return normalize_action_plan(output)
