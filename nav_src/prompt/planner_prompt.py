PLANNER_PROMPT = """Given the long instruction: {instruction}

Divide the long instruction into action steps with detailed descriptions in the following format:
Action plan:
1. action_step_1
2. action_step_2
...

Action plan:"""

ACTION_PROMPT = """You are an agent following an action plan to navigation in indoor environment.

Action plan: {action_plan}

You are currently at one of the steps in the plan. You will be given the history of previous steps you have taken, the current observation of the environment, and the navigable viewpoints for the next step.

You should:
1) evaluate the history and observation to decide which step of action plan you are at.
2) choose one viewpoint from the navigable viewpoints.

Each navigable viewpoint has a unique 32-character ID.

----
Output exactly these two blocks and no other text:

<Think>In at most 60 words, state the current subgoal, visible evidence, and why the selected action is consistent.</Think>
<Action>action_maker("32-character viewpoint ID")</Action>
----

Begin!

History: {history}
Observation: {observation}
Navigable viewpoints: {navigable_viewpoints}
Decision:"""

HISTORY_PROMPT = """You are an agent navigating in indoor environment.

You have reached a new viewpoint after taking previous action. You will be given the navigation history, the current observation of the environment, and the previous action you taken.

You should:
1) evaluate the new observation and history.
2) update the history with the previous action and the new observation.

History: {history}
Previous action: {previous_action}
Observation: {observation}
Update history with the new observation:"""

MAKE_ACTION_TOOL_NAME = "action_maker"
MAKE_ACTION_TOOL_DESCRIPTION = (
    "Move to one adjacent viewpoint. The argument must be an exact "
    "32-character viewpoint ID from the current navigable candidates. "
    'Canonical output: <Action>action_maker('
    '"4a153b13a3f6424784cb8e5dabbb3a2c")</Action>.'
)

BACK_TRACE_PROMPT = """You are an agent following an action plan to navigation in indoor environment.

You are currently at an intermediate step of the trajectory but seems going off the track. You will be given the action plan describing the whole trajectory, the history of previous steps you have taken, the observations of the viewpoints along the trajectory.

You should evaluate the history, the action plan and the observations along the way to decide the viewpoints to go back to.

Each previous viewpoint has a unique 32-character ID.
You must choose one from the navigable viewpoints, DO NOT answer None of the above.

----
Output exactly these two blocks and no other text:

<Think>In at most 60 words, explain why backtracking is needed and which earlier viewpoint should be revisited.</Think>
<Action>back_tracer("32-character viewpoint ID")</Action>
----

Begin!

Action plan: {action_plan}
History: {history}
Observation: {observation}
Decision:"""

BACK_TRACE_TOOL_NAME = "back_tracer"
BACK_TRACE_TOOL_DESCRIPTION = (
    "Move to a previous viewpoint on the trajectory when recovery is needed. "
    "The argument must be an exact 32-character viewpoint ID from the history. "
    'Canonical output: <Action>back_tracer('
    '"4a153b13a3f6424784cb8e5dabbb3a2c")</Action>.'
)


_MULTI_TOOL_POLICY_PROMPT = """You are an embodied navigation policy operating on a
pre-defined indoor viewpoint graph. Follow the navigation input efficiently,
avoid revisiting viewpoints, and never invent a viewpoint ID.

At every decision:
1. Identify the current subgoal from the navigation input.
2. Ground the decision in the current observation and trajectory history.
3. Select only a valid 32-character viewpoint ID exposed by the available
   candidates/tools, or stop only after reaching the described destination.

Available tools: {tool_names}
{tool_descriptions}

Output exactly one of the following forms and no text outside the two blocks.
Keep Think within 60 words and emit Action immediately after it:

<Think>At most 60 words of evidence-grounded reasoning about subgoal, observation, history, and action.</Think>
<Action>action_maker("32-character adjacent viewpoint ID")</Action>

<Think>At most 60 words explaining the evidence for returning to an earlier viewpoint.</Think>
<Action>back_tracer("32-character previously visited viewpoint ID")</Action>

<Think>At most 60 words of evidence that the destination has been reached.</Think>
<Action>Finish!</Action>

Navigation input:
{action_plan}

Initial observation:
{init_observation}

Decision:
{agent_scratchpad}"""


_SINGLE_ACTION_POLICY_PROMPT = """You are an embodied navigation policy
operating on a pre-defined indoor viewpoint graph. Follow the navigation input
efficiently, avoid revisiting viewpoints, and never invent a viewpoint ID.

At every decision:
1. Identify the current subgoal from the navigation input.
2. Ground the decision in the current observation and trajectory history.
3. Select only an adjacent 32-character viewpoint ID exposed by the current
   observation, or stop only after reaching the described destination.

Available tools: {tool_names}
{tool_descriptions}

Output exactly one of the following forms and no text outside the two blocks.
Keep Think within 60 words and emit Action immediately after it:

<Think>At most 60 words of evidence-grounded reasoning about subgoal, observation, history, and action.</Think>
<Action>action_maker("32-character adjacent viewpoint ID")</Action>

<Think>At most 60 words of evidence that the destination has been reached.</Think>
<Action>Finish!</Action>

Navigation input:
{action_plan}

Initial observation:
{init_observation}

Decision:
{agent_scratchpad}"""


VLN_ORCHESTRATOR_PROMPT = _MULTI_TOOL_POLICY_PROMPT
VLN_ORCHESTRATOR_ABS_PROMPT = _MULTI_TOOL_POLICY_PROMPT
VLN_GPT4_PROMPT = _SINGLE_ACTION_POLICY_PROMPT
VLN_GPT35_PROMPT = _SINGLE_ACTION_POLICY_PROMPT

VLN_ORCHESTRATOR_TOOL_PROMPT = """You are an embodied navigation policy
operating on a pre-defined indoor viewpoint graph. Follow the navigation input,
ground every decision in observations and history, and never invent IDs.

Available tools: {tool_names}
{tool_descriptions}

Output exactly one <Think>...</Think> block followed immediately by exactly one
<Action>...</Action> block. Keep Think within 60 words. The action must be
action_maker("viewpoint_id"), back_tracer("viewpoint_id"), or Finish!. Do not
output any other text.

Navigation input:
{action_plan}

Initial observation:
{init_observation}

Current observation:
{observation}

Decision:
{agent_scratchpad}"""


PROMPT_SETS = {
    "plain": {
        "planner": PLANNER_PROMPT,
        "action": ACTION_PROMPT,
        "history": HISTORY_PROMPT,
        "back_trace": BACK_TRACE_PROMPT,
        "vln_orchestrator": VLN_ORCHESTRATOR_PROMPT,
        "vln_orchestrator_tool": VLN_ORCHESTRATOR_TOOL_PROMPT,
        "vln_gpt4": VLN_GPT4_PROMPT,
        "vln_gpt35": VLN_GPT35_PROMPT,
    },
}


def get_prompt_set(chat_template: str) -> dict:
    """Return backend-neutral prompts.

    HF and GGUF chat rendering belongs to their model wrappers so that the
    same navigation prompt is used for local and online models.
    """

    return PROMPT_SETS["plain"]
