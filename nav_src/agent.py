"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""
import numpy as np
from typing import Any, List, Optional, Tuple, Dict, Union

from env import R2RNavBatch
from argparse import Namespace
from agent_base import BaseAgent

from langchain.agents.agent import AgentExecutor, AgentOutputParser
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    OutputParserException
)
from langchain.base_language import BaseLanguageModel

from policy_output import (
    FINISH_ACTION,
    PolicyOutputParseError,
    parse_policy_output,
)
from prompt.planner_prompt import (
    MAKE_ACTION_TOOL_NAME,
    MAKE_ACTION_TOOL_DESCRIPTION,
    BACK_TRACE_TOOL_NAME,
    BACK_TRACE_TOOL_DESCRIPTION,
    get_prompt_set,
)
from planner import build_planner_chain, generate_action_plan
from navigation_state import (
    NavigationPromptConfig,
    NavigationStateBuilder,
    PromptTraceStep,
    build_policy_scratchpad,
    describe_turn,
)

MAX_SCRATCHPAD_LENGTH = 7000


class NavGPTOutputParser(AgentOutputParser):
    """Adapt the canonical Think/Action protocol to LangChain agent actions."""

    def get_format_instructions(self) -> str:
        return (
            '<Think>reasoning</Think>\n'
            '<Action>action_maker("32-character viewpoint ID")</Action> '
            "or <Action>Finish!</Action>"
        )

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            output = parse_policy_output(text)
        except PolicyOutputParseError as exc:
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=(
                    f"Invalid navigation output: {exc} Output exactly one "
                    "<Think>...</Think> block followed by one "
                    "<Action>...</Action> block."
                ),
                llm_output=text,
                send_to_llm=True,
            ) from exc

        if output.is_finish:
            return AgentFinish(
                {"output": FINISH_ACTION},
                text,
            )

        return AgentAction(output.action_name, output.viewpoint_id, text)

    @property
    def _type(self) -> str:
        return "mrkl-NavGPT"

class VLNAgent(ZeroShotAgent):

    history: Optional[List[str]] = None 
    max_scratchpad_length: int = MAX_SCRATCHPAD_LENGTH

    @property
    def llm_prefix(self) -> str:
        """Prompt prefix used after every tool observation."""

        return "Decision:"

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        history = self.history or []
        move_index = 1
        trace = []
        for action, observation in intermediate_steps:
            history_text = (
                history[move_index]
                if move_index < len(history)
                else observation
            )
            trace.append(
                PromptTraceStep(
                    model_output=action.log,
                    action_name=action.tool,
                    history=history_text,
                    observation=observation,
                )
            )
            if action.tool == MAKE_ACTION_TOOL_NAME:
                move_index += 1
        return build_policy_scratchpad(trace, self.max_scratchpad_length)

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)[
            -self.max_scratchpad_length:
        ]
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        if len(intermediate_steps) == 0:
            full_inputs = {**kwargs, **new_inputs}
        else:
            kwargs["init_observation"] = self.history[0]
            full_inputs = {**kwargs, **new_inputs}
        return full_inputs

class NavAgent(BaseAgent):
    def __init__(
            self, 
            env: R2RNavBatch, 
            config: Namespace):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)
        self.config = config
        self.state_builder = NavigationStateBuilder(
            NavigationPromptConfig.from_namespace(config)
        )

        if config.llm_backend == 'openai':
            self.prompt_set = get_prompt_set('plain')
            if config.llm_model_name.split('-')[0] == 'gpt':
                if (
                    "turbo" in config.llm_model_name
                    and "instruct" not in config.llm_model_name
                ) or config.llm_model_name.startswith("gpt-4"):
                    self.llm = ChatOpenAI(
                        temperature=config.temperature,
                        model_name=config.llm_model_name,
                    )
                else:
                    self.llm = OpenAI(
                        temperature=config.temperature,
                        model_name=config.llm_model_name,
                    )
            elif config.llm_model_name == 'llama-2-13b':
                from LLMs.Langchain_llama import Custom_Llama
                ckpt_dir = "LLMs/llama/llama-2-13b"
                tokenizer_path = "LLMs/llama/tokenizer.model"
                self.llm = Custom_Llama.from_model_id(
                    temperature=config.temperature,
                    ckpt_dir = ckpt_dir,
                    tokenizer_path = tokenizer_path,
                    max_seq_len = 8000,
                    max_gen_len = 500,
                    max_batch_size = 1,
                )
            else:
                raise ValueError(f"Unsupported llm_model_name for openai backend: {config.llm_model_name}")
        elif config.llm_backend == 'hf':
            self.prompt_set = get_prompt_set('plain')
            self.llm = self._build_hf_llm()
        elif config.llm_backend == 'gguf':
            self.prompt_set = get_prompt_set('plain')
            self.llm = self._build_gguf_llm()
        else:
            raise ValueError(f"Unsupported llm_backend: {config.llm_backend}")
        # elif config.llm_model_name == 'Vicuna-v1.5-13b':
        #     from LLMs.Langchain_Vicuna import Custom_Vicuna
        #     self.llm = Custom_Vicuna.from_config(
        #         config = config,
        #     )
        # elif config.llm_model_name == 'FlanT5XXL':
        #     from LLMs.Langchain_FlanT5 import Custom_FlanT5
        #     self.llm = Custom_FlanT5.from_config(
        #         config = config,
        #     )
        # elif config.llm_model_name == 'Emu-14B':
        #     from LLMs.Langchain_Emu import Custom_Emu
        #     self.llm = Custom_Emu.from_config(
        #         config = config,
        #     )
        # else:
        #     from LLMs.Langchain_InstructBLIP import Custom_NavGPT_InstructBLIP
        #     self.llm = Custom_NavGPT.from_config(
        #         config = config,
        #     )

        self.output_parser = NavGPTOutputParser()
        self.agent_executor = self.create_vln_agent()

        self.plan_chain = build_planner_chain(self.llm)

    def _build_hf_llm(self) -> BaseLanguageModel:
        if not self.config.local_model_path:
            raise ValueError("local_model_path is required when llm_backend=hf")
        if self.config.local_model_path.endswith(".gguf"):
            raise ValueError("GGUF models are not supported by transformers. Download a HF model or use a GGUF backend.")

        import torch
        from LLMs.hf_chat import HuggingFaceChatLLM

        dtype = torch.bfloat16 if self.config.local_dtype == "bf16" else torch.float16
        return HuggingFaceChatLLM.from_model_path(
            model_path=self.config.local_model_path,
            adapter_path=(
                getattr(self.config, "local_adapter_path", "") or None
            ),
            dtype=dtype,
            device_map=self.config.hf_device_map,
            chat_template=self.config.local_chat_template,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
        )

    def _build_gguf_llm(self) -> BaseLanguageModel:
        if not self.config.local_model_path:
            raise ValueError("local_model_path is required when llm_backend=gguf")
        if not self.config.local_model_path.endswith(".gguf"):
            raise ValueError("GGUF backend requires a .gguf model file")

        from LLMs.gguf_llama import GGUF_Llama

        n_threads = self.config.gguf_n_threads if self.config.gguf_n_threads > 0 else None
        return GGUF_Llama.from_model_path(
            model_path=self.config.local_model_path,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n_ctx=self.config.gguf_n_ctx,
            n_gpu_layers=self.config.gguf_n_gpu_layers,
            n_threads=n_threads,
            chat_template=self.config.local_chat_template,
        )
    
    def parse_action(self, llm_output: str) -> Tuple[str, str]:
        output = parse_policy_output(llm_output)
        action = FINISH_ACTION if output.is_finish else output.viewpoint_id
        return output.thought, action

    def get_his_viewpoints(self) -> str:
        '''Return the history of visited viewpoints for back tracing.'''
        his_viewpoints = ''
        # The last vp is not included in the history
        for i, detail in enumerate(self.traj[0]['details'][:-1]):
            viewpointID = detail['viewpointID']
            viewpoint_ob = detail['feature']
            his_viewpoints += f"Step {i+1}. Viewpoint ID '{viewpointID}':\n {viewpoint_ob}\n\n"
        return his_viewpoints
    
    def get_history(self, obs: dict, angle: str) -> str:
        '''Return the history of actions taken.'''
        return self.state_builder.history_after_move(obs, angle)

    def get_navigable_str(self, cur_heading: float, cur_elevation: float, navigable: dict) -> str:
        '''Return the navigable viewpoints as a string.'''
        return self.state_builder.get_navigable_str(
            cur_heading,
            cur_elevation,
            navigable,
        )

    def modify_heading_angles(self, heading_angle, observation_list, candidate_dict, object_list):
        return self.state_builder.modify_heading_angles(
            heading_angle,
            observation_list,
            candidate_dict,
            object_list,
        )

    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        # Record the navigation path
        self.traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': [],
        } for ob in obs]
        # Record the history of actions taken
        self.agent_executor.agent.history = [
            self.state_builder.initial_history(obs[0])
        ]

    def _create_make_action_tool(
            self,
            llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to make single action prediction in MP3D.

        The tool is invoked with the simulation environment and records the
        action taken by the agent.
        The tool interacts with the environment to obtain the current observation, 
        uses the LLM to predict the next action, and to summarize the previous trajectory
        into history.
        """

        action_prompt = PromptTemplate(
            template=self.prompt_set["action"],
            input_variables=["action_plan", "observation", "history", "navigable_viewpoints"],
        )
        history_prompt = PromptTemplate(
            template=self.prompt_set["history"],
            input_variables=["history", "previous_action", "observation"],
        )
        self.action_chain = LLMChain(llm=llm, prompt=action_prompt)
        self.history_chain = LLMChain(llm=llm, prompt=history_prompt)

        def _make_action(*args, **kwargs) -> str:
            '''Make single step action in MatterSim.'''
            # Get current observation
            cur_obs = self.env._get_obs()[0]

            # Get current feature
            feature = cur_obs['obs']
            heading = np.rad2deg(cur_obs['heading'])
            elevation = np.rad2deg(cur_obs['elevation'])
            objects = cur_obs['objects']
            orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
            navigable = cur_obs['candidate']
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(heading, feature, navigable, objects)
            if self.config.use_navigable:
                navigable = self.get_navigable_str(heading, elevation, navigable)

            if self.config.use_tool_chain:
                # Get current action plan
                action_plan = self.cur_action_plan
                # Single step action
                LLM_action_output = self.action_chain.run(
                    action_plan = action_plan, 
                    observation = feature, 
                    history = self.agent_executor.agent.history[-1], 
                    navigable_viewpoints = navigable
                )
                # Parse LLM output, action is the next viewpoint ID
                thought, action = self.parse_action(LLM_action_output)
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            # Make the action in Simulator
            if action not in self.env.env.sims[0].navigable_dict.keys():
                # Update history
                history = f'ViewpointID "{action}" is not valid, no action taken for the agent.'
                self.agent_executor.agent.history.append(history)
                if self.config.use_navigable:
                    return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}"
            else:
                turned_angle, new_obs = self.make_equiv_action([action])

            # Update the current feature
            new_feature = new_obs['obs']
            new_feature_sum = new_obs['obs_summary']
            new_navigable = new_obs['candidate']
            new_objects = new_obs['objects']
            new_heading = np.rad2deg(new_obs['heading'])
            new_elevation = np.rad2deg(new_obs['elevation'])
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(new_heading, new_feature, new_navigable, new_objects)
            new_orientation = f'\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}'
            if self.config.use_navigable:
                new_navigable = self.get_navigable_str(new_heading, new_elevation, new_navigable)

            # Update history
            if self.config.use_history_chain:
                history = self.history_chain.run(
                    observation = new_feature_sum, 
                    history = self.agent_executor.agent.history[-1], 
                    previous_action = turned_angle
                )
            else:
                history = self.get_history(new_obs, turned_angle)
            
            self.agent_executor.agent.history.append(history)
            # Record single step detail
            if self.config.use_tool_chain:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "acion_maker_thought": thought,
                    "feature": new_feature,
                    "history": self.agent_executor.agent.history[-1],
                }
            else:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "feature": new_feature,
                    "history": self.agent_executor.agent.history[-1],
                }
            self.traj[0]['details'].append(detail)
            # Return LLM chain output as the observation of tool
            if self.config.use_tool_chain:
                return f"\n\tAction_maker Thought:\n{thought}\n\tAction_maker Action:\n{turned_angle}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f'\nCurrent Viewpoint "{action}":\n{new_feature}'
            else:
                if self.config.use_navigable:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"
                

        return Tool(
            name=MAKE_ACTION_TOOL_NAME,
            func=_make_action,
            description=MAKE_ACTION_TOOL_DESCRIPTION,
        )

    def _create_back_trace_tool(
            self,
            llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to back trace during navigation.

        The tool is invoked with the history of navigation trajectory.
        Using the LLM to find a viewpoint on the trajectory to back trace to.
        """
        prompt = PromptTemplate(
            template=self.prompt_set["back_trace"],
            input_variables=["action_plan", "history", "observation"],
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        def _back_trace(*args, **kwargs) -> str:
            '''Back trace the action plan.'''
            cur_obs = self.env._get_obs()[0]

            # Get current feature
            feature = cur_obs['obs']
            navigable = cur_obs['candidate']
            objects = cur_obs['objects']
            heading = np.rad2deg(cur_obs['heading'])
            elevation = np.rad2deg(cur_obs['elevation'])
            orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(heading, feature, navigable, objects)
            if self.config.use_navigable:
                navigable = self.get_navigable_str(heading, elevation, navigable)

            if self.config.use_tool_chain:
                # Get current action plan
                action_plan = self.cur_action_plan
                # Get all previous viewpoints observation
                previous_vp = self.get_his_viewpoints()
                # Back trace
                LLM_output = chain.run(action_plan = action_plan, observation = previous_vp, history = self.agent_executor.agent.history[-1])
                # Parse LLM output, action is the next viewpoint ID
                thought, action = self.parse_action(LLM_output)
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            # Make the action in Simulator
            if action not in self.env.env.sims[0].navigable_dict.keys():
                if self.config.use_navigable:
                    return f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    return f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"
            else:
                _, new_obs = self.make_equiv_action([action])
            
            # Update the current feature
            new_feature = new_obs['obs']
            new_navigable = new_obs['candidate']
            new_objects = new_obs['objects']
            new_heading = np.rad2deg(new_obs['heading'])
            new_elevation = np.rad2deg(new_obs['elevation'])
            new_orientation = f'\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}'
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(new_heading, new_feature, new_navigable, new_objects)
            if self.config.use_navigable:
                new_navigable = self.get_navigable_str(new_heading, new_elevation, new_navigable)

            # Update history
            history = self.get_history(new_obs, 'Seems going in a wrong way, back trace to a previous point.')
            self.agent_executor.agent.history.append(history)
            # Record single step detail
            if self.config.use_tool_chain:
                return f"\tBack_tracer Thought:\n{thought}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\nCurrent Viewpoint:{action}\n{new_feature}"
            else:
                if self.config.use_navigable:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"

        return Tool(
            name=BACK_TRACE_TOOL_NAME,
            func=_back_trace,
            description=BACK_TRACE_TOOL_DESCRIPTION,
        )

    def create_vln_agent(
        self,
    ) -> AgentExecutor:
        """Instantiate API planner and controller for a given trajectory.

        We use a top-level "orchestrator" agent to invoke the planner and controller,
        rather than a top-level planner
        that invokes a controller with its plan. This is to keep the planner simple.
        """

        self.action_maker = self._create_make_action_tool(self.llm)
        self.back_tracer = self._create_back_trace_tool(self.llm)

        tools = [
            self.action_maker,
            self.back_tracer
        ]

        if self.config.use_tool_chain:
            prompt = PromptTemplate(
                template=self.prompt_set["vln_orchestrator_tool"],
                input_variables=["action_plan", "init_observation", "observation", "agent_scratchpad"],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
        elif self.config.use_single_action:
            tools = [self.action_maker]
            prompt = PromptTemplate(
                template=self.prompt_set["vln_gpt4"] if self.config.llm_model_name == 'gpt-4' else self.prompt_set["vln_gpt35"],
                input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
        else:
            prompt = PromptTemplate(
                template=self.prompt_set["vln_orchestrator"],
                input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
        agent = VLNAgent(
            llm_chain=LLMChain(llm=self.llm, prompt=prompt),
            allowed_tools=[tool.name for tool in tools],
            output_parser=self.output_parser,
            max_scratchpad_length=self.config.max_scratchpad_length,
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors = True,
            return_intermediate_steps=True,
            max_iterations=self.config.max_iterations,
        )
    
    def make_equiv_action(self, actions: List[str]) -> Tuple[str, dict]:
        """
        Interface between Panoramic view and Egocentric view
        Take in the next viewpoint ID and move the agent to that viewpoint
        return the turned angle and new observation
        """
        # Get current agent facing angle
        cur_obs = self.env._get_obs()[0]
        cur_heading = np.rad2deg(cur_obs['heading'])
        # Make the action
        new_obs = self.env.step(actions)[0]
        new_heading = np.rad2deg(new_obs['heading'])
        # Record the trajectory
        self.traj[0]['path'].append(self.env.env.sims[0].gmap.bfs_shortest_path(cur_obs['viewpoint'], actions[0])[1:])
        # Calculate the turned angle
        action_description = describe_turn(cur_heading, new_heading)
        return action_description, new_obs

    def rollout(self, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajecotry(obs)

        # Load the instruction
        instructions = [ob['instruction'] for ob in obs]
        if self.config.navigation_input_mode == 'instruction':
            action_plans = instructions
        elif self.config.navigation_input_mode == 'action_plan':
            missing = [ob['instr_id'] for ob in obs if 'action_plan' not in ob]
            if missing:
                raise ValueError(
                    "navigation_input_mode=action_plan requires a cached plan "
                    f"for every instruction; missing for {missing[:3]}"
                )
            action_plans = [ob['action_plan'] for ob in obs]
        else:
            action_plans = []
            for instruction in instructions:
                action_plan = generate_action_plan(self.plan_chain, instruction)
                action_plans.append(action_plan)

        for i, init_ob in enumerate(obs):
            self.cur_action_plan = action_plans[i]
            # Take the first action
            if self.config.use_tool_chain:
                first_obs = self.action_maker('')
                input = {
                    'action_plan': self.cur_action_plan,
                    'init_observation': init_ob['obs_summary'],
                    'observation': first_obs,
                }
            else:
                input = {
                    'action_plan': self.cur_action_plan,
                    'init_observation': (
                        self.state_builder.format_initial_observation(init_ob)
                    ),
                }
            output = self.agent_executor(input)

            self.traj[i]['llm_output'] = output['output']
            self.traj[i]['action_plan'] = output['action_plan']
            # extract agent's thought from llm output
            intermediate_steps = output['intermediate_steps']
            self.traj[i]['llm_thought'] = []
            self.traj[i]['llm_observation'] = []
            for action, observation in intermediate_steps:
                thought = action.log
                self.traj[i]['llm_thought'].append(thought)
                self.traj[i]['llm_observation'].append(observation)

        return self.traj
