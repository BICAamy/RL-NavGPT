import argparse
import os


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="")

    # datasets
    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r'])
    parser.add_argument('--output_dir', type=str, default='../datasets/R2R/exprs/gpt-3.5-turbo', help='experiment id')
    # parser.add_argument('--output_dir', type=str, default='../datasets/R2R/exprs/LlaMA-2-13b-test', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    # Agent
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature for llm')
    parser.add_argument('--llm_model_name', type=str, default='gpt-3.5-turbo', help='llm model name')
    parser.add_argument('--llm_backend', type=str, default='openai', choices=['openai', 'hf', 'gguf'], help='llm backend')
    parser.add_argument(
        '--local_model_path',
        type=str,
        default='',
        help='HF model directory or GGUF model file path',
    )
    parser.add_argument(
        '--local_adapter_path',
        type=str,
        default='',
        help=(
            'legacy-only local NavGPT LoRA adapter directory for the HF '
            'backend; official Base/LoRA evaluation must use '
            'nav_src/scripts/evaluate_r2r_native.py'
        ),
    )
    parser.add_argument(
        '--allow_legacy_adapter_evaluation',
        action='store_true',
        default=False,
        help=(
            'DANGEROUS legacy opt-in: allow --local_adapter_path in the old '
            'LangChain NavGPT.py evaluator. Its results are marked as not '
            'official-RL-comparable. Use nav_src/scripts/evaluate_r2r_native.py for '
            'formal Base/LoRA evaluation.'
        ),
    )
    parser.add_argument(
        '--local_chat_template',
        type=str,
        default='qwen',
        choices=['auto', 'plain', 'qwen'],
        help=(
            'local chat rendering: HF uses tokenizer.apply_chat_template for '
            'auto/qwen; GGUF uses model metadata for auto or ChatML for qwen'
        ),
    )
    parser.add_argument('--local_dtype', type=str, default='bf16', choices=['bf16', 'fp16'], help='dtype for local models')
    parser.add_argument(
        '--hf_device_map',
        type=str,
        default='single',
        choices=['single', 'auto'],
        help=(
            'HF placement policy. single (default) keeps the full model on '
            'one visible GPU; auto may split layers across devices.'
        ),
    )
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p for local models')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='max new tokens for local models')
    parser.add_argument('--gguf_n_ctx', type=int, default=4096, help='context length for gguf models')
    parser.add_argument('--gguf_n_gpu_layers', type=int, default=0, help='gpu layers for gguf models')
    parser.add_argument('--gguf_n_threads', type=int, default=0, help='cpu threads for gguf models (0=auto)')
    # parser.add_argument('--llm_model_name', type=str, default='gpt-4', help='llm model name')
    # parser.add_argument('--llm_model_name', type=str, default='LlaMA-2-13b', help='llm model name')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_iterations', type=int, default=10)

    # General config
    parser.add_argument(
        '--iters',
        type=int,
        default=10,
        help='number of episodes to run; use -1 to evaluate the complete split',
    )
    # parser.add_argument('--iters', type=int, default=None, help='number of iterations to run')
    parser.add_argument(
        '--max_scratchpad_length',
        type=int,
        default=7000,
        help='maximum number of recent scratchpad characters kept in the prompt',
    )
    parser.add_argument('--test', action='store_true', default=False)
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_0')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_1')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_2')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_3')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_4')
    parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr')

    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument(
        '--navigation_input_mode',
        choices=['planner', 'instruction', 'action_plan'],
        help=(
            'planner (default): generate an action plan with the selected LLM; '
            'instruction: use the raw R2R instruction; action_plan: read an '
            'action plan from --action_plan_cache'
        ),
    )
    input_mode.add_argument(
        '--load_instruction',
        dest='navigation_input_mode',
        action='store_const',
        const='instruction',
        help='deprecated alias for --navigation_input_mode instruction',
    )
    input_mode.add_argument(
        '--load_action_plan',
        dest='navigation_input_mode',
        action='store_const',
        const='action_plan',
        help='deprecated alias for --navigation_input_mode action_plan',
    )
    parser.set_defaults(navigation_input_mode='planner')
    parser.add_argument(
        '--action_plan_cache',
        type=str,
        default='',
        help=(
            'merged Planner JSONL cache required by '
            '--navigation_input_mode action_plan'
        ),
    )

    parser.add_argument(
        '--use_relative_angle',
        '--use-relative-angle',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--use_history_chain',
        '--use-history-chain',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--use_tool_chain',
        '--use-tool-chain',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--use_navigable',
        '--use-navigable',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--use_single_action',
        '--use-single-action',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        '--detailed_output',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # parser.add_argument('--valid_file', type=str, default='../datasets/R2R/exprs/4-R2R_val_unseen_instr/4-R2R_val_unseen_instr.json', help='valid file name')
    parser.add_argument('--valid_file', type=str, default=None, help='valid file name')

    args = parser.parse_args(argv)

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    if args.local_adapter_path:
        if not args.allow_legacy_adapter_evaluation:
            raise ValueError(
                '--local_adapter_path is disabled in legacy NavGPT.py by '
                'default. Formal Base/LoRA evaluation must use '
                'nav_src/scripts/evaluate_r2r_native.py. For an explicitly '
                'non-comparable historical reproduction only, add '
                '--allow_legacy_adapter_evaluation.'
            )
        if args.llm_backend != 'hf':
            raise ValueError(
                '--local_adapter_path is valid only with --llm_backend hf'
            )
        if not args.local_model_path:
            raise ValueError(
                '--local_model_path is required with --local_adapter_path'
            )
        if args.hf_device_map != 'single':
            raise ValueError(
                'LoRA inference requires --hf_device_map single'
            )

    ROOTDIR = args.root_dir

    # Setup input paths
    args.obs_dir = os.path.join(ROOTDIR, 'R2R', 'observations_list_summarized')
    args.obs_summary_dir = os.path.join(ROOTDIR, 'R2R', 'observations_summarized')
    args.obj_dir = os.path.join(ROOTDIR, 'R2R', 'objects_list')

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
    args.navigable_dir = os.path.join(ROOTDIR, 'R2R', 'navigable')

    # Build paths
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args
