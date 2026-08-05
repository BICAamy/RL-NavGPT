# parser.py 参数说明

## 模型后端

- `--llm_backend {openai,hf,gguf}`
  - `openai`：在线 OpenAI completion 接口，使用后端中性的 Prompt。
  - `hf`：Hugging Face causal LM，使用 tokenizer 的
    `apply_chat_template()`。
  - `gguf`：`llama.cpp` 本地模型，使用 chat completion 或原始
    completion。
- `--llm_model_name`：实验记录和在线模型名称。
- `--local_model_path`：HF 模型目录或 GGUF 文件路径。
- `--local_chat_template {auto,qwen,plain}`
  - `auto`：HF 使用 tokenizer 模板；GGUF 使用模型元数据。
  - `qwen`：HF 仍使用 tokenizer 模板；GGUF 显式使用 ChatML。
  - `plain`：不套聊天模板，仅用于原始 completion 模型。
- `--local_dtype {bf16,fp16}`：HF 权重精度。
- `--hf_device_map {single,auto}`：HF 放置方式。论文主实验使用默认的
  `single`，并通过 `CUDA_VISIBLE_DEVICES` 为每个进程只暴露一张卡；
  `auto` 可能跨卡切层，不用于当前已经验证的 Qwen14B 路径。
- `--temperature`、`--top_p`、`--max_new_tokens`：生成参数。
- `--gguf_n_ctx`、`--gguf_n_gpu_layers`、`--gguf_n_threads`：
  `llama.cpp` 参数。

## 导航状态输入

- `--navigation_input_mode {planner,instruction,action_plan}`
  - `planner`：默认。先用当前模型根据原始 instruction 生成 action
    plan，再用该 plan 导航。
  - `instruction`：跳过 planner，直接使用 R2R 原始 instruction。
  - `action_plan`：通过 `--action_plan_cache` 指定合并后的 Planner JSONL，
    按 `instr_id` 读取缓存；原始 R2R annotation 保持不变。
- `--action_plan_cache PATH`：`action_plan` 模式必填。缓存中的
  `instr_id` 和 instruction 必须与当前 split 完整匹配，否则立即报错。

旧参数 `--load_instruction` 和 `--load_action_plan` 暂时保留为兼容
别名，但新实验应使用 `--navigation_input_mode`。

零样本在线基线使用 `planner`；正式训练和训练后主实验统一使用
`action_plan`。缓存由相同模型、Prompt、Chat Template 和 greedy 解码
生成，并通过抽样逐字符复核，因此两种路径送入 Policy 的 action plan
内容一致，同时训练期间的状态不会随 LoRA 更新而漂移。

## 运行范围

- `--iters N`：运行 N 个 episode。
- `--iters -1`：遍历完整验证 split。
- `--val_env_name`：验证集名称。
- `--output_dir`：日志、详细输出和预测文件目录。
- `--valid_file`：不加载 LLM，仅评测已有预测 JSON。

## 布尔选项

布尔参数同时支持开启和关闭。例如：

```bash
--use-relative-angle
--no-use-relative-angle
--use-navigable
--no-use-navigable
```

下划线形式继续兼容。命令行现在严格检查未知参数，拼写错误不会再被
静默忽略。
