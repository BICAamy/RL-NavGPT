- `--llm_backend`：选择使用哪种模型后端  
  - `openai`：在线 API（原版）  
  - `hf`：本地 HF safetensors  
  - `gguf`：本地 GGUF（llama.cpp）

- `--local_model_path`：本地模型路径  
  - HF 模型：是目录  
  - GGUF：是 `.gguf` 文件路径

- `--local_chat_template`：Prompt 模板类型  
  - `plain`：原始纯文本  
  - `qwen`：Qwen 专用 `<|im_start|>` Chat 模板

- `--local_dtype`：HF 本地模型的精度  
  - `bf16` / `fp16`（仅 HF 生效，GGUF 不用）

- `--top_p`：采样的“概率截断”上限  
  - 控制生成的多样性，越小越保守

- `--max_new_tokens`：每次生成的最大 token 数  
  - 太大速度慢，太小可能截断

- `--gguf_n_ctx`：GGUF 的上下文长度（KV cache）  
  - 越大越吃内存

- `--gguf_n_gpu_layers`：GGUF 的 GPU 层数  
  - Metal / CUDA 加速的层数，越大越快但更吃显存

- `--gguf_n_threads`：GGUF 的 CPU 线程数  
  - 0=自动，适合本机 CPU 核数

如果你需要，我可以再给你一份**“推荐默认值”配置表（mac / A100 版）**。