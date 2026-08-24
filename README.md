<div align="center">

<h1>🎇NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models</h1>

<div>
    <a href='https://github.com/GengzeZhou' target='_blank'>Gengze Zhou<sup>🍕</sup></a>;
    <a href='http://www.yiconghong.me' target='_blank'>Yicong Hong<sup>🌭</sup></a>;
    <a href='http://www.qi-wu.me' target='_blank'>Qi Wu<sup>🍕</sup></a>
</div>
<sup>🍕</sup>Australian Institude for Machine Learning, The University of Adelaide <sup>🌭</sup>The Australian National University

<br>

<div>
    <a href='https://github.com/GengzeZhou/NavGPT' target='_blank'><img alt="Static Badge" src="https://img.shields.io/badge/NavGPT-v0.1-blue"></a>
    <a href='https://arxiv.org/abs/2305.16986' target='_blank'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://github.com/langchain-ai/langchain"><img alt="Static Badge" src="https://img.shields.io/badge/🦜️🔗-Langchain-green"></a>
</div>

</div>


## 🍹 Abstract
 Trained with an unprecedented scale of data, large language models (LLMs) like ChatGPT and GPT-4 exhibit the emergence of significant reasoning abilities from model scaling. Such a trend underscored the potential of training LLMs with unlimited language data, advancing the development of a universal embodied agent. 
 In this work, we introduce the NavGPT, a purely LLM-based instruction-following navigation agent, to reveal the reasoning capability of GPT models in complex embodied scenes by performing zero-shot sequential action prediction for vision-and-language navigation (VLN).
 At each step, NavGPT takes the textual descriptions of visual observations, navigation history, and future explorable directions as inputs to reason the agent's current status, and makes the decision to approach the target.
 Through comprehensive experiments, we demonstrate NavGPT can explicitly perform high-level planning for navigation, including decomposing instruction into sub-goal, integrating commonsense knowledge relevant to navigation task resolution, identifying landmarks from observed scenes, tracking navigation progress, and adapting to exceptions with plan adjustment. 
 Furthermore, we show that LLMs is capable of generating high-quality navigational instructions from observations and actions along a path, as well as drawing accurate top-down metric trajectory given the agent's navigation history. Despite the performance of using NavGPT to zero-shot R2R tasks still falling short of trained models, we suggest adapting multi-modality inputs for LLMs to use as visual navigation agents and applying the explicit reasoning of LLMs to benefit learning-based models.

## 🍸 Method
![](assets/NavGPT.png)

## 🍻 TODOs

- [x] Release 🎇NavGPT code.
- [x] Data preprocessing code.
- [x] Custuomized LLM inference guidance.

## 🧋 Prerequisites

### 🍭 Installation

Create a conda environment and install all dependencies:

```bash
conda create --name NavGPT python=3.9
conda activate NavGPT
pip install -r requirements.txt
```

For the later LoRA + GRPO training stage, install the additional training
dependencies with `pip install -r requirements-train.txt`.

### 🍬 Data Preparation

Download R2R data from [Dropbox](https://www.dropbox.com/sh/i8ng3iq5kpa68nu/AAB53bvCFY_ihYx1mkLlOB-ea?dl=1). Put the data in `datasets` directory.

Related data preprocessing code can be found in `nav_src/scripts`.

### 🍫 OpenAI API

Get an [OpenAI API Key](https://platform.openai.com/account/api-keys) and add to your environment variables:

```bash
# prepare your private OpenAI key (for Linux)
export OPENAI_API_KEY={Your_Private_Openai_Key}

# prepare your private OpenAI key (for Windows)
set OPENAI_API_KEY={Your_Private_Openai_Key}
```

Alternatively, you can set the key in your code:
```python
import os
os.environ["OPENAI_API_KEY"] = {Your_Private_Openai_Key}
```

## 🍷 R2R Navigation

### 🍴 Reproduce Validation Results

To replicate the performance reported in our paper, use GPT-4 and run validation with following configuration:
```bash
# Linux or MacOS
cd nav_src
python NavGPT.py --llm_backend openai \
    --llm_model_name gpt-4 \
    --navigation_input_mode planner \
    --output_dir ../datasets/R2R/exprs/gpt-4-val-unseen \
    --val_env_name R2R_val_unseen_instr \
    --iters -1
```

Results will be saved in `datasets/R2R/exprs/gpt-4-val-unseen` directory.

The defualt `--llm_model_name` is set as `gpt-3.5-turbo`.

An economic way to try 🎇NavGPT is by using GPT-3.5 and run validation on the first 10 samples with following configuration:
```bash
# Linux or MacOS
cd nav_src
python NavGPT.py --llm_backend openai \
    --llm_model_name gpt-3.5-turbo \
    --navigation_input_mode planner \
    --output_dir ../datasets/R2R/exprs/gpt-3.5-turbo-test \
    --val_env_name R2R_val_unseen_instr \
    --iters 10

# Windows
cd nav_src
python NavGPT.py --llm_backend openai `
  --llm_model_name gpt-3.5-turbo `
  --navigation_input_mode planner `
  --output_dir ../datasets/R2R/exprs/gpt-3.5-test `
  --val_env_name R2R_val_unseen_instr `
  --iters 10
```

### 🥢 Local HF and GGUF backends

`NavGPT.py` supports `--llm_backend openai`, `hf`, and `gguf` for reproducing
the historical LangChain `<Think>/<Action>` system. This entry point is a
legacy evaluator: its output manifest records
`evaluator_family=legacy_langchain` and `official_rl_comparable=false`.
It remains useful for the original OpenAI-system baseline, but it is not the
formal evaluator for the GRPO policy.

Formal Base-Qwen and Base-Qwen+LoRA evaluation uses the single native tool-call
entry point instead (run from the repository root):

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
python nav_src/scripts/evaluate_r2r_native.py \
  --policy-kind base \
  --model-path models/Qwen2.5-14B-Instruct-1M \
  --candidate-label base-qwen \
  --output-dir outputs/native-eval/base-val-unseen \
  --max-new-tokens 256

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
python nav_src/scripts/evaluate_r2r_native.py \
  --policy-kind adapter \
  --model-path models/Qwen2.5-14B-Instruct-1M \
  --adapter-path outputs/<run>/validation/snapshots/step-375 \
  --candidate-label rl-lora-step-375 \
  --output-dir outputs/native-eval/rl-lora-step-375-val-unseen \
  --max-new-tokens 256
```

The two formal runs must keep the same dataset paths and protocol arguments;
only the adapter may differ. Historical evaluator results generated with a
512-token decision budget use an old protocol and are not directly comparable
with the fixed 256-token protocol. `--full-best-run-dir` is accepted only for
a future training run whose completed full validation already carries this
same native 256-token/2349-item protocol; use an explicit immutable snapshot
for historical runs.

See [docs/训练命令.md](docs/训练命令.md#121-唯一正式-native-evaluator) for the
complete four-GPU Base/step-375/step-750 protocol and
[docs/parse中的参数解析.md](docs/parse中的参数解析.md) for CLI options.

## 🧃 Citation
If 🎇`NavGPT` has been beneficial to your research and work, please cite our work using the following format:
```
@article{zhou2023navgpt,
  title={NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models},
  author={Zhou, Gengze and Hong, Yicong and Wu, Qi},
  journal={arXiv preprint arXiv:2305.16986},
  year={2023}
}
```
