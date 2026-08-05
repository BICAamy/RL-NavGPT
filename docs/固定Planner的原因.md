“离线缓存固定 Planner”可以拆成三个词理解：离线、缓存、固定 Planner。

## 1. Planner 到底是什么

这里的 Planner 不只是 `PLANNER_PROMPT` 提示词，而是一套完整的生成函数：

```text
原始 R2R instruction
    ↓
Qwen2.5-14B 基础模型
+ PLANNER_PROMPT
+ apply_chat_template()
+ temperature=0
+ max_new_tokens=256
    ↓
action plan
```

例如：

```text
原始 instruction:
Exit the bedroom, turn left, pass the kitchen and stop near the refrigerator.

Planner 输出:
Action plan:
1. Exit the bedroom.
2. Turn left into the corridor.
3. Continue past the kitchen.
4. Stop near the refrigerator.
```

随后导航 Policy 接收的主要任务状态是这个 action plan，而不是直接接收原始 instruction。

这正是你选择的创新设定：

```text
Planner 负责高层任务分解
Policy 负责根据实时观察逐步导航
```

## 2. “固定 Planner”是什么意思

固定的是整个 Planner，不只是提示词：

- 固定基础模型：`Qwen2.5-14B-Instruct-1M`
- 不挂载正在训练的 Policy LoRA
- 不更新 Planner 参数
- 固定 `PLANNER_PROMPT`
- 固定 `apply_chat_template()`
- 固定 Transformers 版本
- 固定温度、最大生成长度等参数
- 固定生成结果对应的模型和 Prompt 版本

也就是：

```text
Planner：冻结的基础 Qwen14B
Policy：基础 Qwen14B + 正在训练的 LoRA
```

虽然两者可以使用同一份基础权重，但角色不同。

如果不做隔离，训练过程中 Policy LoRA 不断变化，Planner 也可能在挂载 LoRA 的情况下生成计划。这样同一条 instruction 在不同训练阶段会得到不同 action plan，相当于训练状态本身一直在变化。

## 3. “离线缓存”是什么意思

离线缓存不是重新下载 R2R，也不是复制整个数据集，更不是 GPU KV cache。

它表示：在正式训练前，把每条 instruction 的 action plan 预先生成一次，然后保存下来。

例如生成一个 JSONL 文件：

```json
{"instr_id":"6250_0","instruction":"Exit the bedroom...","action_plan":"Action plan:\n1. Exit...\n2. Turn left..."}
{"instr_id":"6250_1","instruction":"Leave the room...","action_plan":"Action plan:\n1. Leave...\n2. Continue..."}
```

训练时不再调用 Planner，而是查询：

```python
action_plan = cache[instr_id]
```

于是流程从：

```text
每次 rollout：
instruction → 调用 Qwen Planner → action plan → Policy 导航
```

变为：

```text
训练前一次性：
instruction → 固定 Qwen Planner → action plan → 保存

每次 rollout：
instr_id → 读取缓存 action plan → Policy 导航
```

action plan 仍然是 Planner 生成的，只是提前生成，而不是训练过程中反复生成。

## 4. 为什么要缓存

### 保证 GRPO 同组样本状态一致

GRPO 通常会对同一条 instruction 采样多条 rollout，再比较它们的奖励。

正确情况：

```text
同一 instruction
+ 同一个 action plan
→ rollout 1
→ rollout 2
→ rollout 3
→ 比较奖励
```

如果每条 rollout 都重新调用 Planner：

```text
同一 instruction
→ action plan A → rollout 1
→ action plan B → rollout 2
→ action plan C → rollout 3
```

此时奖励差异不仅来自 Policy，也可能来自 action plan 不同，GRPO 的组内比较就被污染了。

### 防止训练目标漂移（最重要的！）

Policy 的输入状态包含 action plan。假如 Planner 随训练变化，同一个任务的状态表示会不断变化：

```text
epoch 1：plan A
epoch 2：plan B
epoch 3：plan C
```

Policy 一边学习，输入分布一边移动，训练会更不稳定。

缓存后：

```text
同一个 instr_id 永远对应同一个 action plan
```

这样只有 Policy 在变化，状态定义保持不变。

### 隔离 Planner 和 Policy 的贡献

你的核心训练创新是：

- LoRA Policy
- GRPO
- 导航奖励
- CLIP 语义奖励
- Think 质量奖励

如果 Planner 也同时变化，最后性能提升无法判断究竟来自：

- Policy LoRA；
- 三类奖励；
- Planner 计划变好了；
- 或 Planner 随机生成了更容易的计划。

固定 Planner 后，实验可以明确说明：

> 在相同高层 action plan 下，GRPO 和三类奖励提升了逐步导航 Policy。

这对论文消融实验非常重要。

### 大幅减少训练计算

训练集共有 14039 条 instruction。

不缓存时，假设：

- 每条 instruction 每个 epoch 调用一次 Planner；
- GRPO 每组生成多条 rollout；
- 训练多个 epoch；

Planner 调用次数可能变成数十万次。

缓存后只需要：

```text
14039 条 instruction → 14039 次 Planner 调用
```

以后所有 epoch 都直接读文件。

### 提高复现性

即使 `temperature=0`，不同的：

- Transformers 版本；
- CUDA kernel；
- GPU；
- 多卡执行方式；
- 模型实现；

仍可能导致细微生成差异。你刚刚遇到的 Blackwell 多卡问题就是例子。

缓存后可以明确记录：

```text
instr_id 6250_0 始终使用 plan X
```

别人复现实验时不需要重新生成完全相同的 Planner 输出。

## 5. 为什么在零样本基线之后做

严格来说，缓存并不是算法上必须在基线之后进行。如果一开始已经完全确定模型、Prompt 和软件环境，也可以先缓存。

但在当前项目中，先跑基线再缓存更合理。

### 基线帮助我们确定“固定哪个 Planner”

在基线之前，我们还没有完全验证：

- HF 模型能否正确加载；
- `apply_chat_template()` 是否正确；
- 新 `<Think>/<Action>` 协议能否执行；
- Transformers 版本是否兼容；
- Blackwell 多 GPU 是否可用；
- temperature 0 是否能正常生成；
- Qwen14B 是否具备足够的规划能力。

现在经过完整基线，已经确定可用配置：

```text
Qwen2.5-14B-Instruct-1M
Transformers 4.48.3
PyTorch 2.7.1+cu128
BF16
单 GPU
apply_chat_template()
temperature=0
max_new_tokens=256
PLANNER_PROMPT
```

现在才有资格把这套配置称为“固定 Planner”。

如果在之前错误的多 GPU 环境下缓存，14039 条计划可能全部是乱码，只能全部重做。

### 先建立未经训练的 B0

实验顺序应当是：

```text
阶段一：原始 Qwen14B + 在线 Planner
        → 得到零样本基线 B0

阶段二：固定经过验证的 Planner
        → 缓存训练集和验证集 action plan

阶段三：固定 action plan
        → 使用 GRPO 训练 Policy LoRA

阶段四：固定相同验证集 action plan
        → 测试训练后的 Policy

阶段五：与 B0 比较
```

这样能够证明训练带来的提升，而不是由于后来更换了 Planner。

### 基线日志已经帮我们缓存了验证集

完整基线的详细日志里已经保存了每条样本实际使用的：

```json
{
  "instr_id": "...",
  "action_plan": "..."
}
```

因此2349条验证集计划可以直接从基线日志提取，不需要重新生成。

这样后续评测训练后 Policy 时，可以使用与 B0 完全相同的验证集 action plan：

```text
B0 Policy       + plan X
训练后 Policy    + plan X
```

比较非常干净。

训练集没有跑过完整基线，所以需要另外为 14039 条训练 instruction 生成缓存。

## 6. 这会不会削弱你的创新

不会。

你的状态仍然是：

```text
Planner 生成的 action plan
+ 当前场景观察
+ 历史轨迹
```

缓存只是把：

```text
现在调用 Planner
```

变成：

```text
提前调用同一个 Planner
```

它没有改成原始 instruction，也没有使用真实路径或答案生成计划，因此不会造成标签泄漏。

Planner 只看到自然语言 instruction，不应该看到：

- ground-truth path；
- 目标 viewpoint；
- 导航奖励；
- 未来观察。

## 7. 推理时怎么办

需要区分论文基准评测和真实新指令。

论文验证集：

```text
读取固定验证集 action-plan 缓存
→ 保证与 B0 公平比较
```

真实的新 instruction：

```text
关闭 Policy LoRA
→ 使用冻结基础 Qwen14B 在线生成 action plan
→ 再开启 Policy LoRA 执行导航
```

所以 Planner 能力没有被删除，只是在训练和标准评测中用缓存保证稳定。

一句话总结：

> “离线缓存固定 Planner”就是先用经过零样本基线验证的冻结 Qwen14B，将每条 instruction 转换成唯一、可追溯的 action plan；训练期间始终读取这个计划，使 GRPO 只优化导航 Policy，而不让 Planner、训练状态和计算成本同时漂移。