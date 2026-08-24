# RL-NavGPT GRPO 训练现状、问题定位与修复审查任务书

> 目标读者：VS Code Codex / 代码审查与修复 Agent  
> 仓库：`https://github.com/BICAamy/RL-NavGPT`  
> 本次正式训练基准提交：`383f4b90ce1055e7894e4d2c68edfb55e604c7f0`  
> 当前仓库默认分支相对该提交已有少量后续修改，至少包含 `rl_env.py` 的 TRL external-cutoff 协议修复。  
> **注意：分析与修复时必须区分“本次训练实际使用的代码”与“当前默认分支代码”。不要直接假定默认分支就是训练时的完整实现。**

---

# 1. 当前任务目标

当前项目在做 R2R Vision-Language Navigation 的 GRPO 强化学习训练。

核心结构：

```text
R2R instruction
    ↓
固定 Planner 离线生成 action plan
    ↓
Qwen2.5-14B-Instruct-1M
    ↓
LoRA Policy
    ↓
GRPO
    ↓
Navigation + Semantic + Thought composite reward
```

当前目标不是立刻继续长时间训练，而是：

1. 审查当前 GRPO 训练链路是否正确；
2. 查清为什么 RL 相对 zero-shot 只产生较小提升；
3. 修复已经确认的协议、reward、验证流程问题；
4. 在小规模 smoke run 上重新验证 reward alignment；
5. 只有通过验收后，才重新进行正式长训练。

---

# 2. 当前训练配置

本次正式训练使用：

```text
Base model:
Qwen2.5-14B-Instruct-1M

LoRA:
r = 16
alpha = 32
dropout = 0.05
target modules:
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj

训练精度:
bf16

GRPO:
num_generations = 4
per_device_train_batch_size = 1
beta = 0.001
temperature = 1.0
top_p = 0.95

导航:
max_navigation_steps = 10
max_tool_calling_iterations = 10

训练:
trainer_max_steps = 750
learning_rate = 2e-6
seed = 0

验证:
fast Val-Unseen subset = 128
fast validation step = 375 / 750
validation_max_new_tokens = 512
```

训练集大小：

```text
R2R train instructions = 14,039
```

本项目当前语义是：

```text
1 optimizer/global step
≈ 1 instruction
≈ 1 GRPO group
≈ 4 rollouts
```

因此本次训练仅覆盖：

```text
750 / 14039 ≈ 5.34% epoch
```

750 step 并不等于“一轮训练”，而只是约 5.3% 的训练任务覆盖。

---

# 3. 当前已有评测结果

## 3.1 Zero-shot baseline

使用相同 Qwen2.5-14B，无 LoRA，固定 action-plan 输入：

```text
Val-Unseen full 2349
max_new_tokens = 256

SR        = 22.05
Oracle SR = 38.06
SPL       = 14.21
nDTW      = 30.71
SDTW      = 14.36
CLS       = 31.51
Nav Error = 8.09
```

## 3.2 Ours step-375

固定 action-plan：

```text
SR        = 21.46
Oracle SR = 39.12
SPL       = 13.88
nDTW      = 30.88
SDTW      = 14.00
CLS       = 31.71
Nav Error = 7.95
```

step-375 相比 zero-shot：

```text
SR 下降
SPL 下降
Oracle SR 提升
Nav Error 改善
nDTW 小幅提升
CLS 小幅提升
```

## 3.3 Ours step-750

固定 action-plan：

```text
SR        = 23.33
Oracle SR = 39.85
SPL       = 14.86
nDTW      = 30.90
SDTW      = 15.14
CLS       = 31.48
Nav Error = 8.01
```

step-750 相比 zero-shot：

```text
SR:        22.05 → 23.33   (+1.28 pp)
SPL:       14.21 → 14.86   (+0.65)
Oracle SR: 38.06 → 39.85   (+1.79 pp)
SDTW:      14.36 → 15.14   (+0.78)
nDTW:      30.71 → 30.90   (+0.19)
Nav Error: 8.09  → 8.01    (改善 0.08 m)
```

结论：

> RL 确实产生了正向变化，但增益目前比较有限。

---

# 4. max_new_tokens 已基本排除

step-750 分别运行：

```text
max_new_tokens = 256
max_new_tokens = 512
```

最终指标完全一致：

```text
action_steps: 8.69
steps: 8.70
lengths: 17.41
nav_error: 8.01
oracle_error: 4.69
sr: 23.33
oracle_sr: 39.85
spl: 14.86
nDTW: 30.90
SDTW: 15.14
CLS: 31.48
```

因此当前没有证据说明 `max_new_tokens=256` 是性能瓶颈。模型显然在 256 token 以内就完成单次决策生成。

**不要把主要精力继续放在 max_new_tokens。**

---

# 5. 已确认问题一：训练 / fast validation / 当前 full eval 推理协议不统一

这是当前最高优先级问题之一。

## 5.1 训练与训练内 validation

训练内使用：

```text
Qwen native chat template
+
tools=[submit_navigation_decision]
+
TRL native tool-call transcript
```

核心流程位于：

```text
nav_src/r2r_evaluation.py
ToolPolicyEpisodeRunner
```

逻辑类似：

```python
tokenizer.apply_chat_template(
    conversation=messages,
    tools=[environment.submit_navigation_decision],
    ...
)

model.generate(...)
parse_response(...)
tool_call.arguments["policy_output"]
```

模型训练和训练内 quick validation 实际学习/评估的是：

```text
assistant
  ↓
native tool_call
  ↓
submit_navigation_decision
  ↓
policy_output:
<Think>...</Think>
<Action>...</Action>
```

## 5.2 当前 full 2349 evaluation

目前 full evaluation 是通过：

```text
NavGPT.py
```

完成。

它走的是旧的 LangChain / AgentExecutor 逻辑，不是训练时 TRL native tool transport。

虽然：

```text
NavigationStateBuilder
policy prompt
action plan
scene text
history
```

有较多共享实现，但外层生成协议不同。

因此：

```text
训练 / quick validation
≠
当前 full evaluation
```

### 后果

之前 quick-128：

```text
step375 > step750
```

但 full-2349：

```text
step750 > step375
```

这个 checkpoint ranking inversion 不能简单归因于 `128 sample 太小`，首先必须消除 evaluator protocol mismatch。

---

# 6. 已确认问题二：max_steps 训练结束不会自动 full validation

当前 validation callback：

```text
on_step_end
→ fast validation

on_epoch_end
→ full validation
```

但本次训练：

```text
trainer_max_steps = 750
```

在不到一个 epoch 时提前停止，因此不会触发 `on_epoch_end`，也就不会自动产生 `full_best`。

这会导致：

```text
validation/state.json
full_best = null
```

即使训练已经正常结束。

## 修复要求

当：

```text
trainer_max_steps > 0
```

且训练正常结束时，应该增加 final validation 逻辑。

建议：

```text
train end
    ↓
确保 final checkpoint / snapshot 存在
    ↓
准备候选：
- current final checkpoint
- current quick_best（若存在）
    ↓
full Val-Unseen
    ↓
selection_key
    ↓
更新 full_best
```

需要保证：

1. 中断恢复安全；
2. evaluation artifact 不重复覆盖；
3. 仍然使用现有 immutable snapshot / queue / selector 机制；
4. 不破坏 epoch-based full validation。

---

# 7. reward_metadata 完全没有接入训练数据

已实际检查：

```text
total instructions               = 14039
with reward_metadata             = 0
with nonempty subgoal_viewpoints = 0
with nonempty landmark_viewpoints= 0
```

因此以下 reward 在整个训练中从未生效：

```text
navigation/subgoal_completion
navigation/landmark_deviation
```

真实日志：

```text
navigation/subgoal_completion
nonzero = 0.00%

navigation/landmark_deviation
nonzero = 0.00%
```

而 reward config 原本定义：

```text
subgoal_completion_reward = +30
landmark_deviation_penalty = -50
```

## 问题本质

当前 `attach_action_plans()` 只附加：

```text
action_plan
planner_fingerprint
```

不会生成 `reward_metadata`。

环境也只是：

```python
if "reward_metadata" in item:
    ob["reward_metadata"] = item["reward_metadata"]
```

所以 reward 设计与数据流水线没有真正接通。

---

# 8. reward 真实统计

训练日志：

```text
navigation_rollouts.jsonl
```

经过 canonicalization 后：

```text
raw rollouts       = 3036
canonical rollouts = 3000
stale rollouts     = 36

canonical groups   = 750
complete groups    = 750
```

36 条 stale rollout 来源于：

```text
session 0 跑过 checkpoint-100 后继续到了约 step109
之后从 checkpoint-100 resume
```

因此旧 session 中 step>=100 的 9 groups（36 rollouts）应排除。

---

# 9. GRPO group 内训练信号是健康的

关键数据：

```text
zero-range groups = 0 / 759
range < 1 groups  = 1 / 759
same-path groups  = 1 / 759
mean unique paths/group   ≈ 3.80 / 4
mean unique rewards/group ≈ 3.97 / 4
```

结论：

> 当前 GRPO 并不存在“四条 rollout 几乎完全一样 → reward 一样 → advantage 约等于 0”的主要问题。

GRPO 有充分的 group 内 variation。

因此：

```text
“RL 没效果是因为 GRPO advantage 没信号”
```

目前基本可以排除。

---

# 10. Success / Failure terminal reward 是正确的

canonical groups：

```text
all-fail      = 555 (74.00%)
mixed         = 192 (25.60%)
all-success   = 3   (0.40%)
```

在 mixed group 中：

```text
success reward > failure reward
= 100.00% (621/621)

best-reward rollout is successful
= 100.00% (192/192)
```

在 protocol-clean mixed groups 中同样为 100%。

因此：

```text
success +200
failure -80
```

总体方向正确。

**暂时不要首先修改 terminal success/failure reward。**

---

# 11. 核心问题：74% group 全失败，训练主要依赖 dense shaping

```text
all-fail groups = 555 / 750 = 74%
```

也就是说四分之三 optimizer steps 中：

```text
4 条 rollout 全失败
```

没有 `success +200` 帮忙建立明确 preference。

因此这些 group 的学习主要依赖：

```text
navigation dense shaping
semantic
thought
```

这使 dense reward 是否正确排序失败轨迹成为当前最重要的问题。

---

# 12. Reward alignment audit 结果

最重要的是：

```text
D. CLEAN ALL-FAIL GROUPS
groups = 354
```

这些 group：

- 没有成功 rollout；
- 没有 protocol violation；
- 完全依赖 dense shaping。

## 12.1 Total reward vs final distance

```text
group-centered Pearson(reward, -final_distance)
= 0.231369

mean within-group Spearman
= 0.231060

pairwise ranking accuracy
= 60.57%

best-reward == best-final-distance
= 45.76%

mean final-distance regret of reward winner
= 2.2755 m
```

说明：

> total reward 对最终 endpoint 的排序有正相关，但并不强。

## 12.2 Total reward vs minimum distance

```text
Pearson(reward, -minimum_distance)
= 0.477754

Spearman
= 0.449989

pairwise ranking accuracy
= 73.04%

best-reward == best-minimum-distance
= 68.64%
```

明显比 final distance 强。

---

# 13. 当前最重要的结论：reward 偏向“曾经靠近目标”，而不是“最终停得好”

当前 reward：

```text
reward vs minimum distance
明显强于
reward vs final distance
```

这和最终 evaluation 中：

```text
Oracle SR 提升较明显
SR / SPL 提升较小
```

高度一致。

也就是说当前 RL 更容易优化：

```text
“轨迹某个时刻曾经进入/靠近目标区域”
```

而不是：

```text
“最终停在目标附近并形成高 SR/SPL trajectory”
```

这是当前 RL 提升有限的主要机制性解释之一。

---

# 14. Navigation progress reward 设计过于粗糙

当前代码本质是：

```python
distance_delta = previous_distance - current_distance

if moved and distance_delta > epsilon:
    reward += 5
```

因此：

```text
靠近 0.01m → +5
靠近 3.00m → +5

走远 0.01m → 0
走远 3.00m → 0
```

这会鼓励：

```text
“这一动作是否比上一动作稍微靠近”
```

而不是：

```text
“整个 episode 最后是否真正靠近目标”
```

## 建议修改方向

优先考虑 continuous distance potential：

```python
distance_delta = previous_distance - current_distance
reward = progress_scale * distance_delta
```

这样：

```text
靠近很多 → 大正奖励
靠近一点 → 小正奖励
走远 → 负奖励
```

整条 episode：

```text
Σ(d_t - d_{t+1})
= d_start - d_final
```

天然和 final endpoint 对齐。

## 注意

不要直接修改后立刻跑长训练。

必须：

1. unit test；
2. synthetic reward validation；
3. 20~50 step smoke run；
4. 再做 reward alignment audit。

---

# 15. Semantic reward：方向尚可，但量级极弱

真实 reward family：

```text
navigation:
mean = -4.2838
std  = 94.6313

semantic:
mean = -0.0023
std  = 0.1229

thought:
mean = 16.0662
std  = 18.1300
```

Semantic：

```text
range ≈ [-0.439, +0.435]
```

当前 `semantic/alignment_delta` 在 clean all-fail 中：

```text
corr_final = 0.189329
corr_min   = 0.094865
pair_final = 57.73%
```

说明：

> semantic 的方向不是完全错误，它和最终质量存在一些正相关。

但是数值尺度太小，极易被 navigation / thought 淹没。

## 修复要求

不要简单无脑：

```text
semantic × 100
```

应该：

1. 分析 semantic 单独与 final-distance / success / SPL surrogate 的相关性；
2. 重新标定尺度；
3. 保证其不会压倒 navigation terminal signal；
4. 最终通过 smoke-run 的 group-level alignment 证明增益。

---

# 16. Thought reward 目前非常可疑

真实：

```text
thought mean = +16.0662
std = 18.1300
```

组件：

```text
thought/action_consistency
nonzero = 98.91%
mean = +11.3426

thought/subgoal_alignment
nonzero = 59.45%
mean = +5.6275

thought/fact_consistency
nonzero = 8.83%
mean = -0.9038
```

但是 clean all-fail：

```text
thought corr_final = 0.036438
pair_final         = 53.69%
```

即：

> Thought reward 对最终导航质量几乎没有线性相关性，但数值影响却很大。

---

# 17. Thought subgoal_alignment 逻辑存在目标错位

当前实现：

1. 把 action plan 拆成文本 subgoals；
2. 对 thought 做 CLIP embedding；
3. 找最相似 subgoal；
4. 若命中 expected subgoal index：`+5`；
5. `_next_thought_subgoal_index += 1`。

问题：

```text
_next_thought_subgoal_index
```

与真实机器人是否完成对应 subgoal 没有强绑定。

因此可能出现：

```text
Step 1:
嘴上描述 plan #1
→ +5

Step 2:
嘴上描述 plan #2
→ +5

但机器人实际没有完成对应物理 subgoal
```

## 修复方向

Thought subgoal reward 应考虑与真实 navigation state 绑定，例如：

```text
只有当：
- thought 对应当前目标 subgoal
AND
- 实际 trajectory / viewpoint progress 也满足该 subgoal condition
才奖励
```

如果没有可靠 subgoal metadata，则：

> 宁可暂时降低或禁用这部分 reward，也不要保留一个“奖励说得像但不奖励走得对”的高权重信号。

---

# 18. Thought action_consistency 过于容易获得

当前 action consistency 大量依赖语言 cue：

```text
move
proceed
continue
enter
turn
head toward
follow
walk
...
```

只要 thought 与自身 action 不明显矛盾，就可能获得奖励。

因此它主要优化：

```text
reasoning/action linguistic self-consistency
```

而不一定优化：

```text
navigation correctness
```

当前实证：

```text
nonzero ≈ 99%
mean ≈ +11.3
corr_final ≈ 0
```

说明这一项应该：

1. 降权；
2. 或重构；
3. 或从总 reward 中分离为较弱 auxiliary reward。

---

# 19. 已确认严重协议问题：tool_call_after_episode_end

canonical training：

```text
violating canonical rollouts
= 358 / 3000
= 11.93%
```

其中：

```text
invalid rollouts that had success reward = 33
```

也就是说至少 33 条 rollout：

```text
环境已经成功
↓
已经得到 navigation/success
↓
模型又调用一次 submit_navigation_decision
↓
记录 tool_call_after_episode_end
↓
finalize 时被 protocol invalidation
↓
成功 trajectory 被当成 invalid / truncated failure
```

---

# 20. 协议污染对 GRPO group 的影响更严重

完全 clean groups：

```text
475 / 750
= 63.33%
```

说明：

```text
275 / 750
= 36.67%
```

optimizer groups 至少包含一条 protocol-invalid rollout。

因此不能只看：

```text
11.93% rollout violation
```

GRPO 是 group-relative 学习。一条 rollout 被错误修改 reward，就可能改变整个 group 的 advantage 排序。

**这个问题必须在下一轮训练前修复。**

---

# 21. 已有 external-cutoff patch 不能解决这个问题

当前默认分支已有一个 `rl_env.py` patch，解决：

```text
TRL 在 max_tool_calling_iterations 边界
最后生成一个 pending tool call
但该 call 没有被执行
```

即允许合法的：

```text
native_tool_calls = executed_tool_calls + 1
```

但这不等于解决：

```text
tool_call_after_episode_end
```

两者必须区分。

不要把当前 patch 当作完整 protocol fix。

---

# 22. tool-after-end 建议修复方式

当前 terminal tool result 应明确告诉模型：

```text
Episode terminated/truncated.
DO NOT call submit_navigation_decision again.
End the conversation now.
```

非 terminal tool result 才返回：

```text
Call submit_navigation_decision with the next canonical decision.
```

同时应审查：

```text
format_trl_navigation_observation()
system/tool instruction
Qwen tool schema
TRL tool-loop termination behavior
```

不要仅靠字符串 prompt 修复。

必须确认：

> 当 environment 已 terminated/truncated 后，TRL / tool execution layer 不应再实际执行新 navigation decision。

---

# 23. 修复 protocol 后必须增加回归测试

至少覆盖：

## Case A：正常移动

```text
tool call
→ environment non-terminal
→ 可以继续下一 tool call
```

## Case B：成功 Finish

```text
submit_navigation_decision("<Action>Finish!</Action>")
→ terminated=True
→ success=True
→ conversation ends
→ 不再调用 tool
→ protocol_violations=[]
```

## Case C：premature Finish

```text
terminated=True
→ failure
→ conversation ends
→ 不允许后续 tool
```

## Case D：max_steps

```text
truncated=True
→ conversation ends
→ 不允许后续 tool
```

## Case E：TRL external cutoff

仍然保留已有合法 pending-tool-call 兼容逻辑。

---

# 24. Training success 与论文 SR 存在目标定义差异

训练 environment 的 `success=True` 需要：

```text
模型显式 Finish
AND
distance < 3m
```

而标准 R2R evaluation 的 SR：

```text
final trajectory endpoint distance < 3m
```

并不要求显式 Finish action。

这不一定是 bug，但属于 objective mismatch。

需要审查：

1. 是否应该保持显式 Stop/Finish 学习；
2. 是否应该另外增加 endpoint terminal shaping；
3. 是否需要把 stop correctness 与 R2R SR objective 做更清晰的权衡。

不要未经分析直接删除 Finish 机制。

---

# 25. 修复前不要直接进行 14,039-step 长训练

当前如果直接长训约 1 epoch，有可能只是把以下问题优化得更充分：

```text
tool protocol contamination

dead subgoal reward

dead landmark reward

weak semantic reward

over-strong thought reward

binary progress reward

train/full-eval protocol mismatch
```

因此正确顺序必须是：

```text
修代码
↓
unit tests
↓
synthetic tests
↓
20~50 step smoke training
↓
reward alignment audit
↓
small full/fast evaluation
↓
通过后才允许长训练
```

---

# 26. 修复后的 Reward Alignment 验收指标

当前 clean all-fail：

```text
reward vs final distance:

Pearson          = 0.231
pairwise         = 60.57%
winner agreement = 45.76%

reward vs minimum distance:

Pearson          = 0.478
pairwise         = 73.04%
winner agreement = 68.64%
```

希望修改后至少看到：

```text
reward vs final:
Pearson > 0.4

pairwise ranking accuracy > 70%

best-reward == best-final-distance > 60%
```

同时最重要：

```text
final-distance alignment
与
minimum-distance alignment
之间的差距应明显缩小
```

不能只继续提高 minimum-distance alignment。

---

# 27. Protocol 修复验收指标

当前：

```text
protocol violating rollouts = 11.93%

groups containing violation ≈ 36.67%
```

修复后目标：

```text
tool_call_after_episode_end ≈ 0

正常 terminal episode
不得产生 protocol violation

合法 TRL external cutoff
不得被误判为 violation
```

---

# 28. Reward family 修复后必须重新审计

必须重新统计：

```text
navigation mean/std
semantic mean/std
thought mean/std
```

并重新计算：

```text
corr_final
corr_min
pairwise final ranking
winner agreement
```

尤其检查 `thought`，不能继续出现：

```text
reward magnitude 很强
但 corr_final ≈ 0
```

---

# 29. 关于 reward_metadata 的修复要求

Codex 需要先分析：

```text
action plan cache
R2R annotation
path
candidate graph
planner output
```

是否能够可靠生成：

```text
subgoal_viewpoints
key_landmark_viewpoints
```

不要凭字符串猜 viewpoint。

如果没有可靠、可复现的自动构造方式：

## 方案 A

增加真正的数据预处理 pipeline：

```text
action plan / path
→ reward metadata generator
→ manifest + hashes
→ training annotation/cache
```

并保证：

```text
train / validation
可重复生成
有 provenance
有 unit tests
```

## 方案 B

如果暂时无法可靠生成：

明确：

```text
NavigationRewardConfig
subgoal_completion reward disabled
landmark_deviation disabled
```

不要继续保留“配置上启用但实际永远 0”的隐式死逻辑。

---

# 30. 最终 evaluator 必须统一

当前推荐：正式 RL evaluation 应优先使用：

```text
r2r_evaluation.py
ToolPolicyEpisodeRunner
```

即与训练一致的：

```text
Qwen native tool calling
```

需要建立一个正式 CLI / script，用于：

```text
Base Qwen full Val-Unseen
step375 full Val-Unseen
step750 full Val-Unseen
future full_best
```

所有组必须一致：

```text
same split
same action plan cache
same prompt/state builder
same native tool transport
same max_navigation_steps
same max_tool_calling_iterations
same max_new_tokens
same seed
same metrics implementation
```

只有：

```text
adapter_path
```

不同。

这样才能严格隔离 RL 增益。

---

# 31. Base / Ours 直接对照要求

真正的 RL ablation 必须是：

```text
Base Qwen2.5-14B
+ fixed action plan
+ native tool protocol
+ same evaluator
+ no adapter

VS

same Base Qwen2.5-14B
+ fixed action plan
+ native tool protocol
+ same evaluator
+ LoRA adapter
```

禁止混入：

```text
planner mode
LangChain transport
不同 max_new_tokens
不同 prompt
不同 action plan
```

---

# 32. 建议的代码审查顺序

请 Codex 按以下顺序审查。

## P0：Protocol

重点文件：

```text
nav_src/rl_env.py
nav_src/r2r_evaluation.py
nav_src/grpo_training.py
nav_src/grpo_runtime.py
```

检查：

```text
tool loop termination
terminal tool result
external cutoff handling
tool transcript validation
success invalidation
```

## P0：Evaluation consistency

重点：

```text
NavGPT.py
r2r_evaluation.py
LLMs/hf_chat.py
navigation_state.py
prompt/chat_prompt.py
```

输出：

```text
train
fast eval
native full eval
legacy NavGPT.py full eval
```

之间 prompt / generation / tool protocol 的逐项 diff。

## P1：Navigation reward

重点：

```text
navigation_rewards.py
```

审查：

```text
binary progress
distance potential
failure shaping
terminal reward
revisit
invalid streak
```

要求优先修正 final-distance alignment。

## P1：Thought reward

审查：

```text
subgoal_alignment
action_consistency
fact_consistency
```

明确哪些真正反映 navigation quality，哪些只是 language/style consistency。

## P1：Semantic reward

审查：

```text
CLIP feature construction
potential scale
episode telescoping
reward magnitude
group ranking contribution
```

## P1：Reward metadata

审查：

```text
annotation
action-plan cache
data preprocessing
reward_metadata generation
```

## P2：Validation lifecycle

审查：

```text
on_step_end
on_epoch_end
on_train_end
resume_pending
snapshot
selector
```

使 max-step run 也能最终产生：

```text
full_best
```

---

# 33. Codex 必须输出的审查结果

请不要只“直接改代码”。先输出一份明确审查报告。

## A. Confirmed bugs

每个 bug 给出：

```text
文件
函数
当前行为
为什么错误
如何复现
影响
```

## B. Design weaknesses

例如：

```text
progress shaping
thought reward
semantic scaling
```

要区分：

```text
bug
vs
design choice
```

## C. Proposed changes

逐文件列出：

```text
修改内容
兼容性影响
checkpoint compatibility
run manifest compatibility
test impact
```

## D. Tests

新增/修改哪些：

```text
unit tests
regression tests
smoke tests
```

---

# 34. 修改代码时的原则

不要：

```text
为了让指标好看而直接 hard-code
删除现有 provenance
绕过 run manifest
绕过 adapter validation
绕过 protocol validation
```

必须保留当前项目已经做得较好的：

```text
immutable evaluation snapshots
adapter provenance
base-model fingerprint
resume safety
checkpoint validation
group identity validation
reward logging
```

---

# 35. 特别注意本次训练 commit

本次训练科学身份：

```text
383f4b90ce1055e7894e4d2c68edfb55e604c7f0
+
训练中授权的 rl_env protocol patch
```

当前 GitHub 默认分支比该 commit 至少 ahead 2 commits。

因此任何修复都必须先：

```text
git diff / compare
```

确认：

```text
哪些代码属于训练时版本
哪些属于之后修复
```

不能用当前默认分支行为倒推训练时实际行为。

---

# 36. 推荐的修复阶段

## Phase 1：只修 protocol + evaluator consistency

先不要同时动全部 reward。

目标：

```text
tool_call_after_episode_end ≈ 0

native full evaluation 可用

Base / step375 / step750
全部通过统一 native evaluator
```

重新确认 RL gain。

## Phase 2：修改 navigation progress

从 binary：

```text
if distance_delta > 0:
    +5
```

改成经过充分测试的 continuous potential shaping。

然后跑：

```text
20~50 optimizer steps
```

再做 alignment audit。

## Phase 3：Thought / Semantic / Metadata

依次处理：

```text
thought weight / subgoal grounding
semantic scaling
reward_metadata
```

不要一次改五六个 reward 后再无法做 attribution。

---

# 37. Smoke run 之后必须重新生成的关键统计

至少：

```text
canonical rollouts
complete groups
protocol violation rate
group reward diversity
group path diversity
all-fail / mixed / all-success
reward family mean/std
reward component nonzero rate
success-vs-failure preference

clean all-fail:
reward vs final distance
reward vs minimum distance

navigation correlation
semantic correlation
thought correlation
```

---

# 38. 当前结论总结

当前不能得出：

```text
“GRPO 根本没学到”
```

因为：

```text
group reward diversity 很高
trajectory diversity 很高
step750 相比 base 有小幅正增益
```

也不能简单得出：

```text
“只是训练步数太少”
```

因为已经确认多个实际问题：

```text
1. train/fast eval 与当前 full eval tool protocol 不一致

2. max_steps run 不会自动 full validation

3. reward_metadata 完全缺失

4. subgoal / landmark reward 永远为 0

5. semantic reward 数值极弱

6. thought reward 很强但与 final quality 几乎无关

7. progress reward 只奖励“变近”，不按距离变化连续 shaping

8. reward 对 minimum distance 的 alignment
   明显强于 final distance

9. 11.93% rollout 存在 tool_call_after_episode_end

10. 36.67% GRPO group 至少含一个 protocol-invalid rollout

11. 至少 33 条原本已经有 success reward 的 rollout
    被后续 tool-after-end 污染
```

当前最合理的解释是：

> RL 确实在学习，但训练目标、dense reward 和执行协议存在明显错位，因此它更容易优化“轨迹过程中曾经接近目标、reasoning 看起来合理”，而没有充分优化“最终 endpoint 成功、SR/SPL 更高”。

---

# 39. 当前最优先行动

严格按以下顺序：

```text
P0
修 tool_call_after_episode_end

P0
统一 native-tool full evaluation

P1
改 progress shaping，使其更对齐 final distance

P1
重新评估 thought reward 权重和定义

P1
处理 semantic scale

P1
接通或显式关闭 reward_metadata 相关 reward

P2
max-step train end 自动 full validation
```

完成以上修复后：

```text
20~50 step smoke training
↓
reward alignment audit
↓
native full/fast validation
↓
通过验收
↓
再决定是否跑 750 / 1500 / 1 epoch / 2 epochs
```

---

# 40. 不应立即做的事情

暂时不要：

```text
直接跑 14,039 step
直接把 semantic ×100
直接删除 Thought reward
直接删除 Finish success
继续调 max_new_tokens
继续使用旧 NavGPT.py 作为唯一正式 RL full evaluator
因为 step750 有一点提升就认为问题已经解决
```

先把上述链路修正确，再重新训练。

---

# 41. 给 Codex 的最终任务

请对仓库执行以下工作：

1. **完整审查**上述问题；
2. 对每一项给出代码证据；
3. 区分 bug / design weakness；
4. 先提出修改方案；
5. 修改 P0 问题；
6. 增加 regression tests；
7. 修复 reward 时保持可解释、可归因；
8. 给出修复后 smoke-run 命令；
9. 给出修复后 reward-alignment 检查命令；
10. 不要直接启动正式长训练；
11. 不要破坏现有 checkpoint / provenance / resume 机制；
12. 所有变更必须可通过 git diff 清楚审阅。

优先保证：

```text
训练协议正确
reward 优化目标正确
评测协议一致
实验具有可解释性与可复现性
```

而不是优先追求“马上把 SR 跑高”。
