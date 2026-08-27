# 3 Method

We propose a parameter-efficient reinforcement-learning framework for text-based vision-and-language navigation. It combines a frozen instruction planner, a local language policy adapted with LoRA, and a training-only reward for navigation behavior, visual grounding, and reasoning quality. The policy learns through closed-loop R2R interaction and uses the same structured interface at inference, when the reward modules and RL optimizer are removed.

## 3.1 Problem Formulation

We represent an R2R environment as a navigation graph \(\mathcal{G}=(\mathcal{V},\mathcal{E})\), where each node is a Matterport3D viewpoint and each edge is a traversable connection (Anderson et al., 2018). A task contains an instruction \(x\), an initial pose \((v_0,\psi_0,\phi_0)\), and a target viewpoint \(v^*\). At step \(t\), the agent observes a textual panorama \(o_t\) at \(v_t\) and may move to an adjacent node in \(\mathcal{N}(v_t)\) or stop. The target and reference path remain inside the environment for reward computation and evaluation and are never exposed to the policy.

A frozen planner first decomposes \(x\) into an action plan \(p\). Before step \(t\), the environment has recorded the ordered interaction history

\[
\tau_{<t}=\big((c_0,a_0,o_1),\ldots,(c_{t-1},a_{t-1},o_t)\big),
\qquad
s_t=\mathcal{F}(p,o_0,\tau_{<t},o_t).
\]

Here, \(c_k\) is the generated thought, \(a_k\) is the executed action, and \(o_{k+1}\) is the observation returned after that action. Thus, \(\tau_{<t}\) is the historical trajectory before the current decision. The formatter \(\mathcal{F}\) combines this history with the plan and observations, and the policy \(\pi_\theta\) generates \(y_t=(c_t,a_t)\). An episode succeeds only when the agent chooses to stop within 3 m shortest-path distance of \(v^*\). Passing through the goal region counts only as oracle success. Each episode is limited to 10 decisions.

## 3.2 Frozen Planning and Structured Policy

The planner decomposes an instruction once, while the policy repeatedly grounds decisions in the current state. We use an unadapted Qwen2.5-14B-Instruct-1M planner (Qwen Team, 2024) with temperature 0. It receives only the instruction and produces ordered textual subgoals; neither the target nor reference trajectory is provided. Plans are cached by instruction ID, with hashes and model/decoding provenance ensuring that same-task rollouts receive the same plan.

The policy operates on a structured textual state. Each panorama is divided into eight egocentric sectors containing a scene description, nearby objects, and the identifiers, distances, and relative directions of navigable viewpoints. The prompt combines the plan, initial panorama, and a bounded history of previous decisions and observations. Raw RGB is not appended to the policy prompt; it is used only by the semantic reward in Section 3.4.

At every decision step, the policy must produce exactly one non-empty `<Think>` block followed by one `<Action>` block, with no additional text. The action either selects an adjacent candidate viewpoint or stops the episode. A shared strict parser rejects malformed blocks and nonexistent viewpoint identifiers. Invalid decisions leave the agent in place and return the current observation and valid neighboring candidates.

## 3.3 Stateful Navigation Rollouts

We formulate navigation as a stateful reinforcement-learning environment. At each turn, it validates one thought--action decision, executes the transition, updates history, retrieves the raw-visual feature, and returns the next textual observation. Each rollout uses an isolated environment instance.

GRPO group members share the instruction, plan, start pose, and initial prompt; only policy sampling differs. Stopping ends the episode, while the 10-step limit causes truncation. Unparseable or length-truncated outputs are treated as unsuccessful. The environment alone accumulates step rewards and provides one episode return, preventing duplicate feedback.

## 3.4 Composite Navigation Reward

For transition \(t\), we combine navigation, raw-visual semantic, and thought-quality feedback:

\[
r_t=\lambda_{\mathrm{nav}}r_t^{\mathrm{nav}}
    +\lambda_{\mathrm{sem}}r_t^{\mathrm{sem}}
    +\lambda_{\mathrm{th}}r_t^{\mathrm{th}},
\qquad
r_t^{\mathrm{th}}=r_t^{\mathrm{sub}}+r_t^{\mathrm{act}}+r_t^{\mathrm{fact}}.
\]

All weights equal 1 in the full model and can be disabled independently for ablation.

### Navigation reward

The navigation term is

\[
r_t^{\mathrm{nav}}
=r_t^{\mathrm{prog}}+r_t^{\mathrm{rev}}+r_t^{\mathrm{inv}}+r_t^{\mathrm{term}}.
\]

Let \(d_t\) be the shortest-path distance to the target. We define the navigation potential and progress reward as

\[
\Phi_{\mathrm{nav}}(s_t)=-5d_t,
\qquad
r_t^{\mathrm{prog}}
=\Phi_{\mathrm{nav}}(s_t)-\Phi_{\mathrm{nav}}(s_{t-1})
=5(d_{t-1}-d_t).
\]

The progress term is applied only to executed movements: approaching the goal receives a positive reward proportional to the distance reduction, while moving away receives a symmetric negative reward. It is not clipped or thresholded, so its episode sum telescopes to \(5(d_0-d_T)\) and is aligned with the final endpoint rather than the minimum distance visited. Entering an already visited viewpoint additionally gives \(r_t^{\mathrm{rev}}=-10\), and each non-moving attempt after the consecutive invalid count reaches three gives \(r_t^{\mathrm{inv}}=-20\). The other components are zero when their conditions are not met. Correct active stopping gives \(r_t^{\mathrm{term}}=+200\). The return of an unsuccessful episode is defined separately below so that such trajectories remain distinguishable without receiving an excessive score.

### Raw-visual semantic reward

To recover visual evidence omitted by captions, we construct a training-only signal from Matterport3D RGB views. With CLIP ViT-L/14 (Radford et al., 2021), we precompute a normalized instruction embedding \(u=E_T(x)\) and normalized image embeddings for 36 views at each viewpoint: 12 headings at three elevations. At runtime, \(z_t\) is the feature nearest to the current orientation. We use the potential difference

\[
\Phi(s_t)=4u^\top z_t,
\qquad
r_t^{\mathrm{sem}}=\Phi(s_t)-\Phi(s_{t-1}).
\]

Because cosine similarity is bounded, \(r_t^{\mathrm{sem}}\in[-8,8]\). Its cumulative value is zero when a trajectory returns to the same discretized pose, preventing repeated semantic gain from cycling through the same states. This reward is computed exclusively from raw-image features, never from substituted caption embeddings.

### Thought-quality reward

Thought quality is implemented as the versioned conservative auxiliary protocol `grounded_auxiliary_v1`, with \(\lambda_{\mathrm{th}}=0.25\). It retains text--plan alignment as a diagnostic, but does not turn it into reward because the production data do not provide a versioned mapping from each textual plan line to a physically reached viewpoint. For plan lines \(p_1,\ldots,p_M\) and thought \(c_t\), we log

\[
j_t^*=\arg\max_j\cos(E_T(c_t),E_T(p_j)),
\qquad
q_t^{\mathrm{sub}}=\mathbf{1}
\left[\cos(E_T(c_t),E_T(p_{j_t^*}))\geq0.25\right],
\qquad
r_t^{\mathrm{sub}}=0.
\]

Thus, plausible plan paraphrasing cannot be rewarded without physical progress. Thought--action agreement is defined by the nominal score

\[
\widetilde r_t^{\mathrm{act}}=
\begin{cases}
+5, & c_t\text{ exactly supports an executed, grounded }a_t,\\
-8, & c_t\text{ contradicts }a_t\text{ or the decision is invalid},\\
0, & \text{otherwise}.
\end{cases}
\]

Positive support requires an environment-confirmed successful Finish, or an actually executed move/backtrack whose unique direction or viewpoint identifier exactly matches the thought. Generic phrases such as “move”, “proceed”, or “continue”, an ambiguous set of directions, an unexecuted action, and an unconfirmed Finish claim receive zero. Explicitly conflicting actions, directions, or viewpoint identifiers retain the negative score. For factual consistency, let \(e_t=(o_t,\mathcal{O}_t,\tau_{<t})\) collect the available observation text, visible-object set \(\mathcal{O}_t\), and recent history. Let \(U(c_t,e_t)\) indicate an explicit present-tense visual claim in \(c_t\) that is absent from this evidence. For a valid decision, the nominal score is

\[
\widetilde r_t^{\mathrm{fact}}=-8\,\mathbf{1}[U(c_t,e_t)].
\]

The reward contribution is \(\lambda_{\mathrm{th}}(r_t^{\mathrm{sub}}+\widetilde r_t^{\mathrm{act}}+\widetilde r_t^{\mathrm{fact}})\), so a grounded positive contributes \(+1.25\) and a contradiction contributes \(-2\). We log CLIP similarities for analysis but do not use low similarity alone as a contradiction: different wording can express the same valid reasoning. Parse errors and invalid actions receive the scaled action-consistency penalty. The protocol name, weight, diagnostic-only subgoal mode, and component magnitudes are part of the immutable run identity.

### Episode return

For an unsuccessful trajectory, we first sum all non-terminal components:

\[
R_{\mathrm{dense}}=\sum_{t=1}^{T}\left[
\lambda_{\mathrm{nav}}(r_t^{\mathrm{prog}}+r_t^{\mathrm{rev}}+r_t^{\mathrm{inv}})
+\lambda_{\mathrm{sem}}r_t^{\mathrm{sem}}
+\lambda_{\mathrm{th}}(r_t^{\mathrm{sub}}+r_t^{\mathrm{act}}+r_t^{\mathrm{fact}})
\right].
\]

This value distinguishes, for example, a trajectory that approaches the target from one dominated by revisits and invalid actions. We introduce the bounded, monotonic mapping

\[
R_i=
\begin{cases}
\sum_{t=1}^{T}r_t, & \text{if trajectory }i\text{ succeeds},\\
F-\Delta+\Delta\tanh\!\left(R_{\mathrm{dense}}/T_f\right),
& \text{otherwise}.
\end{cases}
\]

Here, \(F\) is the unsuccessful-return ceiling, \(\Delta\) controls its range, and \(T_f\) controls sensitivity to dense rewards. We use \(F=-80\), \(\Delta=20\), and \(T_f=100\), selected according to the scale of the step rewards. Consequently, unsuccessful returns lie in \((F-2\Delta,F)=(-120,-80)\): better partial behavior receives a higher score, but never crosses the unsuccessful ceiling. The resulting \(R_i\) is the trajectory's final scalar reward used to compute the group-relative advantage in Section 3.6; it is not added as another step reward.

## 3.5 LoRA Navigation Policy

We adapt Qwen2.5-14B-Instruct-1M using LoRA (Hu et al., 2022). For a frozen projection \(W_0\), its output becomes

\[
h=W_0x+\frac{\alpha}{r}BAx,
\]

with rank \(r=16\) and scale \(\alpha=32\). We adapt the query, key, value, attention-output, gate, up, and down projections in all 48 Transformer blocks, yielding 68,812,800 trainable parameters while freezing the backbone. Zero initialization of \(B\) makes the initial policy identical to the base model. Training uses BF16 and gradient checkpointing; LoRA dropout is disabled during GRPO likelihood computation to keep policy, old-policy, and reference scores consistent.

## 3.6 Group-Relative Policy Optimization

For each task, we sample \(G=4\) trajectories from the same initial condition. Given finalized returns \(R_i\), GRPO computes the within-group advantage (Shao et al., 2024)

\[
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon},
\qquad
\mu_R=\frac{1}{G}\sum_{i=1}^{G}R_i.
\]

This normalization compares alternative behavior on the same task rather than confounding reward with instruction difficulty. We optimize the clipped token-level GRPO objective with KL coefficient \(\beta=0.001\) against the frozen initial policy. No critic or value network is required. Every policy-generated `<Think>` and `<Action>` token in a trajectory shares its episode advantage, directly coupling generated reasoning and executed navigation outcomes. Unsuccessful and truncated trajectories remain in the loss.

We sample with temperature 1.0 and top-\(p=0.95\), using one four-trajectory group per optimizer step. For acceleration, PyTorch Distributed Data Parallel (DDP) with NCCL evenly divides the group across available GPUs: one GPU generates four members, two generate two each, and four generate one each. Only LoRA gradients are synchronized, so multi-GPU execution preserves the group and effective optimization batch.

## 3.7 Training and Inference

Training starts from the identity adapter without expert action or reasoning supervision. The reference trajectory specifies the start and target, and R2R shortest-path distances are used only for reward and evaluation, never as action labels. Checkpoints omit the frozen 14B weights. At inference, the selected adapter is attached to the base model; the planner and navigation interface remain, while CLIP rewards, thought scorers, and GRPO are removed.

# References

1. Anderson, P., Wu, Q., Teney, D., Bruce, J., Johnson, M., Sünderhauf, N., Reid, I., Gould, S., and van den Hengel, A. Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3674–3683, 2018.
2. Qwen Team. Qwen2.5 Technical Report. *arXiv preprint arXiv:2412.15115*, 2024.
3. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning Transferable Visual Models from Natural Language Supervision. *Proceedings of the International Conference on Machine Learning (ICML)*, PMLR 139:8748–8763, 2021.
4. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR)*, 2022.
5. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv preprint arXiv:2402.03300*, 2024.
