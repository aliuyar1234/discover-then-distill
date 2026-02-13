# CODEX HANDOFF: LLM from scratch (mHC + SDFT + TTT-Discover)

**Build date (UTC): 2026-02-02**

This document is the *single source of truth* for how to build the project end-to-end.
It is written so an automated coding agent can execute it phase-by-phase **without asking questions**.
If you are an agent: follow instructions literally; do not invent missing pieces.

## Agent Contract (anti-hallucination / anti-drift)

You (the implementing agent) MUST follow these rules:

1. **No invention.** If a value, formula, or file path is not specified here, you must:
   - search within this repository first (`ripgrep`, `find`, `ls`, open files), and
   - if still missing, consult the PDFs in `docs/references/`,
   - if still missing, implement a *minimal placeholder* with a loud `TODO(NEEDS_SPEC)` and a failing unit test that states exactly what is missing.
   - Do **not** silently “choose something reasonable” unless the document explicitly gives you a “default choice” for that situation.

2. **Everything is validated.** Every phase ends with:
   - at least one automated test,
   - and at least one end-to-end run (even if tiny).

3. **Reproducibility over cleverness.**
   - pin versions where possible,
   - avoid magic constants outside configs,
   - prefer simple, readable code over micro-optimizations.

4. **Single GPU assumption.**
   - Treat the machine as a single workstation with a single large NVIDIA GPU (96GB VRAM).
   - Do not design for multi-node distributed training unless a later “Scaling” phase instructs it.

5. **Explicit artifacts.** Every phase produces well-defined artifacts (files) that are committed or saved.

6. **No background tasks.** Do not assume anything runs “later”. If something must run, add it as a command in a phase.

If you ever detect a mismatch between this handoff and the repo code, **the handoff wins**: update the code to match the handoff, and add a regression test to prevent reintroducing the mismatch.

## Repository map (what is already scaffolded)

- `README.md` – short overview
- `docs/CODEX_HANDOFF.md` – THIS file (long blueprint)
- `docs/references/` – the three source papers (PDF):
  - `mHC - Manifold-Constrained Hyper-Connections.pdf`
  - `Self-Distillation Enables Continual Learning (SDFT) - 2601.19897v1.pdf`
  - `Learning to Discover at Test Time (TTT-Discover).pdf`

- `src/llm_mhc_sdft_tttd/` – minimal PyTorch implementation:
  - `config.py` – explicit configs (ModelConfig, PretrainConfig, SDFTConfig, TTTDiscoverConfig)
  - `model/` – transformer + mHC + LoRA
  - `training/` – training loops (pretrain, sdft, ttt_discover)
  - `data/` – tokenizer wrapper + dataset formats
  - `eval/` – perplexity

- `scripts/` – CLI entry points (plus `_bootstrap.py` to run without editable install)
- `configs/` – model JSON configs (120M / 350M / 900M)

- `tests/` – unit tests (sinkhorn, model shapes, SDFT collate, padded-prompt generation)

This repository is not “production ready”. It is a blueprint + reference implementation meant to be extended.

## Non-negotiable decisions (frozen design choices)

These decisions are *deliberately* fixed so you can build end-to-end without ambiguity.

### A. Base LM architecture (dense decoder-only)
1. **Architecture family:** LLaMA-style decoder-only Transformer (pre-norm, RMSNorm, RoPE, SwiGLU MLP).
2. **Attention:** standard multi-head causal self-attention using `torch.nn.functional.scaled_dot_product_attention` (SDPA).
3. **Tokenizer:** SentencePiece BPE (default 32k vocab).
4. **Training objective:** standard next-token cross-entropy on packed tokens.
5. **Precision:** bf16 (if supported).

### B. mHC integration (from the mHC paper)
1. **Residual stream expansion rate:** n = 4.
2. **Per-sublayer mHC:** mHC residual update is applied **twice** per block:
   - once around Attention
   - once around MLP
3. **H_res constraint:** H_res is projected to the **Birkhoff polytope** (doubly-stochastic) using Sinkhorn-Knopp with t_max=20.
4. **Dynamic + static mappings:** use Eq. (7) + Eq. (8) from the mHC paper:
   - H~ = α (x_norm φ) + b
   - H_pre = sigmoid(H~_pre)
   - H_post = 2·sigmoid(H~_post)
   - H_res = Sinkhorn-Knopp(H~_res)
5. **Initialization choices (explicit):**
   - H_pre starts uniform and sums to ~1 via bias init to logit(1/n).
   - H_post starts uniform and sums to ~1 via bias init to logit(1/(2n)).
   - H_res starts approx-identity via diagonal-positive/off-diagonal-negative bias init before Sinkhorn.
   - α_pre, α_post, α_res all start at 0.01.

### C. Continual learning (SDFT paper)
1. **Core method:** Self-Distillation Fine-Tuning (SDFT) = minimize reverse KL:
   - D_KL( πθ(·|x) || π(·|x,c) )
   where the “teacher” π(·|x,c) is the same model conditioned on a demonstration c.
2. **Teacher:** Exponential Moving Average (EMA) of the student weights.
3. **On-policy trajectories:** sample y from student πθ(·|x) (not from teacher).
4. **Loss implementation:** token-level reverse-KL (analytic, full-vocab) along the student-sampled trajectory.

### D. Test-time training (TTT-Discover paper)
1. **Environment:** each problem induces a continuous reward function R(s) and transition T(a) via parsing action -> state.
2. **Objective:** entropic utility objective (Eq. in paper):
   - J_β(θ) = E_{s ~ reuse(H)} [ log E_{a~πθ(.|s)} exp(β(s) R(s,a)) ].
3. **Gradient estimator:** weighted policy gradient with entropic weights w_β; baseline -1; plus KL shaping term:
   - A(a;s) = w_β(a;s) - 1 - λ log ( πθ(a|s) / πθ0(a|s) ).
4. **Adaptive β(s):** choose β(s) per start state by constraining KL(q_β || u) = γ, with γ = ln(2).
5. **Reuse:** PUCT-inspired start-state selection from archive (score formula in paper).
6. **Parameter updates at test time:** **LoRA-only** (base weights frozen).

### E. Combined system policy
1. **Pretraining** produces the base checkpoint `θ0`.
2. **TTT-Discover** produces per-problem LoRA adapters `Δθ_problem` + a buffer of discovered states.
3. **SDFT consolidation** uses discovered states as demonstrations to update a *continual* student `θ_continual`, while mitigating catastrophic forgetting.
4. The system NEVER writes test-time LoRA deltas into the base model directly; only through controlled consolidation (SDFT).

These choices can be changed later, but only by editing `docs/CODEX_HANDOFF.md` AND updating configs + tests.

## Papers: what we are extracting and why (high-level)

This project combines **three distinct mechanisms**:

1) **mHC (Manifold-Constrained Hyper-Connections)**  
   *Goal:* make very deep transformers easier to train by improving gradient flow and expressivity via expanded residual streams and constrained mixing.  
   *Key mathematical objects:* H_pre, H_post, H_res; Sinkhorn-Knopp projection onto doubly-stochastic matrices; identity mapping property.

2) **SDFT (Self-Distillation Enables Continual Learning)**  
   *Goal:* continual learning with minimal forgetting by distilling from the model’s own in-context “teacher” behavior.  
   *Key mathematical objects:* reverse-KL distillation objective; on-policy sampling; EMA teacher; interpretation as inverse RL.

3) **TTT-Discover (Learning to Discover at Test Time)**  
   *Goal:* for a *single hard instance*, run online RL at test time with an objective + reuse strategy that prioritize the *best* solution (max), not average reward.  
   *Key mathematical objects:* entropic utility objective; adaptive β via KL constraint on weights; PUCT archive reuse.

These methods are compatible because they operate on different axes:
- mHC = architecture / optimization landscape at training time.
- SDFT = how to update weights across time/tasks (continual learning).
- TTT-Discover = how to adapt quickly to a new *single* instance at inference time.

The integration challenge is **stability**: we must avoid test-time adaptation corrupting the base policy. That is why we:
- restrict TTT to LoRA adapters,
- and use SDFT to consolidate only after evaluation gates.

# mHC: mathematical specification (implement exactly)

This section specifies **exactly** how to implement mHC in code.

## 1. Baseline residual connection (standard Transformer)

For a residual stream vector \(x_l \in \mathbb{R}^{C}\), a standard residual layer is:

\[
x_{l+1} = x_l + F(x_l; W_l)
\]

where:
- \(F(\cdot)\) is the transformation implemented by the layer (e.g., attention or MLP)
- \(W_l\) are that layer’s parameters.

## 2. Hyper-Connections (HC): expand residual stream

HC expands the residual stream by a factor \(n\) (called the expansion rate).

Represent the stream as:

\[
X_l \in \mathbb{R}^{n \times C}
\]

You can think of \(X_l\) as **n parallel residual streams**.

HC introduces three learnable mappings per layer:

- \(H^{\text{pre}}_l \in \mathbb{R}^{1 \times n}\)  (aggregation into the transformation)
- \(H^{\text{post}}_l \in \mathbb{R}^{1 \times n}\) (distribution of layer output back into streams)
- \(H^{\text{res}}_l \in \mathbb{R}^{n \times n}\)  (mixing on the residual path)

The HC update (Eq. (3) in the mHC paper) is:

\[
X_{l+1} = H^{\text{res}}_l X_l \;+\; (H^{\text{post}}_l)^{\top} \; F\big( H^{\text{pre}}_l X_l; W_l \big)
\]

### 2.1 Shapes and implementation mapping

Assume batch and sequence exist but are omitted in math.

- \(X_l\): shape \([n, C]\)
- \(H^{\text{pre}}_l\): \([1, n]\)
- \(H^{\text{pre}}_l X_l\): \([1, C]\)  (weighted sum over streams)
- \(F(\cdot)\): maps \([1, C] \to [1, C]\)
- \((H^{\text{post}}_l)^{\top} F(\cdot)\): \([n, 1] \times [1, C] = [n, C]\)
- \(H^{\text{res}}_l X_l\): \([n, n] \times [n, C] = [n, C]\)

In the code, we represent \(X_l\) as `[B, T, n, C]`.

Aggregation is:

```python
x_in = einsum("btn,btnc->btc", H_pre, X)  # [B,T,C]
```

Residual mixing:

```python
res = einsum("btnm,btmc->btnc", H_res, X)  # [B,T,n,C]
```

Distribution:

```python
upd = einsum("btn,btc->btnc", H_post, y)  # [B,T,n,C]
```

Final:

```python
X_next = res + upd
```

## 3. Identity mapping property & the manifold constraint

HC can improve optimization, but if \(H^{\text{res}}\) is unconstrained it may break the desirable property
that deep networks can represent the identity mapping easily.

The mHC paper enforces an **identity mapping property** by constraining:

\[
H^{\text{res}}_l \in \mathcal{B}
\]

where \(\mathcal{B}\) is the **Birkhoff polytope** = the set of **doubly-stochastic matrices**:

- all entries non-negative
- each row sums to 1
- each column sums to 1

Key implication: if \(H^{\text{res}}\) is doubly-stochastic, the mean across streams is preserved.

Define the mean projection:

\[
i(X_l) = \frac{1}{n} \mathbf{1}^{\top} X_l \in \mathbb{R}^{1 \times C}
\]

Then doubly-stochastic \(H^{\text{res}}\) preserves this mean (sketch):
\(\mathbf{1}^{\top} H^{\text{res}} = \mathbf{1}^{\top}\) and \(H^{\text{res}}\mathbf{1} = \mathbf{1}\).

## 4. mHC: how to compute (H_pre, H_post, H_res)

mHC uses both **dynamic** (input-dependent) and **static** (bias) components, with a small gating scalar α.
This is Eq. (7) in the paper.

We implement it per token position.

### 4.1 Flattened residual stream representation

Flatten the stream dimension:

\[
\bar{x}_l \in \mathbb{R}^{1 \times (nC)}
\]

In code:

```python
x_flat = X.reshape(B, T, n*C)
x_norm = RMSNorm(x_flat)
```

### 4.2 Dynamic projections (φ)

Learnable matrices:

- \(\phi^{\text{pre}}_l \in \mathbb{R}^{nC \times n}\)
- \(\phi^{\text{post}}_l \in \mathbb{R}^{nC \times n}\)
- \(\phi^{\text{res}}_l \in \mathbb{R}^{nC \times n^2}\)

(We implement these as `nn.Linear(nC, n)` etc.)

Compute intermediate (unconstrained) mappings:

\[
\tilde{H}^{\text{pre}}_l = \alpha^{\text{pre}}_l (\bar{x}'_l \phi^{\text{pre}}_l) + b^{\text{pre}}_l
\]
\[
\tilde{H}^{\text{post}}_l = \alpha^{\text{post}}_l (\bar{x}'_l \phi^{\text{post}}_l) + b^{\text{post}}_l
\]
\[
\tilde{H}^{\text{res}}_l = \alpha^{\text{res}}_l \text{mat}(\bar{x}'_l \phi^{\text{res}}_l) + b^{\text{res}}_l
\]

Where `mat` reshapes a length \(n^2\) vector into an \(n\times n\) matrix.

### 4.3 Final constrained mappings (Eq. (8))

\[
H^{\text{pre}}_l = \sigma(\tilde{H}^{\text{pre}}_l)
\]
\[
H^{\text{post}}_l = 2\sigma(\tilde{H}^{\text{post}}_l)
\]
\[
H^{\text{res}}_l = \text{Sinkhorn-Knopp}(\tilde{H}^{\text{res}}_l)
\]

We implement `Sinkhorn-Knopp` to output a doubly-stochastic matrix.

## 5. Sinkhorn-Knopp projection (Eq. (9))

Given an unconstrained matrix \(\tilde{H}^{\text{res}}\), compute:

\[
M^{(0)} = \exp(\tilde{H}^{\text{res}})
\]
then iterate:
\[
M^{(t)} = T_r( T_c( M^{(t-1)} ) )
\]

- \(T_c\) normalizes each column to sum to 1
- \(T_r\) normalizes each row to sum to 1

After \(t_{\max}\) iterations:

\[
H^{\text{res}} = M^{(t_{\max})}
\]

### Numerical notes

- clamp \(\tilde{H}^{\text{res}}\) before `exp` to avoid overflow
- add epsilon to denominators.

## 6. Our implementation choices (explicit)

Because the paper does not fully specify bias initialization, we freeze explicit initializations:

- `b_pre = logit(1/n)` so `sigmoid(b_pre)=1/n`.
- `b_post = logit(1/(2n))` so `2*sigmoid(b_post)=1/n`.
- `b_res` diagonal = +2, off-diagonal = -2 (then Sinkhorn produces an approx-identity matrix).

This is implemented in `src/llm_mhc_sdft_tttd/model/mhc.py`.

## 7. Where mHC is applied in the Transformer

A standard GPT block has **two residual adds**:
1) around Attention
2) around MLP

We apply mHC to both, so the expanded stream \(X\) persists across the entire network.

Implementation location:
- `src/llm_mhc_sdft_tttd/model/transformer.py`
  - `MHCTransformerBlock`
  - `AttentionSublayer` + `MLPSublayer`
  - each wrapped by `MHCResidual`

## 8. Known failure modes + fixes

1) **Sinkhorn instability (NaNs)**
   - clamp input before exp
   - increase eps
   - reduce α_init further (e.g., 0.001)
2) **Model outputs explode**
   - check that H_pre/H_post biases are set correctly (uniform sum ~1)
   - reduce learning rate
   - ensure RMSNorm eps is not too small
3) **Training slower than baseline**
   - expected: mHC adds overhead
   - reduce n_streams (try n=2 for debugging)

# SDFT: continual learning specification (implement exactly)

This section is derived from the SDFT paper in `docs/references/`.
We implement the *core* algorithm (reverse KL distillation from a demonstration-conditioned teacher).

## 1. Problem framing

We have prompts/questions \(x\) and optional demonstrations \(c\) (examples of good responses).

- Student policy: \(\pi_\theta(y \mid x)\)
- Teacher policy: \(\pi(y \mid x, c)\)

In SDFT, the “teacher” is **the same model class**, but conditioned on the demonstration \(c\).
In practice:
- teacher = EMA copy of the student weights,
- teacher prompt includes both question and demonstration.

## 2. Objective: reverse KL distillation

The SDFT objective is the reverse KL:

\[
\mathcal{L}(\theta) = D_{KL}\big(\pi_\theta(\cdot\mid x)\;||\;\pi(\cdot\mid x,c)\big)
\]

Expanded:

\[
\mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot\mid x)}\left[\log \pi_\theta(y\mid x) - \log \pi(y\mid x,c)\right]
\]

Key properties:
- **On-policy:** expectation is under the student policy.
- **No reward model:** teacher is implicit “reward” via in-context conditioning.

## 3. Prompt template (teacher conditioning)

The paper provides an explicit template (we implement it verbatim):

```
<Question>
{prompt}

This is an example for a response to the question:
<Demonstration>
{demonstration}

Now answer with a response of your own, including the thinking process:
```

Implementation: `src/llm_mhc_sdft_tttd/data/sdft_dataset.py::make_teacher_prompt`.

## 4. Teacher construction: EMA

Teacher parameters are updated as an exponential moving average:

\[
\theta_{\text{ema}} \leftarrow \tau \theta_{\text{ema}} + (1-\tau)\theta
\]

We expose \(\tau\) as `teacher_ema_decay` (default 0.999).

## 5. Loss implementation: token-level analytic reverse KL

We use the token-level analytic KL between the student and teacher distributions along the student-sampled trajectory.

Given a sampled sequence \(y = (y_1,\dots,y_T)\), the token-level decomposition is:

\[
D_{KL}\big(\pi_\theta(\cdot\mid x)\;||\;\pi(\cdot\mid x,c)\big)
= \sum_{t=1}^{T} D_{KL}\Big(\pi_\theta(\cdot\mid y_{<t},x)\;||\;\pi(\cdot\mid y_{<t},x,c)\Big)
\]

And each token-level KL is:

\[
D_{KL}(p_s || p_t) = \sum_{v \in \mathcal{V}} p_s(v)\left[\log p_s(v) - \log p_t(v)\right]
\]

where:
- \(p_s(v)=\pi_\theta(v\mid y_{<t},x)\)
- \(p_t(v)=\pi(v\mid y_{<t},x,c)\)

### Why analytic KL?
The SDFT paper reports that the full-vocab analytic estimator is stable and performs best among estimators (see their appendix on estimators).

In code:
- compute `log_softmax` for student and teacher logits
- compute `p_s = exp(log_p_s)`
- compute `KL = sum_v p_s * (log_p_s - log_p_t)`

Implementation: `src/llm_mhc_sdft_tttd/training/sdft.py::compute_reverse_kl`.

## 6. On-policy sampling

Algorithm step:
1) given prompt \(x\), sample \(y \sim \pi_\theta(\cdot|x)\)
2) evaluate teacher distribution on the same prefix \(y_{<t}\) but with augmented context (x,c)
3) update θ to reduce reverse KL

This is implemented in `train_sdft()`:
- `model.generate()` samples y
- we re-run forward passes on:
  - student context + y
  - teacher context + y

## 7. Dataset format (decision)

We define a minimal jsonl dataset format for continual learning:

Each line is:
```json
{"prompt": "...", "demonstration": "..."}
```

- prompt: the query/task
- demonstration: an example of a good response

This format is used by `SDFTJsonlDataset`.

## 8. Replay (optional but recommended)

SDFT reduces catastrophic forgetting, but in practice you still want replay:
- store a buffer of past examples (prompts + demos)
- mix replay examples into the continual learning stream

Decision:
- `replay_ratio = 0.2` (20% replay batches)
- buffer size default 50k

NOTE: The reference implementation does not implement replay yet; it is specified in Phase 8.

## 9. Evaluation gates (must implement)

Before accepting a continual update, run:
- perplexity on a stable text set
- capability probes (unit tests / eval prompts)
- regression check: ensure key metrics do not degrade beyond threshold

These gates prevent “drift” (continual updates causing broad regressions).

## 10. Failure modes

1) **Teacher collapse** (teacher = student, no learning)
   - ensure teacher is EMA and not identical for too long
   - use \(\tau=0.999\) but consider smaller (0.99) if updates too slow
2) **KL too small (no adaptation)**
   - increase generation length, temperature
   - increase learning rate slightly
3) **Forgetting still occurs**
   - implement replay
   - reduce LR
   - constrain updates with additional KL to a frozen base model

# TTT-Discover: test-time training specification (implement exactly)

This section is derived from the NVIDIA “Learning to Discover at Test Time” paper in `docs/references/`.

## 1. Environment / discovery problem interface

The paper frames a discovery problem as:

- state \(s \in \mathcal{S}\)
- action \(a \in \mathcal{A}\)
- transition \(s' = T(a)\) (parsing model output into new solution)
- reward \(R(s') \in \mathbb{R}\), continuous and verifiable

We implement the following interface:

```python
class DiscoveryEnv:
    def __init__(self, problem_description: str): ...
    def initial_state(self) -> str: ...
    def context_from_archive(self, state: str, archive: Archive) -> str: ...
    def transition(self, action: str) -> str: ...
    def reward(self, state: str) -> float: ...
```

A minimal toy environment is included (string match) to validate the loop. Replace it with a real discovery environment.

## 2. Entropic utility objective

The paper defines the objective (Eq. (1) in their notation):

\[
J_\beta(\theta) = \mathbb{E}_{s \sim \text{reuse}(H)}\Big[\log \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \exp(\beta(s)\,R(s,a))\Big]
\]

Interpretation:
- inner expectation is like a softmax over rewards
- log makes it focus on high-reward outcomes (approaches max as β→∞)
- β(s) controls exploitation vs exploration

## 3. Gradient estimator (weighted policy gradient)

The gradient (Eq. (2)) is:

\[
\nabla_\theta J_\beta(\theta) = \mathbb{E}_{s}\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[w_\beta(a;s)\,\nabla_\theta\log\pi_\theta(a|s)\right]
\]

where:

\[
w_\beta(a;s) = \frac{\exp(\beta(s)R(s,a))}{\mathbb{E}_{a'\sim \pi_\theta(\cdot|s)} \exp(\beta(s)R(s,a'))}
\]

This is like REINFORCE with a special normalized weight.

### Baseline (variance reduction)

The paper uses baseline \(-1\), so advantage becomes:

\[
A(a;s) = w_\beta(a;s) - 1
\]

Implementation detail:
- We use a **leave-one-out** estimator for the normalization term to reduce bias (Appendix A.1).

## 4. KL shaping / regularization to base model

To avoid drifting too far from the base model \( \pi_{\theta_0} \), they shape the advantage:

\[
A(a;s) = w_\beta(a;s) - 1 - \lambda \log\frac{\pi_\theta(a|s)}{\pi_{\theta_0}(a|s)}
\]

We implement a simple REINFORCE loss:

\[
\mathcal{L}(\theta) = -\mathbb{E}[A(a;s)\,\log \pi_\theta(a|s)]
\]

and we **detach A** in code (treat it as a fixed weight for this gradient step).

In our minimal implementation:
- \(\log \pi_\theta(a|s)\) is the summed token log-prob of the generated action continuation
- \(\log \pi_{\theta_0}(a|s)\) is computed similarly under the frozen base model
- \(A\) is:
  - entropic advantages (LOO) minus λ times logprob difference

## 5. Adaptive β(s) (Appendix A.1)

Instead of a fixed β, the paper chooses β(s) by solving:

\[
D_{KL}\big(q_{\beta(s)}(\cdot|s)\;||\;\pi_\theta(\cdot|s)\big) = \gamma
\]

In practice, with N sampled actions in a batch, define:

\[
q_\beta(n) = \frac{\exp(\beta r_n)}{\sum_m \exp(\beta r_m)}
\]

and constrain the KL against a uniform distribution (proxy):

\[
D_{KL}(q_\beta || u) = \gamma
\]

where \(u(n)=1/N\). They choose \(\gamma=\ln 2\).

We implement bisection to find β for each batch.

## 6. Leave-one-out (LOO) advantages (Appendix A.1)

Let \(r_n\) be rewards, and \(r_{\max} = \max_n r_n\).
Define:

\[
w_n = \exp(\beta(r_n - r_{\max}))
\]

Compute:

\[
\hat{Z}_{-n} = \frac{\sum_m w_m - w_n}{N-1}
\]

Then LOO advantage:

\[
A_n = \frac{w_n}{\hat{Z}_{-n}} - 1
\]

We implement this exactly in:
- `entropic_advantages_loo()`.

## 7. Reuse strategy via archive + PUCT (Appendix A.2)

The paper maintains an archive \(H_t\) of promising states and selects a start state for new rollouts.

Score:

\[
\text{score}(s) = Q(s) + c \cdot (R_{\max} - R_{\min}) \cdot P(s) \cdot \sqrt{1 + \frac{T}{1+n(s)}}
\]

Where:
- \(Q(s)=m(s)\) if \(n(s)>0\), else \(Q(s)=R(s)\)
- \(m(s)\) = best reward among expanded children
- \(n(s)\) = count of times state expanded (and ancestors updated)
- \(T\) = total expansions so far
- \(P(s)\) = a rank-based prior (higher for higher-reward states)
- \(c\) = exploration coefficient

We implement:
- `Archive` data structure
- `puct_select_start_state()`
- `update_after_expand()` which updates m(s), n(s), and ancestors

## 8. Parameter updates: LoRA-only (frozen base model)

At test time, we must adapt quickly without corrupting the base.

Decision:
- freeze the pretrained model weights θ0
- add LoRA adapters to selected linear layers
- optimize only LoRA weights at test time

We implement LoRA ourselves (minimal, robust, no external dependencies):
- `src/llm_mhc_sdft_tttd/model/lora.py`
- `apply_lora()` replaces targeted `nn.Linear` layers with `LoRALinear`.
- `mark_only_lora_trainable()` freezes everything else.

## 9. Minimal implementation status

The provided implementation in:
- `src/llm_mhc_sdft_tttd/training/ttt_discover.py`

is intended as a *reference skeleton*:
- it implements archive, PUCT selection, adaptive β, LOO advantages, LoRA-only updates
- but the toy environment is simplistic
- real discovery tasks require a real reward function

## 10. Failure modes (and mitigation)

1) **Test-time RL diverges**
   - reduce LR
   - reduce β or enforce adaptive β
   - increase KL penalty λ
2) **Overfits to bad parsing**
   - tighten env.transition parsing
   - reward invalid outputs with 0
3) **No improvement**
   - increase rollouts_per_step
   - increase ttt_steps
   - ensure reward is informative/continuous

# End-to-end build plan (phases, tasks, definitions of done)

This section is the operational heart of the handoff.  
Follow it in order. Each phase ends with **Definition of Done (DoD)** checks.

## Phase numbering

- Phase 0: repository bootstrap & environment
- Phase 1: tokenizer training
- Phase 2: data packing pipeline
- Phase 3: baseline LM forward & loss
- Phase 4: mHC implementation + unit tests
- Phase 5: pretraining loop (tiny smoke run)
- Phase 6: pretraining run (real-ish)
- Phase 7: SDFT dataset + SDFT loop
- Phase 8: replay + regression gates
- Phase 9: LoRA system (adapter save/load)
- Phase 10: TTT-Discover loop + toy env
- Phase 11: integrate TTT -> SDFT consolidation
- Phase 12: evaluation suite + dashboards
- Phase 13: packaging and reproducibility locks

Each phase below includes:
- **Inputs**
- **Steps**
- **Artifacts**
- **DoD (Definition of Done)**
- **Common failures & fixes**

## Phase 0 — Bootstrap the repo and Python environment

### Inputs

- A workstation with Linux recommended (Ubuntu 22.04+). Windows works but symlinks may behave differently.
- NVIDIA GPU with CUDA installed (RTX Pro 6000 96GB VRAM).
- Python 3.10+.


### Steps

1) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
```

2) Install the package in editable mode with dev deps:

```bash
# IMPORTANT (GPU): install a CUDA-enabled PyTorch build first (from the official
# PyTorch instructions) so that `pip install` does not accidentally pull a CPU-only
# torch wheel.

pip install -e .[dev]

# If editable install is not available in your environment, you can still run
# everything from the repo root without installing the package:
#   export PYTHONPATH=$PWD/src
# (scripts/ and tests/ are bootstrapped to work in this mode.)
```

3) Verify GPU visibility:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("vram (GB):", torch.cuda.get_device_properties(0).total_memory/1e9)
PY
```

4) Run unit tests:

```bash
pytest -q
```


### Artifacts produced

- `.venv/` created
- package import works: `import llm_mhc_sdft_tttd`
- tests pass


### Definition of Done (DoD)

- All commands succeed.
- `pytest -q` passes (all tests).


### Common failures & fixes

- **torch.cuda.is_available() == False**
  - check NVIDIA driver, CUDA runtime, and correct PyTorch CUDA build.
- **pytest import fails**
  - run `pip install -e .[dev]` again
  - ensure you are in the repo root
  - if editable install still fails, run from source:
    - `export PYTHONPATH=$PWD/src`
    - then rerun `pytest -q`



## Phase 1 — Train tokenizer (SentencePiece BPE)

### Inputs

- Raw text corpus files in `data/raw/` (you create these).
  - Each line should be a document or paragraph.
- Decision: vocab_size=32000, BPE.


### Steps

1) Create folder structure:

```bash
mkdir -p data/raw data/tokenizer
```

2) Provide a tiny corpus for smoke test:

```bash
cat > data/raw/tiny.txt <<'EOF'
Hello world.
This is a tiny corpus for tokenizer smoke tests.
EOF
```

3) Train tokenizer:

```bash
python scripts/train_tokenizer.py \
  --input data/raw/tiny.txt \
  --model_prefix data/tokenizer/spm32k \
  --vocab_size 32000 \
  --model_type bpe
```

4) Quick encode/decode sanity:

```bash
python - <<'PY'
from llm_mhc_sdft_tttd.data.tokenizer import SpmTokenizer
tok = SpmTokenizer("data/tokenizer/spm32k.model")
s = "The quick brown fox jumps over the lazy dog."
ids = tok.encode(s, add_bos=True, add_eos=True)
print("ids[:20] =", ids[:20])
print("decoded =", tok.decode(ids))
print("vocab_size =", tok.vocab_size)
PY
```


### Artifacts produced

- `data/tokenizer/spm32k.model`
- `data/tokenizer/spm32k.vocab`


### Definition of Done (DoD)

- Tokenizer trains without error.
- encode/decode roundtrip produces readable text.
- `vocab_size` equals requested (or close; SentencePiece may adjust slightly).


### Common failures & fixes

- **Training extremely slow**: your corpus is huge; start with tiny.
- **Unicode errors**: ensure raw text is UTF-8.



## Phase 2 — Prepare packed token datasets (.bin)

### Inputs

- Tokenizer from Phase 1.
- Raw train/val text files.
- Decision: append EOS after each line/document.


### Steps

1) Create tiny train/val text:

```bash
mkdir -p data/raw
cp data/raw/tiny.txt data/raw/train.txt
cp data/raw/tiny.txt data/raw/val.txt
```

2) Pack them:

```bash
mkdir -p data/packed
python scripts/prepare_data.py \
  --tokenizer data/tokenizer/spm32k.model \
  --input data/raw/train.txt \
  --output data/packed/train.bin \
  --append_eos 1

python scripts/prepare_data.py \
  --tokenizer data/tokenizer/spm32k.model \
  --input data/raw/val.txt \
  --output data/packed/val.bin \
  --append_eos 1
```

3) Inspect metadata:

```bash
cat data/packed/train.bin.json
```


### Artifacts produced

- `data/packed/train.bin` + `.json`
- `data/packed/val.bin` + `.json`


### Definition of Done (DoD)

- Metadata JSON exists and has:
  - correct vocab_size
  - n_tokens > 0
- The `.bin` file size is consistent with dtype (uint16/uint32).


### Common failures & fixes

- **n_tokens = 0**: your input file lines are empty; check raw data.
- **dtype mismatch**: if vocab_size > 65535, the script uses uint32.



## Phase 3 — Baseline LM forward pass and cross-entropy loss

### Inputs

- Tokenizer and packed dataset from Phases 1-2.
- Use `configs/model_mhc_120m.json` for fast smoke test.


### Steps

1) Install package (already done in Phase 0).

2) Run a forward pass on random input:

```bash
python - <<'PY'
import json, torch
from llm_mhc_sdft_tttd.config import ModelConfig
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM

cfg = ModelConfig.from_json(open("configs/model_mhc_120m.json","r",encoding="utf-8").read())
cfg.vocab_size = 32000
m = MHCTransformerLM(cfg)
x = torch.randint(0, cfg.vocab_size, (2, 64))
logits = m(x)
print("logits:", logits.shape)
PY
```

3) Compute one batch loss from packed dataset:

```bash
python - <<'PY'
import json, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from llm_mhc_sdft_tttd.config import ModelConfig
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM
from llm_mhc_sdft_tttd.data.dataset import PackedTokenDataset

cfg = ModelConfig.from_json(open("configs/model_mhc_120m.json","r",encoding="utf-8").read())
m = MHCTransformerLM(cfg)
ds = PackedTokenDataset("data/packed/train.bin", seq_len=64, dtype="uint16")
dl = DataLoader(ds, batch_size=2)
x,y = next(iter(dl))
logits = m(x)
loss = F.cross_entropy(logits.view(-1,logits.size(-1)), y.view(-1))
print("loss:", float(loss))
PY
```


### Artifacts produced

- Successful forward pass prints tensor shapes.
- One-step loss is finite (not NaN/Inf).


### Definition of Done (DoD)

- forward pass completes on CPU.
- loss is finite.


### Common failures & fixes

- **Shape mismatch**: ensure `d_model = n_heads * d_head` in config.
- **NaN loss**: check tokenizer vocab and ensure tokens < vocab_size.



## Phase 4 — Verify mHC invariants (doubly-stochastic + identity-mapping sanity)

### Inputs

- Existing unit tests: `tests/test_sinkhorn.py`, `tests/test_model_shapes.py`.
- This phase adds deeper checks and ensures mHC is actually used.


### Steps

1) Run existing tests:

```bash
pytest -q
```

2) Add a new test: check mean-preservation under H_res.

Create `tests/test_mhc_identity_property.py` with:
- construct random X
- compute H_res from MHCMapping
- verify that mean(X) approx equals mean(H_res@X)

3) Run tests again:

```bash
pytest -q
```


### Artifacts produced

- New test file `tests/test_mhc_identity_property.py`


### Definition of Done (DoD)

- All tests pass.
- The mean-preservation check passes within tolerance (e.g., 1e-3 to 1e-2 depending on float dtype).


### Common failures & fixes

- **Fails due to numerical tolerance:** increase Sinkhorn tmax or relax tolerance.
- **Fails due to implementation bug:** verify einsum dims and Sinkhorn normalization.



## Phase 5 — Pretraining smoke run (tiny data, few steps)

### Inputs

- Packed data exists.
- Use tiny model config: 120M.
- Goal: validate the training loop end-to-end.


### Steps

1) Run a tiny pretrain (e.g., 20 steps) on CPU or GPU:

```bash
python scripts/pretrain.py \
  --train_bin data/packed/train.bin \
  --val_bin data/packed/val.bin \
  --out runs/pretrain_smoke \
  --model configs/model_mhc_120m.json \
  --steps 20 \
  --seq_len 64 \
  --micro_bs 2 \
  --grad_accum 1
```

2) Confirm artifacts:

- `runs/pretrain_smoke/model_config.json`
- `runs/pretrain_smoke/pretrain_config.json`
- `runs/pretrain_smoke/ckpt_latest.pt` (always written)
- `runs/pretrain_smoke/ckpt_step_0000020.pt` (final checkpoint for `--steps 20`)

3) Load checkpoint and run generation:

```bash
python - <<'PY'
import torch, json
from llm_mhc_sdft_tttd.config import ModelConfig
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM
from llm_mhc_sdft_tttd.data.tokenizer import SpmTokenizer

tok = SpmTokenizer("data/tokenizer/spm32k.model")
cfg = ModelConfig.from_json(open("runs/pretrain_smoke/model_config.json","r",encoding="utf-8").read())
m = MHCTransformerLM(cfg)
ckpt = torch.load("runs/pretrain_smoke/ckpt_latest.pt", map_location="cpu")
m.load_state_dict(ckpt["model"], strict=True)
prompt = "Hello"
ids = torch.tensor([tok.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long)
out = m.generate(ids, max_new_tokens=20, eos_token_id=tok.eos_id())
print(tok.decode(out[0].tolist()))
PY
```


### Artifacts produced

- Training logs printed
- Checkpoints in `runs/pretrain_smoke/`


### Definition of Done (DoD)

- Loss decreases slightly across steps (not guaranteed on tiny data, but should be finite).
- Generation produces text without crashing.


### Common failures & fixes

- **CUDA OOM:** reduce seq_len, micro_bs, or use smaller model.
- **Loss NaN:** check sinkhorn clamp and RMSNorm eps.



## Phase 6 — Pretraining run (350M on GPU, limited data)

### Inputs

- Packed dataset created from a larger corpus (you must supply data).
- Use `configs/model_mhc_350m.json`.
- Goal: get a usable base model checkpoint θ0.


### Steps

1) Prepare a larger training set (example sources):
- Common Crawl derived text (C4-like)
- OpenWebText-like corpora
- Code corpora if you want coding ability

2) Pack to `.bin` using `scripts/prepare_data.py`.

3) Run training:

```bash
python scripts/pretrain.py \
  --train_bin data/packed/train.bin \
  --val_bin data/packed/val.bin \
  --out runs/pretrain_350m \
  --model configs/model_mhc_350m.json \
  --steps 20000 \
  --seq_len 1024 \
  --micro_bs 2 \
  --grad_accum 16
```

Notes:
- adjust `steps` and data size to your compute budget
- monitor `val_loss` in logs

4) Keep `runs/pretrain_350m/ckpt_latest.pt` as θ0 baseline.


### Artifacts produced

- `runs/pretrain_350m/ckpt_latest.pt`
- logs with val_loss


### Definition of Done (DoD)

- Training runs stably for >1000 steps without NaNs.
- Validation loss is finite and trends down.
- Checkpoint loads and generates.


### Common failures & fixes

- **Training too slow:** reduce model size (120M) or seq_len.
- **No val improvement:** dataset quality issue; check packing and mixing.



## Phase 7 — Create an SDFT continual-learning dataset and run SDFT

### Inputs

- Base checkpoint θ0 from Phase 6 (or smoke from Phase 5).
- Tokenizer.
- A jsonl file of (prompt, demonstration) pairs.


### Steps

1) Create `data/sdft/demo.jsonl` with at least 10 examples.

Example:
```json
{"prompt":"Translate to German: Hello world.","demonstration":"Hallo Welt."}
```

2) Run SDFT:

```bash
mkdir -p data/sdft
python scripts/sdft.py \
  --ckpt runs/pretrain_350m/ckpt_latest.pt \
  --tokenizer data/tokenizer/spm32k.model \
  --data data/sdft/demo.jsonl \
  --out runs/sdft_smoke \
  --steps 200
```

3) Verify:
- `runs/sdft_smoke/sdft_step_0000200.pt` exists.


### Artifacts produced

- SDFT checkpoints in `runs/sdft_smoke/`


### Definition of Done (DoD)

- SDFT training produces finite loss.
- EMA teacher updates without error.


### Common failures & fixes

- **Very slow:** reduce max_new_tokens in config.
- **No learning:** increase LR or improve demonstration quality.



## Phase 8 — Implement replay + regression gates for continual learning

### Inputs

- SDFT loop exists (Phase 7).
- This phase upgrades it into a robust continual-learning system.


### Steps

1) Implement a replay buffer in `src/llm_mhc_sdft_tttd/training/sdft.py`:
- store past (prompt, demonstration) pairs
- sample replay examples with probability `replay_ratio`
- cap size at `replay_buffer_size`

2) Add regression gates:
- before and after a block of SDFT updates, run:
  - perplexity eval on a fixed small validation text set
  - a “capability probe” prompt list (hard-coded)
- if regression > threshold, revert weights (keep checkpoint before update)

3) Add tests:
- `tests/test_sdft_replay_sampling.py`
- `tests/test_sdft_gate_logic.py`

4) Run:
```bash
pytest -q
```


### Artifacts produced

- Replay buffer implementation
- Gate logic
- Unit tests for both


### Definition of Done (DoD)

- Unit tests pass.
- SDFT loop can run with replay enabled.
- Gates trigger correctly on synthetic regressions.


### Common failures & fixes

- **Complexity explosion:** keep replay minimal (a deque + random sampling).
- **Hard to define thresholds:** start with permissive thresholds and tighten later.



## Phase 9 — LoRA adapters (save/load/merge) for safe updates

### Inputs

- Minimal LoRA exists in `model/lora.py`.
- This phase ensures LoRA adapters are first-class artifacts.


### Steps

1) Add CLI utilities:
- `scripts/lora_export.py` (save LoRA weights)
- `scripts/lora_apply.py` (apply LoRA at inference)

2) Add unit tests:
- train LoRA weights for 1 step on tiny data
- save to disk
- load into fresh model
- verify identical output logits

3) Add an optional “merge” feature:
- for each LoRALinear: W <- W + scale * (B @ A)
- then remove LoRA wrappers
(This is optional and only for experiments; default policy is NOT to merge test-time LoRA into base.)


### Artifacts produced

- New scripts and tests


### Definition of Done (DoD)

- Save/load roundtrip test passes.
- (Optional) merge produces equivalent outputs to wrapped form.


### Common failures & fixes

- **Floating point mismatch:** allow small tolerance in logits compare.



## Phase 10 — TTT-Discover loop end-to-end on toy environment

### Inputs

- Base checkpoint θ0.
- Tokenizer.


### Steps

1) Run toy TTT-Discover:

```bash
python scripts/ttt_discover.py \
  --ckpt runs/pretrain_350m/ckpt_latest.pt \
  --tokenizer data/tokenizer/spm32k.model \
  --out runs/ttt_toy \
  --target "HELLO" \
  --steps 20 \
  --rollouts 32
```

2) Confirm:
- reward improves above 0.0
- LoRA checkpoints saved under `runs/ttt_toy/adapter_step_*/lora.pt`
- `runs/ttt_toy/result.json` exists

3) Add unit test:
- `tests/test_ttt_beta_solver.py` for `solve_beta_by_kl`
- `tests/test_ttt_puct_select.py` for archive selection correctness


### Artifacts produced

- `runs/ttt_toy/result.json`
- LoRA adapter checkpoints
- new unit tests


### Definition of Done (DoD)

- TTT-Discover runs without error and improves reward on toy env.
- unit tests pass.


### Common failures & fixes

- **No improvement:** increase rollouts or steps; toy target too hard.
- **Beta solver unstable:** clamp beta range, increase iters.



## Phase 11 — Integrate TTT discoveries into continual learning (TTT→SDFT)

### Inputs

- A TTT environment that outputs a discovered state/solution.
- SDFT training loop with replay + gates.


### Steps

Goal: convert successful TTT discoveries into demonstration examples for SDFT.

1) Define a “discovery log” format:
- For each problem instance:
  - problem description / prompt
  - best discovered solution string
  - reward

2) Write a script `scripts/ttt_to_sdft_jsonl.py` that converts discovery logs into:
```json
{"prompt": "...", "demonstration": "..."}
```

3) Run:
- execute TTT on N problems (even toy problems)
- convert outputs into `data/sdft/from_ttt.jsonl`
- run SDFT on that dataset

4) Add integration test:
- run toy TTT for few steps
- generate SDFT dataset
- run SDFT for few steps
- ensure pipeline completes


### Artifacts produced

- `scripts/ttt_to_sdft_jsonl.py`
- `data/sdft/from_ttt.jsonl`
- Integration test


### Definition of Done (DoD)

- Full pipeline TTT → dataset → SDFT executes without manual intervention.


### Common failures & fixes

- **Demonstrations are low quality:** filter by reward threshold.
- **Forgetting risk:** ensure replay + gates are enabled.



## Phase 12 — Evaluation suite (perplexity + task probes)

### Inputs

- Base model θ0, continual model θ_continual, and optional LoRA adapters.


### Steps

1) Implement a simple eval harness:
- `eval/perplexity.py` exists
- add `eval/probes.py` with:
  - a list of prompts
  - expected regex or substring checks
  - scoring

2) Add CLI `scripts/eval.py`:
- evaluate a checkpoint + tokenizer on:
  - perplexity dataset
  - probes

3) Add DoD: every update (SDFT) must run eval and log results to JSON.


### Artifacts produced

- `scripts/eval.py`
- `eval/probes.py`
- eval JSON logs


### Definition of Done (DoD)

- You can run `python scripts/eval.py ...` and get a JSON report.
- Regression gates can consume the report.


### Common failures & fixes

- **Probe brittle:** use regex and tolerant checks.



## Phase 13 — Reproducibility locks and packaging

### Inputs

- System is working end-to-end.


### Steps

1) Generate a frozen requirements lock file:

```bash
pip freeze > requirements.lock.txt
```

2) Record GPU + CUDA info into `runs/ENVIRONMENT.md`.

3) Add a `Makefile` with common targets:
- `make test`
- `make tokenizer`
- `make pack`
- `make pretrain_smoke`
- `make sdft_smoke`
- `make ttt_smoke`

4) Final: zip the repo (this document assumes you are reading it *inside* such a zip).


### Artifacts produced

- `requirements.lock.txt`
- `Makefile`
- `runs/ENVIRONMENT.md`


### Definition of Done (DoD)

- `make test` works on a clean machine with same GPU class.
- All phase commands are documented and reproducible.


### Common failures & fixes

- **Dependency mismatch:** adjust minimal dependencies in `pyproject.toml`.


# Compute, memory, and “what is feasible” on a single 96GB GPU

This section is intentionally explicit so you can choose model size and batch sizes without guessing.

## 1. Memory budgeting (rule-of-thumb)

Let:
- \(P\) = number of parameters
- \(b_w\) = bytes per weight (bf16 = 2)
- \(b_g\) = bytes per gradient (bf16 = 2, fp32 = 4)
- \(b_{opt}\) = bytes per optimizer state (Adam has 2 moments; often fp32 => 8 bytes/param)

### 1.1 AdamW memory

Typical AdamW (no sharding) uses:

- weights: bf16 => 2 bytes
- grads: bf16 => 2 bytes
- moments m,v: fp32 => 8 bytes
- optionally master weights fp32 => 4 bytes (depends on implementation)

So worst-case:

\[
\text{bytes/param} \approx 2 + 2 + 8 + 4 = 16
\]

Best-case (no master weights):

\[
\text{bytes/param} \approx 2 + 2 + 8 = 12
\]

Thus parameter+optimizer memory:

\[
M_{\text{params}} \approx P \cdot (12\text{ to }16) \text{ bytes}
\]

Example:
- \(P=350\text{M}\) => ~4.2GB to 5.6GB
- \(P=900\text{M}\) => ~10.8GB to 14.4GB

This fits easily in 96GB.

### 1.2 Activations dominate at long sequence lengths

Activation memory depends on:
- batch size B
- seq length T
- hidden size C
- number of layers L
- checkpointing strategy

A crude upper bound (no checkpointing) is:

\[
M_{\text{act}} \propto B \cdot T \cdot C \cdot L
\]

mHC increases residual stream storage by factor n=4:
- internal stream tensor is `[B,T,n,C]` not `[B,T,C]`
- this multiplies residual-state activations by 4

But note: attention/MLP compute runs on aggregated `[B,T,C]`, so attention KV activations are not multiplied by 4 unless you store per-stream (we do not).

**Practical guidance:**
- Always enable gradient checkpointing for pretraining on single GPU.
- Start with seq_len=512 or 1024, then move to 2048 once stable.

## 2. Feasible model sizes (recommended)

### 2.1 Safe “MVP” sizes

- 120M: fast iteration, runs easily on CPU/GPU
- 350M: best balance for a single GPU
- 900M: feasible but slower; pretraining from scratch is compute-heavy

### 2.2 Why not jump to multi-billion?

Even if memory fits, from-scratch pretraining is compute-limited:
- a 3B model wants trillions of tokens for strong performance
- a single GPU cannot practically reach that regime in reasonable time

The blueprint supports scaling, but expects you to validate algorithms at smaller scale first.

## 3. Recommended batch settings (starting points)

These are initial guesses; you must tune based on your exact GPU and throughput.

### 3.1 120M model
- seq_len=512
- micro_batch_size=8
- grad_accum=1 to 4

### 3.2 350M model
- seq_len=1024
- micro_batch_size=2
- grad_accum=16

### 3.3 900M model
- seq_len=1024
- micro_batch_size=1
- grad_accum=32

## 4. Profiling checklist

When tuning, always log:
- tokens/sec
- GPU utilization
- peak VRAM
- training loss and val loss

Add a small helper script if needed.

## 5. Precision decisions

We default to bf16. If bf16 is unstable:
- use fp16 with loss scaling (requires additional code)
- or use fp32 (slow, but stable)

## 6. Multi-stage training (optional but recommended)

For a strong model:
1) start with seq_len=512 for stability
2) then continue training at seq_len=1024
3) finally seq_len=2048

This reduces early instability and allows the model to learn shorter dependencies first.

# Optional extension: match the DeepSeek-V3-style mHC architecture (MoE + MLA)

The mHC paper’s strongest results are on a DeepSeek-V3-like backbone.
On a single GPU, you will likely not train that full architecture from scratch at meaningful scale,
but you can still implement it for research parity.

This section is **optional**. The base implementation in this repo is **dense** (no MoE).

## 1. What the mHC paper reports (Table 5)

The paper lists hyperparameters for three model sizes (3B / 9B / 27B). Key items:

- sequence length: 4k
- expansion rate n: 4
- α_init: 0.01
- Sinkhorn t_max: 20
- RMSNorm eps: 1e-6
- Rotary position embeddings with θ=10000
- MoE:
  - number of routed experts: 64
  - shared experts: 2
  - active experts per token: 6
  - expert FFN dimension: 512 (as listed)
- Attention includes additional “latent” attention head parameters:
  - latent attention heads: 4
  - latent head dim: 128
  - KV head dim: 64
  (This suggests a special attention mechanism beyond standard MHA.)

**Important:** those details are enough to implement a structurally similar model, but they do *not* specify:
- exact router loss coefficient
- exact attention formulation (“MLA”) without the DeepSeek technical report
- exact training data recipe

Therefore, do not claim parity unless you implement and verify the missing pieces.

## 2. Minimal MoE implementation plan (single GPU)

If you decide to add MoE, implement it in the MLP sublayer:

### 2.1 Router
For each token hidden vector \(h \in \mathbb{R}^C\), compute expert logits:

\[
g = W_r h \in \mathbb{R}^{E}
\]

where E = number of experts (e.g., 8 or 16 for single GPU; the paper uses 64).

Convert to probabilities with softmax:

\[
p = \text{softmax}(g)
\]

Select top-k experts (k = active experts; paper uses k=6). For single GPU, start with k=2.

### 2.2 Dispatch
Dispatch tokens to experts; each expert is an FFN:

\[
\text{FFN}_e(h) = W_{down,e} \,\sigma(W_{up,e} h)
\]

Compute weighted sum over selected experts:

\[
\text{MoE}(h) = \sum_{e \in \text{topk}} p_e \,\text{FFN}_e(h)
\]

### 2.3 Load balancing loss (required)
MoE collapses if router always picks same experts. Add a balancing loss.
A standard choice (Switch Transformer style) is:

- importance: sum of router probs per expert
- load: count of assigned tokens per expert
- penalize variance or encourage uniform.

Because the mHC paper does not specify, you must consult additional sources if you want exact.

## 3. “MLA” attention (latent attention) – how to proceed safely

The mHC Table 5 suggests an attention variant with:
- latent attention heads (4)
- latent head dim (128)
- KV head dim (64)

Without the full DeepSeek attention spec, do not guess.

Safe approach:
1) Keep standard MHA (current repo).
2) Once the pipeline works, add a new attention module behind a feature flag:
   - `attention_impl = "mha" | "mla"`
3) Implement MLA only after you have a precise reference spec.

## 4. Why this is optional
The core research question in this project is the **combination**:
- mHC residual topology
- continual self-learning via SDFT
- test-time discovery via TTT-Discover

All three can be validated on the dense backbone first.

# Combined system blueprint (how the 3 mechanisms interact)

This section explains the **runtime architecture** of a combined system.

## 1. Three timescales of learning

Think in three nested loops:

### 1.1 Outer loop: pretraining (offline, long)
- Train θ0 from scratch on large corpus.
- Output: base checkpoint.

### 1.2 Middle loop: continual learning (online/offline hybrid)
- As new data/tasks arrive, update θ_continual using SDFT + replay + gates.
- Output: continually improved checkpoint(s).

### 1.3 Inner loop: test-time discovery (per-instance)
- For a single hard instance, adapt a LoRA adapter Δθ_problem using TTT-Discover.
- Output: a per-problem adapter + best discovered solution state.

The key principle:
- **test-time adapters are ephemeral and isolated**
- **continual model updates are controlled and gated**

## 2. Data flows

```
                   ┌────────────────────────────┐
                   │  Pretraining corpus        │
                   └─────────────┬──────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ PRETRAIN θ0   │
                         └───────┬───────┘
                                 │
             ┌───────────────────┴───────────────────┐
             │                                       │
             ▼                                       ▼
   ┌─────────────────────┐                 ┌──────────────────────┐
   │ Continual updates    │                 │ Test-time discovery  │
   │ (SDFT + replay +     │                 │ (TTT-Discover + LoRA)│
   │ regression gates)    │                 └──────────┬───────────┘
   └──────────┬──────────┘                            │
              │                                       │
              ▼                                       ▼
      ┌───────────────┐                        ┌───────────────┐
      │ θ_continual    │                        │ Δθ_problem     │
      └───────┬───────┘                        └───────┬───────┘
              │                                       │
              ▼                                       ▼
   ┌─────────────────────┐                 ┌──────────────────────┐
   │ Inference (general)  │                 │ Inference (problem)  │
   └─────────────────────┘                 └──────────────────────┘
```

## 3. Where SDFT and TTT connect

TTT-Discover produces artifacts:
- best discovered state(s) with high reward
- (optionally) a trace of intermediate states/actions
- an adapter Δθ that made discovery possible

We convert those into SDFT demonstrations:
- prompt = problem description (and any relevant context)
- demonstration = best discovered solution (state or final action)

Then SDFT can consolidate:
- run SDFT updates using those demonstrations
- with replay + gates to prevent forgetting

## 4. The “No drift” policy in practice

### 4.1 Never mutate base model directly at test time
- keep θ0 frozen
- keep θ_continual updated only by gated training
- keep Δθ_problem as separate file

### 4.2 Use eval gates
- every SDFT training session:
  - compare before/after metrics
  - revert if regression

### 4.3 Keep a “known good” baseline
- always keep a checkpoint that you can revert to

## 5. Artifacts and versioning

Use a simple directory convention:

```
runs/
  pretrain_350m/
    ckpt_latest.pt
    model_config.json
    pretrain_config.json
  continual/
    sdft_YYYYMMDD_HHMM/
      sdft_step_*.pt
      eval_before.json
      eval_after.json
      gate_decision.json
  ttt/
    problem_<hash>/
      adapter_step_010/lora.pt
      adapter_step_020/lora.pt
      result.json
      discovery_log.jsonl
```

## 6. Minimal “production” loop (single workstation)

1) Pretrain base model θ0 (once).
2) Deploy θ_continual = θ0.
3) For each new problem:
   - run TTT-Discover to get Δθ_problem and best solution
   - answer user with adapter applied
4) Nightly (or when enough logs collected):
   - convert discovery logs into SDFT dataset
   - run gated SDFT to update θ_continual
   - archive old θ_continual checkpoints

# File-by-file specification (what each file MUST do)

This section is a “map” for an implementing agent.

For each file:
- Purpose
- Key invariants
- How to validate

If you change behavior, update this section and add tests.

## 1) Configs

### `configs/model_mhc_120m.json`
- Small smoke-test model.
- Should run on CPU quickly.

### `configs/model_mhc_350m.json`
- Default “real run” model for a single 96GB GPU.

### `configs/model_mhc_900m.json`
- Larger model for experiments; slower.

## 2) Core library (`src/llm_mhc_sdft_tttd/`)

### `config.py`
- Defines dataclasses:
  - TokenizerConfig
  - MHCConfig
  - ModelConfig
  - PretrainConfig
  - SDFTConfig
  - TTTDiscoverConfig
- Must remain the ONLY place where knobs are defined.
- Invariant: values referenced elsewhere must exist here.

Validation:
- import works
- ModelConfig.from_json works with config JSON files

### `model/layers.py`
- Implements:
  - RMSNorm
  - RoPE
  - CausalSelfAttention (with q_proj/k_proj/v_proj/o_proj names)
  - SwiGLU MLP projections (gate_proj/up_proj/down_proj names)
- Invariant: `d_model == n_heads * d_head`.

Validation:
- `tests/test_model_shapes.py`

### `model/mhc.py`
- Implements:
  - Sinkhorn-Knopp projection (`sinkhorn_knopp`)
  - MHCMapping (computes H_pre/H_post/H_res)
  - MHCResidual wrapper

Invariants:
- H_res must be approximately doubly-stochastic (rows/cols sum ~1).
- H_pre in (0,1), H_post in (0,2).
- shape handling must support `[B,T,n,C]`.

Validation:
- `tests/test_sinkhorn.py`
- (Phase 4 adds mean-preservation test)

### `model/transformer.py`
- Implements:
  - AttentionSublayer, MLPSublayer
  - MHCTransformerBlock (applies mHC twice)
  - MHCTransformerLM (full model)
- Invariant: final logits shape `[B,T,V]`
- Invariant: generation runs without crashing.

Validation:
- `tests/test_model_shapes.py`
- Phase 5 generation sanity check

### `model/lora.py`
- Implements a minimal LoRA wrapper for nn.Linear:
  - apply_lora
  - save_lora / load_lora
  - mark_only_lora_trainable
- Invariant: base weights are frozen; only LoRA weights train.

Validation:
- Phase 9 save/load test

## 3) Training modules (`src/llm_mhc_sdft_tttd/training/`)

### `pretrain.py`
- Implements:
  - cosine LR schedule with warmup
  - gradient accumulation
  - checkpointing
- Invariant: loss finite; checkpoint loads.

Validation:
- Phase 5 pretrain smoke run

### `sdft.py`
- Implements SDFT:
  - on-policy sampling
  - EMA teacher
  - reverse KL loss (analytic)
- Invariant: teacher is not updated by gradients.

Validation:
- Phase 7 SDFT smoke run
- Phase 8 adds replay + gates

### `ttt_discover.py`
- Implements:
  - DiscoveryEnv interface
  - Archive + PUCT selection
  - adaptive beta solver
  - leave-one-out entropic advantages
  - LoRA-only test-time updates
- Invariant: base model frozen; only LoRA updates.

Validation:
- Phase 10 toy env run

## 4) Scripts (`scripts/`)

- `train_tokenizer.py`: trains SentencePiece
- `prepare_data.py`: packs tokens into `.bin`
- `pretrain.py`: runs pretraining
- `sdft.py`: runs continual learning
- `ttt_discover.py`: runs test-time discovery (toy)

All scripts must remain runnable on a clean machine after installation.

## 5) Tests (`tests/`)
- `test_sinkhorn.py`: Sinkhorn doubly stochastic check
- `test_model_shapes.py`: forward shape check

Phases add more tests.

## 6) References (`docs/references/`)
- PDFs only; do not edit.


# Appendix: Mathematical derivations (for implementation sanity)

This appendix is not required to run the code, but it is essential for avoiding subtle bugs.
If you (agent) are unsure about a sign, normalization, or indexing, consult here.

---

## A1. Reverse KL distillation loss (SDFT) – token-level form

We start from the sequence-level reverse KL:

\[
D_{KL}(p || q) = \sum_{y} p(y)\log\frac{p(y)}{q(y)}
\]

Here:
- \(p(y)=\pi_\theta(y|x)\)
- \(q(y)=\pi(y|x,c)\)

So:

\[
\mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\left[\log \pi_\theta(y|x) - \log \pi(y|x,c)\right]
\]

Autoregressive factorization:

\[
\log \pi_\theta(y|x) = \sum_{t=1}^{T}\log \pi_\theta(y_t \mid y_{<t}, x)
\]
\[
\log \pi(y|x,c) = \sum_{t=1}^{T}\log \pi(y_t \mid y_{<t}, x, c)
\]

Therefore:

\[
\mathcal{L}(\theta) =
\mathbb{E}_{y \sim \pi_\theta}\left[
\sum_{t=1}^{T}\left(
\log \pi_\theta(y_t|y_{<t},x) - \log \pi(y_t|y_{<t},x,c)
\right)
\right]
\]

This justifies computing loss over the generated tokens only, with teacher evaluated on the same prefixes.

### A1.1 Why analytic token KL equals REINFORCE expectation (intuition)

The KL objective can be differentiated directly, but estimating it from samples is noisy.
The paper compares estimators and finds the **analytic token KL** stable.

At each timestep t, define distributions over vocabulary:
- \(p_s(v)\) from student logits
- \(p_t(v)\) from teacher logits

Compute:

\[
D_{KL}(p_s || p_t) = \sum_v p_s(v)(\log p_s(v) - \log p_t(v))
\]

Summing over t matches the sequence objective in expectation.

---

## A2. Gradient of token-level reverse KL wrt student logits

Let student logits be \(z\in \mathbb{R}^{V}\) for a timestep, with:
- \(p_s = \text{softmax}(z)\)
- \(l_s = \log p_s\)

Teacher log-probs are \(l_t\) (detached).

Token KL:

\[
K(z) = \sum_v p_s(v) (l_s(v) - l_t(v))
\]

This is differentiable wrt z and is exactly what our code computes.

In PyTorch, the simplest stable implementation is:

```python
log_p_s = F.log_softmax(student_logits, dim=-1)
log_p_t = F.log_softmax(teacher_logits.detach(), dim=-1)
p_s = log_p_s.exp()
kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
```

---

## A3. Entropic objective (TTT-Discover) gradient derivation

Define:

\[
J(\theta) = \log \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} [ \exp(\beta R(a)) ]
\]

Let:

\[
Z(\theta) = \mathbb{E}_{a \sim \pi_\theta} [ \exp(\beta R(a)) ]
\]

Then:

\[
\nabla_\theta J(\theta) = \frac{1}{Z(\theta)} \nabla_\theta Z(\theta)
\]

Expand \(Z\):

\[
Z(\theta) = \sum_a \pi_\theta(a|s)\exp(\beta R(a))
\]

Differentiate:

\[
\nabla_\theta Z(\theta) = \sum_a \nabla_\theta \pi_\theta(a|s)\exp(\beta R(a))
\]
\[
= \sum_a \pi_\theta(a|s)\nabla_\theta \log \pi_\theta(a|s)\exp(\beta R(a))
\]

So:

\[
\nabla_\theta J(\theta) = \sum_a \frac{\pi_\theta(a|s)\exp(\beta R(a))}{Z(\theta)} \nabla_\theta \log \pi_\theta(a|s)
\]

Define weights:

\[
w_\beta(a;s) = \frac{\exp(\beta R(a))}{\mathbb{E}_{a'}\exp(\beta R(a'))}
\]

Then the gradient becomes:

\[
\nabla_\theta J(\theta) = \mathbb{E}_{a\sim \pi_\theta} [ w_\beta(a;s) \nabla_\theta \log \pi_\theta(a|s) ]
\]

This is exactly Eq. (2).

### A3.1 Why baseline -1 is valid

Because:
\[
\mathbb{E}_{a\sim\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0
\]

So subtracting any constant baseline \(b\) inside the expectation leaves the gradient unchanged:

\[
\mathbb{E}[ (w_\beta - b)\nabla\log\pi ] = \mathbb{E}[ w_\beta \nabla\log\pi ] - b\mathbb{E}[\nabla\log\pi] = \mathbb{E}[ w_\beta \nabla\log\pi ]
\]

Using \(b=1\) makes weights centered.

---

## A4. Adaptive β via KL constraint on qβ

Given a batch of N samples with rewards \(r_n\), define:

\[
q_\beta(n) = \frac{\exp(\beta r_n)}{\sum_m \exp(\beta r_m)}
\]

As β increases, qβ becomes more peaked on the best reward.

Define uniform \(u(n)=1/N\).

Compute:

\[
D_{KL}(q_\beta || u) = \sum_n q_\beta(n)\log\frac{q_\beta(n)}{1/N}
= \sum_n q_\beta(n)(\log q_\beta(n) + \log N)
\]

Set this equal to γ (default ln2).

We solve for β using bisection because the KL is monotonic in β for distinct rewards.

---

## A5. Sinkhorn-Knopp and doubly stochastic matrices (mHC)

Given a positive matrix \(M\), Sinkhorn alternates:
- normalize columns to sum 1
- normalize rows to sum 1

If M has total support, this converges to a doubly-stochastic matrix.

In mHC we start from:
\[
M^{(0)} = \exp(\tilde{H}^{res})
\]

which ensures positivity.

Practical issues:
- exp overflow: clamp inputs
- convergence: tmax=20 is a heuristic; increase if needed



# Appendix: Data engineering playbook (how to build a usable training corpus)

This blueprint does not ship a corpus. You must source one. This appendix tells you how to do it
without guessing, and in a way that is compatible with continual learning and test-time discovery.

## B1. Data principles

1) **Licensing:** only use data you can legally use for training.
2) **Deduplication:** remove near-duplicates to avoid memorization.
3) **Mixture:** combine:
   - general web text (language fluency)
   - books/papers (long-form reasoning)
   - code (if you want coding)
   - domain documents (if you have a target domain)
4) **Validation split:** keep a stable held-out set for regression gates.

## B2. Suggested corpus structure on disk

```
data/raw/
  train_shard_000.txt
  train_shard_001.txt
  ...
  val.txt
```

Each line:
- one document or paragraph
- avoid extremely long lines (cap at e.g. 50k chars; truncate)

## B3. Tokenization and packing strategy

Packing uses:
- SentencePiece to convert each line to token ids
- append EOS between lines
- concatenate into a single token stream
- write as uint16/uint32 `.bin`

This is exactly what `scripts/prepare_data.py` does.

For better quality:
- shuffle lines before packing
- interleave different sources by alternating shards

## B4. Continual learning datasets (SDFT)

SDFT needs (prompt, demonstration) pairs.

Where to get demonstrations:
1) human-written examples
2) curated from high-quality sources
3) generated by a stronger model (teacher)
4) extracted from successful test-time discovery (TTT-Discover)

Recommended fields:
- prompt (task statement)
- demonstration (good solution)
- optional metadata:
  - source
  - reward score
  - timestamp
  - tags

You can store metadata in jsonl, but the minimal loader only uses prompt+demonstration.

## B5. Evaluation sets

Keep fixed evaluation sets to detect drift:
- perplexity set: a held-out text shard not used in training
- probe prompts: small list of tasks you care about
- for code: small unit tests (python snippets)

Always compare:
- θ0 baseline
- θ_continual current
- after each new SDFT update session

## B6. Practical gotchas

- If you train tokenizer on different domain than pretraining corpus, you may under-utilize vocab.
- If you pack code and text together, consider adding a special token like `<code>` prefix per code line.
- If the corpus includes multiple languages, you may need larger vocab or separate tokenization strategy.

This appendix intentionally avoids specific dataset names to keep the blueprint license-agnostic.



# Appendix: Prompt library for implementation agents (Codex)

This section contains ready-to-use prompts that force an agent to stay grounded.

## C1. “Phase executor” prompt

> You are implementing Phase {K} from docs/CODEX_HANDOFF.md.  
> First, open and quote the Phase {K} section.  
> Then list the exact shell commands you will run.  
> Then implement code changes with minimal diffs.  
> Then run the validation commands.  
> If anything fails, debug by inspecting logs and code; do not invent fixes.

## C2. “No invention” debug prompt

> You are not allowed to guess.  
> For any missing value, search the repository.  
> If not found, search the PDFs in docs/references/ for the exact equation or hyperparameter.  
> If still not found, implement a placeholder with TODO(NEEDS_SPEC) and add a failing test describing the missing spec.

## C3. “Unit test first” prompt

> Write a failing unit test that captures the intended behavior.  
> Then implement the minimum code to pass it.  
> Then run the full test suite.

## C4. “Regression gate” prompt

> Implement the evaluation gate as described.  
> Add a synthetic test case that forces the gate to trigger.  
> Ensure the system reverts to the previous checkpoint when triggered.

## C5. “TTT environment integration” prompt

> Implement a new DiscoveryEnv for the target domain.  
> Define transition parsing rules and a deterministic reward function.  
> Add 3 unit tests for reward correctness and parsing robustness.  
> Run TTT-Discover for 5 steps and show that reward is non-decreasing in archive best.

These prompts can be copy/pasted into an agent session.



# Appendix: Troubleshooting compendium (common issues and deterministic fixes)

This section is intentionally long. When something breaks, do not guess—use the checklist.

## D1. Installation and import issues

### Symptom: `ModuleNotFoundError: llm_mhc_sdft_tttd`
- Cause: package not installed in the environment
- Fix:
  ```bash
  pip install -e .[dev]
  ```
- Verify:
  ```bash
  python -c "import llm_mhc_sdft_tttd; print(llm_mhc_sdft_tttd.__version__)"
  ```

### Symptom: SentencePiece training fails
- Cause: input file path wrong or file empty
- Fix:
  - ensure `--input` points to a real file
  - ensure file is UTF-8

## D2. Data pipeline issues

### Symptom: packed dataset has `n_tokens=0`
- Cause:
  - your text file is empty
  - all lines are whitespace
- Fix:
  - inspect the first 20 lines
  - ensure you are not stripping everything accidentally

### Symptom: `ValueError: Dataset too small for seq_len`
- Cause: seq_len too large for tiny dataset
- Fix:
  - reduce `seq_len` (e.g., 64)
  - or create a larger dataset

## D3. Pretraining issues

### Symptom: CUDA out of memory
Fixes in strict order:
1) reduce `seq_len`
2) reduce `micro_batch_size`
3) increase `grad_accum_steps` (keeps effective batch constant)
4) switch to smaller model config (120M)
5) enable gradient checkpointing (already default in config; ensure it’s implemented if you add it)

### Symptom: loss becomes NaN
Most common root causes:
- exp overflow in Sinkhorn (H_res)
- too large learning rate
- numerical instability in softmax

Fixes:
1) ensure Sinkhorn clamps before exp
2) increase sinkhorn_eps slightly (1e-6 -> 1e-5)
3) reduce α_init (0.01 -> 0.001)
4) reduce LR by 2–10x
5) verify dtype; try fp32 for debugging

### Symptom: loss does not decrease at all
- Cause: dataset too small or not shuffled
- Fix:
  - pack larger dataset
  - verify training loop is actually stepping optimizer (grad_accum logic)
  - print LR and verify it is non-zero

## D4. SDFT issues

### Symptom: SDFT loss ~0 from the start
- Cause: student and teacher distributions nearly identical AND teacher prompt not different
- Fix:
  - verify `make_teacher_prompt` is used
  - ensure demonstration is non-empty
  - ensure teacher is EMA and not always synced

### Symptom: training extremely slow
- Cause: generation inside loop is expensive
- Fix:
  - reduce `max_new_tokens`
  - reduce batch size
  - reduce model size

### Symptom: forgetting still happens
- Fix:
  - implement replay (Phase 8)
  - implement regression gates (Phase 8)
  - optionally add KL penalty to θ0

## D5. TTT-Discover issues

### Symptom: β solver returns ~0 always
- Cause:
  - rewards are nearly identical
  - gamma too high for this reward spread
- Fix:
  - ensure reward function has variance
  - reduce gamma (e.g., ln(1.5))
  - increase rollouts_per_step

### Symptom: archive best reward never improves
- Cause:
  - reward is too sparse (mostly zeros)
  - prompt format prevents model from outputting valid actions
- Fix:
  - redesign reward to be continuous
  - add validity shaping (partial credit)
  - adjust prompt to demand the exact format

### Symptom: LoRA updates change nothing
- Cause:
  - LoRA not applied to any modules (name mismatch)
  - LoRA ranks too small
- Fix:
  - print list of replaced modules from `apply_lora`
  - check target suffixes match real module names
  - increase rank

## D6. Deterministic debugging procedure

When anything breaks, do this:

1) Reproduce on the smallest config:
   - model 120M
   - seq_len 64
   - 2–20 steps

2) Switch to fp32 temporarily.

3) Add asserts on shapes and finiteness:
   ```python
   assert torch.isfinite(tensor).all()
   ```

4) Bisect changes:
   - revert recent edits
   - re-apply one by one



# Appendix: Hyperparameter tables (quick reference)

This appendix lists recommended hyperparameters for each stage, so you can avoid inventing them.

## E1. Tokenizer
- vocab_size: 32000
- model_type: bpe
- character_coverage: 0.9995
- normalization: nmt_nfkc_cf
- special ids: pad=0, unk=1, bos=2, eos=3

## E2. Pretraining (starter)
- optimizer: AdamW
- lr: 3e-4 (120M) / 2e-4 (350M) / 1e-4 (900M)
- betas: (0.9, 0.95)
- weight_decay: 0.1
- warmup_steps: 2000 (adjust if total_steps small)
- lr_decay: cosine
- grad_clip: 1.0
- dtype: bf16

## E3. mHC
- n_streams: 4
- alpha_init: 0.01
- sinkhorn_tmax: 20
- sinkhorn clamp: [-15, 15]
- init:
  - H_pre uniform sum~1 via bias
  - H_post uniform sum~1 via bias
  - H_res approx identity via bias

## E4. SDFT
- teacher EMA decay: 0.999
- lr: 1e-5
- kl_coef: 1.0
- max_new_tokens: 256 (reduce if slow)
- temperature: 1.0
- top_p: 0.95
- replay_ratio: 0.2 (after Phase 8)

## E5. TTT-Discover (single GPU defaults)
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.0
- ttt_steps: 50
- rollouts_per_step: 64
- max_new_tokens: 256
- temperature: 1.0
- top_p: 0.95
- adaptive_beta: True
- target_kl_gamma: ln(2)
- kl_penalty_lambda: 0.1
- puct_c: 2.0

## E6. Model configs
### 120M (`configs/model_mhc_120m.json`)
- n_layers: 12
- d_model: 768
- n_heads: 12
- d_head: 64
- d_ff: 2048
- max_seq_len: 2048

### 350M (`configs/model_mhc_350m.json`)
- n_layers: 24
- d_model: 1024
- n_heads: 16
- d_head: 64
- d_ff: 2816
- max_seq_len: 2048

### 900M (`configs/model_mhc_900m.json`)
- n_layers: 16
- d_model: 2048
- n_heads: 16
- d_head: 128
- d_ff: 5632
- max_seq_len: 2048

If you change these, update configs AND this appendix.



# Appendix: Glossary (terms used consistently)

- **Adapter**: a small set of parameters (e.g., LoRA weights) that modifies a base model.
- **Archive (TTT)**: a set of candidate solution states kept during test-time training.
- **Birkhoff polytope**: set of doubly-stochastic matrices; convex hull of permutation matrices.
- **Causal LM**: decoder-only language model trained to predict next token.
- **Continual learning**: updating a model over time without catastrophic forgetting.
- **Demonstration (SDFT)**: an example response used as in-context teacher conditioning.
- **Doubly-stochastic**: matrix with non-negative entries, each row sum = 1, each column sum = 1.
- **Entropy / entropic objective**: an objective involving log-sum-exp or exp-weighted averages over rewards.
- **EMA teacher**: exponential moving average copy of the student weights used as a teacher model.
- **HC**: Hyper-Connections (residual stream expansion).
- **mHC**: manifold-constrained Hyper-Connections (HC with constrained H_res).
- **PUCT**: selection strategy combining value estimates and exploration prior.
- **Replay**: mixing past data during continual learning to reduce forgetting.
- **Reverse KL**: D_KL(student || teacher), encourages student to cover teacher support.
- **Sinkhorn-Knopp**: iterative projection of a positive matrix to be doubly-stochastic.
- **SDFT**: Self-Distillation Fine-Tuning for continual learning.
- **TTT**: test-time training; adapting model at inference time on a specific instance.
- **TTT-Discover**: specific TTT method for discovery problems using entropic objective and archive reuse.



# Appendix: Token indexing & alignment cheat sheets (avoid off-by-one bugs)

LLM training code is full of off-by-one traps. This appendix makes them explicit.

## F1. Next-token prediction indexing (pretraining)

Given a token sequence:

\[
x = (x_0, x_1, \dots, x_{T-1})
\]

A causal LM outputs logits:

\[
\text{logits}[t] \approx \log p(x_{t+1} \mid x_{\le t})
\]

In standard training we form:
- input_ids = x[0:T-1]
- labels = x[1:T]

But in this repo, `PackedTokenDataset` returns:
- `x` length `seq_len` (positions 0..seq_len-1)
- `y` length `seq_len` shifted by 1

Specifically it samples a chunk of length `seq_len+1`:
- returns chunk[:-1] as x
- returns chunk[1:] as y

Then in training we compute:

```python
logits = model(x)                  # [B, seq_len, V]
loss = CE(logits.view(-1,V), y.view(-1))
```

This matches the standard teacher-forced next-token objective.

### F1.1 Causal masking

We rely on `scaled_dot_product_attention(..., is_causal=True)` which enforces:
- token t attends only to tokens ≤ t

No explicit mask is needed for the core LM.

## F2. Generation indexing (sampling)

Suppose we have context tokens `ctx` of length C, and we generate L new tokens.

We feed the model the current sequence `ids` and get logits:
- logits shape [B, current_len, V]
- next token distribution is logits[:, -1, :]

At each step:
1) compute probs = softmax(logits[:, -1, :]/temperature)
2) sample next_token
3) append to ids

This is implemented in `MHCTransformerLM.generate`.

## F3. Log-prob of a generated continuation (TTT / KL shaping)

We often need:

\[
\log \pi_\theta(a|s)
\]

where `a` is the generated continuation after a prompt/context.

Let:
- full sequence tokens = `[prompt_tokens] + [action_tokens]`
- length S
- context length = C (prompt_tokens length)

Then the model’s logits at position t predict token t+1.

So for the continuation token at index t (t >= C), its logprob is:
- take logits at position t-1
- gather the probability of token t

So sum logprob over continuation tokens is:

\[
\sum_{t=C}^{S-1} \log p(x_t \mid x_{<t})
\]

In code (`sequence_logprob`):
- iterate t from ctx_len to S-1
- prev_pos = t-1
- token = input_ids[t]
- accumulate log_probs[prev_pos, token]

### F3.1 Why we ignore prompt logprob
We only care about the probability of the model generating the **action**.
The prompt is fixed and not sampled; including it would add a constant term (for the same prompt),
but across different prompts in a batch it would distort weights. So we ignore it.

## F4. SDFT alignment (student vs teacher context lengths)

SDFT uses two contexts:
- student context: tokens of prompt x
- teacher context: tokens of template(x, demonstration c)

These contexts usually have different lengths:
- len(student_ctx) = Cs
- len(teacher_ctx) = Ct

We sample a continuation `y` of length L from the student model conditioned on x.

Now we need to compute KL at each generated timestep between:
- student distribution given prefix y_<t and x
- teacher distribution given prefix y_<t and x,c

Implementation approach (used in repo):
1) build `student_full = student_ctx + y`
2) build `teacher_full = teacher_ctx + y`
3) run forward passes to get logits for all positions

We want logits that predict the generated tokens y_1..y_L.

For student:
- y_1 is predicted at position (Cs-1)
- y_t is predicted at position (Cs-1 + (t-1))

So slice logits:
- start = Cs-1
- end = start + L
- logits_student_slice = logits_student_all[start:end]

For teacher:
- same idea with Ct.

Then compute token-level KL between these slices.

### F4.1 Handling early stopping
If generation stops early (EOS), L may be 0 (if EOS immediately). This is rare but possible.
In that case, the current implementation creates a dummy token to avoid empty tensors.
A more principled approach is to skip that example or re-sample.

## F5. Leave-one-out entropic advantages (TTT) sanity checks

Given rewards r[0..N-1]:
- compute rmax = max(r)
- w = exp(beta*(r-rmax))  # stable

If all rewards equal:
- r-rmax = 0
- w = 1
- Z_minus = (N-1)/(N-1) = 1
- A = 1/1 -1 = 0
So no gradient update, which is correct: the batch contains no preference signal.

If one reward is higher:
- its w is 1
- others w < 1
- the best sample gets positive advantage
- the worst get negative advantage
This creates a learning signal even if absolute rewards are small.

## F6. Adaptive beta solver monotonicity check

The KL(q||u) should be:
- 0 at beta=0 (q uniform)
- increasing as beta increases (q more peaked)
- approaching log(N) as beta→∞ (q delta on best reward)

So any solver should satisfy:
- beta ~= 0 when rewards are identical
- beta increases when rewards have higher variance

Our bisection solver assumes monotonicity; if your reward function can produce NaNs, fix reward first.



# Appendix: Reward design patterns for discovery environments (TTT-Discover)

TTT-Discover is only as good as the reward function. This appendix gives grounded patterns.

## G1. Core constraints (from the paper framing)
A good reward function for TTT-Discover must be:
1) **Deterministic** (same state -> same reward)
2) **Verifiable** (computed by code, not model judgment)
3) **Continuous** (not just 0/1; partial credit is critical)
4) **Robust to formatting** (parsing should be strict but helpful)

## G2. General patterns

### Pattern 1: Partial correctness score
If the output is structured (JSON, code, math):
- reward invalid format = 0
- reward valid format but wrong answer = small positive (e.g., 0.1)
- reward increases with number of correct fields / unit tests passed

Example: for JSON with K required keys:
\[
R = 0.1 \cdot \mathbb{1}[\text{valid json}] + 0.9 \cdot \frac{\#\text{correct keys}}{K}
\]

### Pattern 2: Unit-test based reward
For code generation tasks:
- run a test suite in a sandbox
- reward = fraction of tests passed
- optionally add a speed bonus (but only after correctness works)

Example:
\[
R = \frac{\#\text{tests passed}}{\#\text{tests total}}
\]

### Pattern 3: Distance-to-target score
If there is a numeric target vector y* and candidate y:
\[
R = \exp(-\alpha \|y-y^*\|^2)
\]
This yields smooth gradients and stable β adaptation.

### Pattern 4: Curriculum / staged reward
Start with easy subgoals and combine:
\[
R = w_1 R_1 + w_2 R_2 + \dots
\]
Example for theorem proving:
- parse correctness
- proof length penalty
- lemma usage reward
- final theorem check

## G3. Anti-exploitation measures

Reward functions get exploited. Common exploits:
- outputting extremely long garbage that triggers a parser bug
- causing timeouts to avoid penalties

Mitigations:
- strict timeouts and resource limits
- cap output length
- sanitize inputs (no file system access)
- treat any exception as reward=0

## G4. Sandboxing recommendation (for code rewards)

If you evaluate code (Python/C++):
- run in a container or restricted subprocess
- enforce:
  - CPU time limit
  - memory limit
  - no network
  - no filesystem writes (or only temp dir)
- never `eval` raw model outputs without parsing and validation

This repository does not ship a full sandbox implementation; implement it in your environment layer.

## G5. How to log rewards for later SDFT consolidation

Always log:
- prompt/problem
- action text
- parsed state
- reward
- timestamp
- environment version (commit hash if possible)

Then you can:
- filter best trajectories
- build SDFT demonstrations

A minimal jsonl log entry:

```json
{
  "problem_id":"...",
  "prompt":"...",
  "action":"...",
  "state":"...",
  "reward":0.73,
  "timestamp":"...",
  "env_version":"..."
}
```



# Appendix: Practical RL-at-test-time pitfalls and how to avoid them

Even with the correct math, test-time RL with LLMs can fail due to engineering details.

## H1. Sampling collapse
If temperature too low or top_p too small:
- the model generates near-identical actions
- rewards have no variance
- β solver returns ~0
- no learning signal

Fix:
- increase temperature (0.8–1.2)
- increase top_p (0.95–0.98)
- increase rollouts_per_step

## H2. Over-optimization of prompt formatting
The model may learn to exploit the reward by changing format.
Fix:
- lock output format strictly
- reward invalid format = 0
- include formatting in prompt explicitly: “Output ONLY …”

## H3. Reward hacking via parser bugs
Fix:
- treat any parsing exception as invalid
- unit test parsing with adversarial strings

## H4. Catastrophic test-time drift
Even with LoRA, the adapter may learn behaviors that harm general performance.
Fix:
- keep KL penalty λ > 0
- restrict adapter training to a small number of steps
- reset adapter per problem; do not reuse across unrelated problems

## H5. Instability from very large β
As β→∞, wβ becomes near-one-hot and gradients become high variance.
Adaptive β avoids this by bounding KL(q||u)=γ.

If still unstable:
- clamp β to a max (e.g., 1000)
- reduce γ

## H6. Credit assignment in multi-step environments
If your environment is multi-step (state evolves):
- reward should reflect intermediate progress, not only final success
- otherwise the algorithm has sparse feedback

Techniques:
- shaped reward (partial credit)
- archive reuse helps, but only if rewards are informative

## H7. Logging is mandatory
Without logs you cannot:
- debug reward
- reproduce improvements
- consolidate discoveries into continual learning

Always log:
- random seed
- prompt
- action
- reward
- adapter weights path



# Appendix: Experiment tracking and metrics (minimal but sufficient)

To make continual learning and test-time training reliable, you must track experiments consistently.

## I1. What to log during pretraining
For every training run, write a JSONL log with fields:
- step
- train_loss
- val_loss (if computed)
- learning_rate
- tokens_processed
- tokens_per_second
- gpu_memory_allocated (optional)
- wall_time_seconds

If you do not want to add dependencies (wandb, mlflow), a JSONL file is enough.

Example log line:
```json
{"step":1000,"train_loss":3.12,"val_loss":3.20,"lr":0.00021,"toks":32768000,"toks_per_s":145000}
```

## I2. What to log during SDFT
SDFT needs additional logs:
- average reverse-KL per token
- generation length statistics (mean/min/max)
- teacher/student divergence stats
- replay fraction used (after Phase 8)
- gate decision (accept/reject update)

Example:
```json
{"step":500,"kl":0.42,"gen_len_mean":120,"replay":0.2,"gate":"accepted"}
```

## I3. What to log during TTT-Discover
TTT logs must include:
- problem id / hash
- step index (ttt step)
- beta value
- archive best reward
- reward distribution of rollouts (mean/std/max)
- chosen start state id (PUCT)
- adapter path

Example:
```json
{"problem":"abc123","step":10,"beta":7.3,"best_reward":0.81,"r_max":0.81,"r_mean":0.52,"start_state":42,"adapter":"adapter_step_010/lora.pt"}
```

## I4. Why this matters
With these logs you can:
- reproduce improvements
- detect regressions quickly
- build datasets for SDFT consolidation
- debug reward functions (distribution shape tells you if it’s too sparse)

## I5. Directory convention (repeat)
Use the `runs/` layout described earlier. Never overwrite a run directory; create a new one per run.



# Appendix: Roadmap (future work after MVP)

This repository is the MVP blueprint. Once the end-to-end loop works, here is a grounded roadmap.

## J1. Training stability improvements
- Add gradient checkpointing inside blocks (currently config flag exists but code may need explicit integration).
- Add fused optimizers or `torch.compile` for speed (guarded by flags).
- Add proper mixed-precision scaler for fp16 if needed.

## J2. Data quality improvements
- Deduplication pipeline (MinHash, SimHash).
- Document-level shuffling and packing with sampling weights.
- Domain-specific filtering and mixture tuning.

## J3. Model architecture upgrades
- Implement MoE (optional section above).
- Implement improved attention (e.g., grouped-query attention) if needed.
- Add sliding-window or longer-context methods.

## J4. Continual learning upgrades
- Implement replay with prioritized sampling (by recency and importance).
- Add parameter-efficient continual learning (e.g., adapters per domain).
- Add explicit “stability” objective: KL(θ_continual || θ0) regularization.

## J5. TTT-Discover upgrades
- Multi-step environments with intermediate rewards.
- Better state-action reuse: include top-k states and their best actions as context.
- Efficient rollouts: speculative decoding, caching KV.
- Better β control: clamp, per-state smoothing.

## J6. Consolidation upgrades
- Filter TTT discoveries by reward threshold.
- Cluster similar problems and consolidate per cluster.
- Use SDFT with multiple demonstrations (not just one).

## J7. Evaluation upgrades
- Add a real benchmark suite (domain-specific).
- Add long-context eval (if you train long context).
- Add regression dashboards.

The key: do not scale complexity until MVP is stable and reproducible.



# Appendix: Pseudocode (copy into code comments if needed)

This appendix provides explicit pseudocode that mirrors the intended implementation.

## K1. mHC residual wrapper (per sublayer)

Inputs:
- X: residual streams [B,T,n,C]
- sublayer F: maps [B,T,C] -> [B,T,C] (attention or MLP)

Pseudocode:

```
function MHCResidualForward(X):
    # compute mappings
    x_flat = reshape(X, [B,T,n*C])
    x_norm = RMSNorm(x_flat)

    hpre_tilde  = alpha_pre  * (x_norm @ phi_pre ) + b_pre      # [B,T,n]
    hpost_tilde = alpha_post * (x_norm @ phi_post) + b_post     # [B,T,n]
    hres_tilde  = alpha_res  * (x_norm @ phi_res ) + b_res      # [B,T,n*n]
    hres_tilde  = reshape(hres_tilde, [B,T,n,n])

    H_pre  = sigmoid(hpre_tilde)                # [B,T,n]
    H_post = 2 * sigmoid(hpost_tilde)           # [B,T,n]
    H_res  = Sinkhorn(exp(clamp(hres_tilde)))   # [B,T,n,n]

    # aggregate into single stream
    x_in = sum_i H_pre[i] * X[i]                # einsum -> [B,T,C]

    # apply sublayer
    y = F(x_in)                                 # [B,T,C]

    # residual mixing
    res = H_res @ X                              # einsum -> [B,T,n,C]

    # distribute output
    upd[i] = H_post[i] * y                       # einsum -> [B,T,n,C]

    return res + upd
```

## K2. Pretraining loop (next-token prediction)

```
initialize model θ
initialize optimizer AdamW

for step in 1..total_steps:
    for accum in 1..grad_accum_steps:
        (x, y) = sample_batch()             # x,y: [B,T]
        logits = model(x)                   # [B,T,V]
        loss = CE(logits, y) / grad_accum_steps
        backprop(loss)

    clip_grad_norm
    optimizer.step()
    optimizer.zero_grad()

    if step % eval_every == 0:
        compute val loss
    if step % save_every == 0:
        save checkpoint
```

## K3. SDFT loop (reverse KL distillation)

Inputs:
- dataset of (prompt x, demo c)
- base model initialized from θ0

```
initialize student model θ
initialize teacher model θ_ema = θ
freeze teacher gradients

for step in 1..total_steps:
    (x, c) = sample_example()

    # build contexts
    student_ctx = tokenize(x)
    teacher_ctx = tokenize(template(x,c))

    # on-policy sample from student
    y = sample_from_student(student_ctx)

    # compute logits along trajectory
    logits_student = student_forward(student_ctx + y)
    logits_teacher = teacher_forward(teacher_ctx + y)

    # align slices that predict y tokens
    slice_student = logits_student[Cs-1 : Cs-1+L]
    slice_teacher = logits_teacher[Ct-1 : Ct-1+L]

    # compute analytic reverse KL per token
    KL = sum_v p_s(v) (log p_s(v) - log p_t(v))
    loss = mean(KL)

    backprop(loss)
    optimizer.step()

    # EMA update teacher
    θ_ema = τ θ_ema + (1-τ) θ
```

Replay extension (Phase 8):
- with probability replay_ratio, sample (x,c) from buffer instead of new stream
- update buffer with new examples

## K4. TTT-Discover loop (entropic objective + reuse)

Inputs:
- environment with reward
- base model θ0
- LoRA-instrumented model θ = θ0 + Δ (only Δ trainable)

```
archive = { s0 = env.initial_state() }
archive_reward[s0] = env.reward(s0)

for t in 1..ttt_steps:

    # choose start state (reuse)
    if reuse_enabled:
        s = argmax_{s in archive} PUCT_score(s)
    else:
        s = s0

    # create context from archive (optional)
    c = env.context_from_archive(s, archive)

    # sample rollouts/actions
    for i in 1..N:
        prompt_i = make_prompt(problem, s, c)
        a_i = sample_action(π_θ(.|prompt_i))
        s_i = env.transition(a_i)
        r_i = env.reward(s_i)

    # compute beta by KL constraint on q_beta
    beta = solve_beta(rewards, gamma)

    # compute entropic advantages (LOO)
    A_ent[i] = LOO_advantage(rewards, beta)

    # compute logprobs under θ and θ0
    logp_theta[i] = log π_θ(a_i|prompt_i)
    logp_base[i]  = log π_θ0(a_i|prompt_i)

    # shape advantage with KL term
    A[i] = A_ent[i] - λ (logp_theta[i] - logp_base[i])

    # REINFORCE update LoRA params
    loss = -mean_i ( stopgrad(A[i]) * logp_theta[i] )
    backprop(loss)
    optimizer.step()

    # archive update: add top-2 new states from this batch
    add best s_i to archive
    update archive statistics for PUCT
```

This pseudocode matches the reference implementation structure.


# Appendix: Repository snapshot (source listings)

This appendix contains an exact snapshot of the scaffolded source files at the time this zip was generated.
If you modify code, you can re-run a snapshot generator later.



## `README.md`

```markdown
# LLM Blueprint: mHC + SDFT Continual Learning + TTT-Discover

This repository is a **handoff bundle** designed for an AI agent (e.g., Codex) to build an end-to-end system:

1) **Pretrain** a small decoder-only LLM **from scratch** (next-token prediction)  
2) Use **mHC (Manifold-Constrained Hyper-Connections)** to modify the residual stream topology  
3) Add **continual self-learning** via **SDFT (Self-Distillation Fine-Tuning)**  
4) Add **test-time training** for discovery problems via **TTT-Discover** (entropic objective + PUCT reuse)

The authoritative algorithmic details are in the PDFs in `docs/references/` and the main implementation blueprint:
`docs/CODEX_HANDOFF.md`.

## Quick start (local smoke test)

```bash
# 1) create venv
python -m venv .venv
source .venv/bin/activate

# 2) install
pip install -U pip
pip install -e .[dev]

# 3) run unit tests
pytest -q
```

## Folder structure

- `docs/` – long-form blueprint and checklists
- `docs/references/` – the 3 source papers (PDFs)
- `src/llm_mhc_sdft_tttd/` – minimal PyTorch implementation
- `scripts/` – CLI entrypoints
- `configs/` – pinned model configs (JSON)

## Important note on scale

A single RTX Pro 6000 (96GB VRAM) is enough to:
- build & verify the **full pipeline**
- pretrain **small models** (100M–900M) on limited data

Full-scale pretraining (multi-B tokens, multi-B params) still requires substantial compute + data.

For a complete, agent-readable plan (phases, tasks, and “definition of done”), see `docs/CODEX_HANDOFF.md`.

```



## `configs/model_mhc_120m.json`

```json
{
  "vocab_size": 32000,
  "n_layers": 12,
  "d_model": 768,
  "n_heads": 12,
  "d_head": 64,
  "d_ff": 2048,
  "mlp_act": "swiglu",
  "resid_dropout": 0.0,
  "attn_dropout": 0.0,
  "rmsnorm_eps": 1e-06,
  "rope_theta": 10000.0,
  "rope_partial_rotary_factor": 1.0,
  "max_seq_len": 2048,
  "tie_embeddings": true,
  "mhc": {
    "n_streams": 4,
    "alpha_init": 0.01,
    "sinkhorn_tmax": 20,
    "init_hpre": "uniform_sum1",
    "init_hpost": "uniform_sum1",
    "init_hres": "approx_identity",
    "sinkhorn_eps": 1e-06,
    "sinkhorn_clamp_min": -15.0,
    "sinkhorn_clamp_max": 15.0
  }
}
```



## `configs/model_mhc_350m.json`

```json
{
  "vocab_size": 32000,
  "n_layers": 24,
  "d_model": 1024,
  "n_heads": 16,
  "d_head": 64,
  "d_ff": 2816,
  "mlp_act": "swiglu",
  "resid_dropout": 0.0,
  "attn_dropout": 0.0,
  "rmsnorm_eps": 1e-06,
  "rope_theta": 10000.0,
  "rope_partial_rotary_factor": 1.0,
  "max_seq_len": 2048,
  "tie_embeddings": true,
  "mhc": {
    "n_streams": 4,
    "alpha_init": 0.01,
    "sinkhorn_tmax": 20,
    "init_hpre": "uniform_sum1",
    "init_hpost": "uniform_sum1",
    "init_hres": "approx_identity",
    "sinkhorn_eps": 1e-06,
    "sinkhorn_clamp_min": -15.0,
    "sinkhorn_clamp_max": 15.0
  }
}
```



## `configs/model_mhc_900m.json`

```json
{
  "vocab_size": 32000,
  "n_layers": 16,
  "d_model": 2048,
  "n_heads": 16,
  "d_head": 128,
  "d_ff": 5632,
  "mlp_act": "swiglu",
  "resid_dropout": 0.0,
  "attn_dropout": 0.0,
  "rmsnorm_eps": 1e-06,
  "rope_theta": 10000.0,
  "rope_partial_rotary_factor": 1.0,
  "max_seq_len": 2048,
  "tie_embeddings": true,
  "mhc": {
    "n_streams": 4,
    "alpha_init": 0.01,
    "sinkhorn_tmax": 20,
    "init_hpre": "uniform_sum1",
    "init_hpost": "uniform_sum1",
    "init_hres": "approx_identity",
    "sinkhorn_eps": 1e-06,
    "sinkhorn_clamp_min": -15.0,
    "sinkhorn_clamp_max": 15.0
  }
}
```



## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-mhc-sdft-tttd"
version = "0.1.0"
description = "From-scratch LLM blueprint combining mHC, SDFT continual learning, and TTT-Discover."
requires-python = ">=3.10"
dependencies = [
  "torch>=2.2.0",
  "sentencepiece>=0.1.99",
  "numpy>=1.26.0",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

```



## `scripts/prepare_data.py`

```python
#!/usr/bin/env python3
"""
Prepare packed binary token datasets for fast training.

Input:
  - raw text file(s), one document per line (or at least line-delimited)
  - sentencepiece model

Output:
  - .bin file of uint16/uint32 tokens
  - metadata json

Usage:
  python scripts/prepare_data.py \
    --tokenizer data/tokenizer/spm32k.model \
    --input data/raw/train.txt \
    --output data/packed/train.bin \
    --append_eos 1
"""
from __future__ import annotations

import argparse
import os
import json
from typing import List

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.data.tokenizer import SpmTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", required=True, help="Comma-separated text files")
    ap.add_argument("--output", required=True, help="Output .bin")
    ap.add_argument("--append_eos", type=int, default=1)
    ap.add_argument("--max_lines", type=int, default=-1, help="For debugging; -1 means all")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tok = SpmTokenizer(args.tokenizer)
    vocab_size = tok.vocab_size
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    tokens: List[int] = []
    n_docs = 0
    n_lines = 0

    for path in args.input.split(","):
        path = path.strip()
        if not path:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if args.max_lines > 0 and n_lines >= args.max_lines:
                    break
                line = line.strip("\n")
                if not line:
                    continue
                ids = tok.encode(line, add_bos=False, add_eos=False)
                tokens.extend(ids)
                if args.append_eos:
                    tokens.append(tok.eos_id())
                n_docs += 1
                n_lines += 1

    arr = np.array(tokens, dtype=dtype)
    arr.tofile(args.output)

    meta = {
        "tokenizer": args.tokenizer,
        "input": args.input,
        "output": args.output,
        "dtype": str(dtype),
        "vocab_size": vocab_size,
        "n_tokens": int(arr.shape[0]),
        "n_docs": n_docs,
    }
    with open(args.output + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote", args.output, "tokens:", arr.shape[0], "dtype:", dtype)


if __name__ == "__main__":
    main()

```



## `scripts/pretrain.py`

```python
#!/usr/bin/env python3
"""
Pretrain the mHC Transformer LM on a packed token dataset.

Usage:
  python scripts/pretrain.py \
    --train_bin data/packed/train.bin \
    --val_bin data/packed/val.bin \
    --out runs/pretrain_mhc_350m \
    --model configs/model_mhc_350m.json \
    --steps 200000
"""
from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.config import ModelConfig, PretrainConfig
from llm_mhc_sdft_tttd.training.pretrain import train_pretrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_bin", required=True)
    ap.add_argument("--val_bin", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="Path to model config json")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    args = ap.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        model_cfg = ModelConfig.from_json(f.read())

    pre_cfg = PretrainConfig(
        total_steps=args.steps,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_bs,
        grad_accum_steps=args.grad_accum,
        out_dir=args.out,
    )

    train_pretrain(
        model_cfg=model_cfg,
        train_data_path=args.train_bin,
        val_data_path=args.val_bin,
        out_dir=args.out,
        pre_cfg=pre_cfg,
    )


if __name__ == "__main__":
    main()

```



## `scripts/sdft.py`

```python
#!/usr/bin/env python3
"""
Run Self-Distillation Fine-Tuning (SDFT) continual learning.

Dataset format (jsonl):
  {"prompt": "...", "demonstration": "..."}  # one per line

Usage:
  python scripts/sdft.py \
    --ckpt runs/pretrain/ckpt_latest.pt \
    --tokenizer data/tokenizer/spm32k.model \
    --data data/sdft/demo.jsonl \
    --out runs/sdft_run
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.config import SDFTConfig
from llm_mhc_sdft_tttd.training.sdft import train_sdft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    args = ap.parse_args()

    cfg = SDFTConfig(total_steps=args.steps, out_dir=args.out)

    train_sdft(
        base_model_ckpt=args.ckpt,
        tokenizer_path=args.tokenizer,
        sdft_data_path=args.data,
        out_dir=args.out,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()

```



## `scripts/train_tokenizer.py`

```python
#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer from raw text files.

Usage:
  python scripts/train_tokenizer.py \
    --input data/raw/train.txt \
    --model_prefix data/tokenizer/spm32k \
    --vocab_size 32000 \
    --model_type bpe
"""
from __future__ import annotations

import argparse
import os

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Comma-separated list of input text files")
    ap.add_argument("--model_prefix", required=True, help="Output prefix (dir/prefix)")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--character_coverage", type=float, default=0.9995)
    ap.add_argument("--normalization_rule_name", type=str, default="nmt_nfkc_cf")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        normalization_rule_name=args.normalization_rule_name,
        # special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print("Wrote:", args.model_prefix + ".model", args.model_prefix + ".vocab")


if __name__ == "__main__":
    main()

```



## `scripts/ttt_discover.py`

```python
#!/usr/bin/env python3
"""
Run TTT-Discover on a toy environment (string matching).
This is only to validate the loop end-to-end. Replace with real environments.

Usage:
  python scripts/ttt_discover.py \
    --ckpt runs/pretrain/ckpt_latest.pt \
    --tokenizer data/tokenizer/spm32k.model \
    --out runs/ttt_toy \
    --target "HELLO"
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.config import TTTDiscoverConfig
from llm_mhc_sdft_tttd.training.ttt_discover import ToyStringMatchEnv, run_ttt_discover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--rollouts", type=int, default=64)
    args = ap.parse_args()

    env = ToyStringMatchEnv(problem_description="Guess the hidden target string.", target=args.target)
    cfg = TTTDiscoverConfig(out_dir=args.out, ttt_steps=args.steps, rollouts_per_step=args.rollouts)

    result = run_ttt_discover(
        base_ckpt=args.ckpt,
        tokenizer_path=args.tokenizer,
        env=env,
        cfg=cfg,
    )

    print("BEST:", result["best_reward"], result["best_state"])


if __name__ == "__main__":
    main()

```



## `scripts/_bootstrap.py`

```python
"""Bootstrap helper for running repo scripts without installing the package.

This repository is meant to be installed via:

    pip install -e .[dev]

However, editable installs can fail or be skipped in some environments (CI
sandboxes, locked-down containers, etc.). To keep the CLI scripts runnable
without relying on editable install state, this module can be imported from
any script under `scripts/`:

    from _bootstrap import bootstrap
    bootstrap()

It will add `<repo_root>/src` to `sys.path` if needed.

This is intentionally small and dependency-free.
"""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap() -> None:
    """Add `<repo_root>/src` to `sys.path` if not already present."""

    # scripts/_bootstrap.py -> scripts/ -> repo root
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if not src.exists():
        raise RuntimeError(
            f"Expected src directory at {src} (repo root: {repo_root}). "
            "Run scripts from the repo root, e.g. `python scripts/pretrain.py ...`."
        )
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

```



## `src/llm_mhc_sdft_tttd/__init__.py`

```python
__version__ = '0.1.0'

```



## `src/llm_mhc_sdft_tttd/config.py`

```python
"""
Project: LLM from scratch combining:
- mHC (Manifold-Constrained Hyper-Connections)
- SDFT continual learning (Self-Distillation Fine-Tuning)
- TTT-Discover (Learning to Discover at Test Time)

This file defines configuration dataclasses that are intentionally explicit.
Codex (or any agent) should NOT invent new parameters; add them here first.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Dict, Any

import json


DTypeStr = Literal["fp32", "bf16", "fp16"]


@dataclass
class TokenizerConfig:
    """SentencePiece tokenizer training + usage config."""
    vocab_size: int = 32000
    model_type: Literal["bpe", "unigram"] = "bpe"
    character_coverage: float = 0.9995
    # Special tokens (must be consistent everywhere)
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    # text normalization
    normalization_rule_name: str = "nmt_nfkc_cf"


@dataclass
class MHCConfig:
    """
    mHC hyperparameters (from the mHC paper, adapted to our single-GPU context).

    In the paper, they use expansion rate n=4, gating init alpha=0.01, Sinkhorn tmax=20. 
    (see Table 5 and Eq. 8-9 in the paper).
    """
    n_streams: int = 4  # expansion rate n
    alpha_init: float = 0.01  # gating factor init alpha
    sinkhorn_tmax: int = 20
    # initialization choices (not fully specified in the paper; we make explicit)
    init_hpre: Literal["uniform_sum1", "sigmoid_half"] = "uniform_sum1"
    init_hpost: Literal["uniform_sum1"] = "uniform_sum1"
    init_hres: Literal["approx_identity", "uniform"] = "approx_identity"
    # numerical stability
    sinkhorn_eps: float = 1e-6
    sinkhorn_clamp_min: float = -15.0
    sinkhorn_clamp_max: float = 15.0


@dataclass
class ModelConfig:
    """
    Decoder-only Transformer LM configuration (LLaMA-like),
    plus mHC residual-stream expansion.

    NOTE: This is intentionally dense-model only by default. 
    The mHC paper uses a DeepSeek-V3-inspired MoE architecture; implementing MoE is optional.
    """
    # vocab
    vocab_size: int = 32000
    # transformer
    n_layers: int = 16
    d_model: int = 2048
    n_heads: int = 16
    d_head: int = 128  # must satisfy d_model = n_heads * d_head
    # MLP
    d_ff: int = 5632  # typical 2.75x expansion for SwiGLU
    mlp_act: Literal["swiglu", "gelu"] = "swiglu"
    # dropout
    resid_dropout: float = 0.0
    attn_dropout: float = 0.0
    # norms
    rmsnorm_eps: float = 1e-6
    # RoPE
    rope_theta: float = 10000.0
    rope_partial_rotary_factor: float = 1.0  # 1.0 = full rotary
    max_seq_len: int = 2048
    # output head tie
    tie_embeddings: bool = True
    # mHC
    # IMPORTANT: use default_factory to avoid mutable-default dataclass error.
    mhc: MHCConfig = field(default_factory=MHCConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "ModelConfig":
        d = json.loads(s)
        # nested dataclass reconstruction
        mhc = MHCConfig(**d.pop("mhc"))
        return ModelConfig(mhc=mhc, **d)


@dataclass
class PretrainConfig:
    """
    Pretraining config for next-token prediction.
    """
    seed: int = 1337
    dtype: DTypeStr = "bf16"
    device: str = "cuda"
    # optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    # schedule
    warmup_steps: int = 2000
    total_steps: int = 200_000
    lr_decay: Literal["cosine", "linear", "constant"] = "cosine"
    # batch
    micro_batch_size: int = 2
    grad_accum_steps: int = 16
    # data
    seq_len: int = 2048
    # logging/checkpoint
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    out_dir: str = "runs/pretrain"
    # stability toggles
    use_compile: bool = False
    gradient_checkpointing: bool = True


@dataclass
class SDFTConfig:
    """
    Continual learning config (Self-Distillation Fine-Tuning).
    Implements reverse KL distillation from a demonstration-conditioned teacher.
    """
    seed: int = 1337
    dtype: DTypeStr = "bf16"
    device: str = "cuda"

    # generation (student policy sampling)
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95

    # distillation loss
    kl_coef: float = 1.0  # scales the reverse-KL loss
    # teacher construction
    teacher_ema_decay: float = 0.999
    # optimization
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0

    # training loop
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    # IMPORTANT: `total_steps`, `log_every`, `save_every` are *optimizer steps*
    # (after gradient accumulation), matching the pretrain loop.
    total_steps: int = 10_000
    log_every: int = 10
    save_every: int = 500
    out_dir: str = "runs/sdft"

    # replay (optional)
    replay_ratio: float = 0.2  # fraction of batches from replay buffer/dataset
    replay_buffer_size: int = 50_000


@dataclass
class TTTDiscoverConfig:
    """
    Test-time training config (Learning to Discover at Test Time).
    We implement an entropic policy optimization objective (Eq. 1/2 in the paper),
    with optional adaptive beta and PUCT-based state reuse.
    """
    seed: int = 1337
    dtype: DTypeStr = "bf16"
    device: str = "cuda"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # which modules to LoRA-adapt
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")

    # RL / optimization steps
    ttt_steps: int = 50
    rollouts_per_step: int = 64  # reduce for single GPU
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95

    # entropic objective
    beta: float = 10.0  # constant beta; adaptive beta optional
    adaptive_beta: bool = True
    # Adaptive beta is set by constraining the KL of entropic weights q_beta against the uniform batch (Appendix A.1).
    # We default to gamma=ln(2), matching the paper.
    target_kl_gamma: float = 0.6931471805599453
    # regularization to base (theta0)
    kl_penalty_lambda: float = 0.1

    # state reuse via PUCT
    reuse_enabled: bool = True
    puct_c: float = 2.0

    # buffer
    buffer_max_size: int = 4096
    # optimization
    lr: float = 5e-5
    grad_clip: float = 1.0

    out_dir: str = "runs/ttt_discover"


def dump_config(path: str, cfg: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if hasattr(cfg, "to_json"):
            f.write(cfg.to_json())
        else:
            f.write(json.dumps(asdict(cfg), indent=2))

```



## `src/llm_mhc_sdft_tttd/data/dataset.py`

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PackedTokenDataset(Dataset):
    """
    Memory-mapped token dataset for next-token prediction.

    File format:
      - raw binary file containing token ids in little-endian uint16 or uint32
      - shape: [N] tokens

    This dataset returns (x, y) where:
      x: [seq_len]
      y: [seq_len] shifted by 1
    """
    path: str
    seq_len: int
    dtype: str = "uint16"  # or uint32
    seed: int = 1337

    def __post_init__(self):
        assert self.dtype in ("uint16", "uint32")
        np_dtype = np.uint16 if self.dtype == "uint16" else np.uint32
        self.data = np.memmap(self.path, dtype=np_dtype, mode="r")
        self.n_tokens = self.data.shape[0]
        assert self.n_tokens > self.seq_len + 1, "Dataset too small for seq_len"
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        # approximate number of samples; random sampling anyway
        return self.n_tokens // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # random offset
        start = self.rng.integers(0, self.n_tokens - (self.seq_len + 1))
        chunk = np.array(self.data[start : start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

```



## `src/llm_mhc_sdft_tttd/data/sdft_dataset.py`

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from torch.utils.data import Dataset


@dataclass
class SDFTExample:
    prompt: str
    demonstration: str


class SDFTJsonlDataset(Dataset):
    """
    Reads jsonl with fields:
      - "prompt": string
      - "demonstration": string
    """
    def __init__(self, path: str):
        self.path = path
        self.items: List[SDFTExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(SDFTExample(prompt=obj["prompt"], demonstration=obj["demonstration"]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SDFTExample:
        return self.items[idx]


def identity_collate(batch):
    """Return the batch as-is.

    Why?
      `SDFTJsonlDataset` yields `SDFTExample` dataclass objects. PyTorch's default
      `collate_fn` does not know how to stack / collate arbitrary dataclass
      instances and will raise a `TypeError` during DataLoader iteration.

      For SDFT we intentionally keep examples as simple Python objects
      (prompt/demonstration strings). Returning a Python list is sufficient.

    Important:
      This is a *top-level* function so it is picklable for multi-worker
      DataLoaders.
    """
    return batch


def make_teacher_prompt(prompt: str, demonstration: str) -> str:
    """
    Prompt template described in the SDFT paper (Sec. 3).
    """
    return (
        "<Question>\n"
        f"{prompt}\n\n"
        "This is an example for a response to the question:\n"
        "<Demonstration>\n"
        f"{demonstration}\n\n"
        "Now answer with a response of your own, including the thinking process:\n"
    )

```



## `src/llm_mhc_sdft_tttd/data/tokenizer.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import sentencepiece as spm


@dataclass
class SpmTokenizer:
    """
    Thin wrapper around SentencePiece.
    """
    model_path: str

    def __post_init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def bos_id(self) -> int:
        return self.sp.bos_id()

    def eos_id(self) -> int:
        return self.sp.eos_id()

    def pad_id(self) -> int:
        return self.sp.pad_id()

    def unk_id(self) -> int:
        return self.sp.unk_id()

```



## `src/llm_mhc_sdft_tttd/eval/perplexity.py`

```python
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.transformer import MHCTransformerLM
from ..data.dataset import PackedTokenDataset


@torch.no_grad()
def perplexity(model: MHCTransformerLM, data_path: str, seq_len: int, batch_size: int, device: str = "cuda", dtype: str = "uint16", max_batches: int = 100) -> float:
    ds = PackedTokenDataset(data_path, seq_len=seq_len, dtype=dtype, seed=123)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    losses = []
    model.eval()
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    mean_loss = sum(losses) / max(1, len(losses))
    return math.exp(min(20, mean_loss))

```



## `src/llm_mhc_sdft_tttd/model/layers.py`

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm_x = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return x * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (x1, x2) -> (-x2, x1)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """
    Rotary positional embeddings (RoPE). Applies to query and key.

    We generate cached cos/sin up to max_seq_len. Supports partial rotary via rotary_dim.
    """
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        # inv_freq shape [dim/2]
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq, dim]
        cos = emb.cos()[None, None, :, :]  # [1,1,seq,dim]
        sin = emb.sin()[None, None, :, :]  # [1,1,seq,dim]
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B, n_heads, T, head_dim]
        if self.cos_cached is None or seq_len > self.cos_cached.shape[2]:
            self._build_cache(max(seq_len, self.max_seq_len))
        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, max_seq_len: int, rope_theta: float, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model == n_heads * d_head, "d_model must equal n_heads*d_head"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.rope = RoPE(dim=d_head, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,T,C]
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, nh, T, dh]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k = self.rope(q, k, seq_len=T)

        # PyTorch SDPA expects [B, nh, T, dh]
        # use is_causal=True to apply causal mask. If attn_mask provided, merge.
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        out = self.o_proj(out)
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # LLaMA style:
        # gate_proj: d_model -> d_ff
        # up_proj: d_model -> d_ff
        # down_proj: d_ff -> d_model
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, act: str = "swiglu"):
        super().__init__()
        if act == "swiglu":
            self.mlp = SwiGLU(d_model, d_ff)
        elif act == "gelu":
            self.fc1 = nn.Linear(d_model, d_ff, bias=False)
            self.fc2 = nn.Linear(d_ff, d_model, bias=False)
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown act: {act}")
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name == "swiglu":
            return self.mlp(x)
        else:
            return self.fc2(self.act(self.fc1(x)))

```



## `src/llm_mhc_sdft_tttd/model/lora.py`

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear (bias-free by default).
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        # LoRA weights
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

        # init (LoRA paper: A random, B zeros)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

        # freeze base
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.B(self.A(self.dropout(x)))


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """
    Given 'a.b.c', return (module at 'a.b', 'c').
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    target_module_suffixes: Iterable[str],
    r: int,
    alpha: int,
    dropout: float = 0.0,
) -> List[str]:
    """
    Replace every nn.Linear whose module name ends with any of the suffixes in target_module_suffixes
    with LoRALinear.

    Returns list of replaced module names.
    """
    suffixes = tuple(target_module_suffixes)
    replaced = []
    # we must iterate over named_modules, but replace needs parent access; collect first
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(suffixes):
            candidates.append(name)
    for name in candidates:
        parent, attr = _get_parent_module(model, name)
        base = getattr(parent, attr)
        wrapped = LoRALinear(base=base, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        replaced.append(name)
    return replaced


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters (A and B matrices) from model.
    """
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.A.weight"] = module.A.weight.detach().cpu()
            sd[f"{name}.B.weight"] = module.B.weight.detach().cpu()
    return sd


def load_lora_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    """
    Load LoRA parameters into an already LoRA-instrumented model.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.A.weight"
            b_key = f"{name}.B.weight"
            if a_key in sd:
                module.A.weight.data.copy_(sd[a_key].to(module.A.weight.device))
            if b_key in sd:
                module.B.weight.data.copy_(sd[b_key].to(module.B.weight.device))


def mark_only_lora_trainable(model: nn.Module) -> None:
    """
    Freeze everything except LoRA params.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.weight.requires_grad_(True)
            module.B.weight.requires_grad_(True)


def save_lora(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(lora_state_dict(model), path)


def load_lora(model: nn.Module, path: str) -> None:
    sd = torch.load(path, map_location="cpu")
    load_lora_state_dict(model, sd)

```



## `src/llm_mhc_sdft_tttd/model/mhc.py`

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MHCConfig
from .layers import RMSNorm


def sinkhorn_knopp(
    log_alpha: torch.Tensor,
    tmax: int = 20,
    eps: float = 1e-6,
    clamp_min: float = -15.0,
    clamp_max: float = 15.0,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp projection onto the Birkhoff polytope (doubly-stochastic matrices).

    Paper mapping:
      M(0) = exp(H~_res)
      M(t) = Tr( Tc( M(t-1) ) )
    where Tc normalizes columns to sum to 1 and Tr normalizes rows to sum to 1. (Eq. 9)

    Here we operate in normal space (not log-space) but we clamp log_alpha for numerical stability.
    Input:
        log_alpha: [..., n, n] unconstrained (real-valued).
    Output:
        P: [..., n, n] approximately doubly stochastic
    """
    # clamp before exp to avoid overflow
    log_alpha = torch.clamp(log_alpha, clamp_min, clamp_max)
    P = torch.exp(log_alpha)  # positive
    # iterative normalization
    for _ in range(tmax):
        # column normalization
        col_sum = P.sum(dim=-2, keepdim=True)  # sum over rows -> shape [..., 1, n]
        P = P / (col_sum + eps)
        # row normalization
        row_sum = P.sum(dim=-1, keepdim=True)  # shape [..., n, 1]
        P = P / (row_sum + eps)
    return P


class MHCMapping(nn.Module):
    """
    Computes (H_pre, H_post, H_res) for a given residual stream X.

    Shapes:
      X: [B, T, n, C]
      flatten: [B, T, n*C]
      H_pre:  [B, T, n]
      H_post: [B, T, n]
      H_res:  [B, T, n, n]

    Based on Eq. (7) and Eq. (8) in the paper.
    """
    def __init__(self, d_model: int, cfg: MHCConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.n_streams
        self.n = n
        self.d_model = d_model
        self.d_flat = n * d_model

        self.norm = RMSNorm(self.d_flat, eps=1e-6)  # separate eps; can be config if needed

        # dynamic mapping projections (phi matrices)
        self.phi_pre = nn.Linear(self.d_flat, n, bias=False)
        self.phi_post = nn.Linear(self.d_flat, n, bias=False)
        self.phi_res = nn.Linear(self.d_flat, n * n, bias=False)

        # static biases
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))

        # gating alphas (scalars)
        self.alpha_pre = nn.Parameter(torch.tensor(cfg.alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(cfg.alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(cfg.alpha_init))

        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        The paper specifies alpha init and Sinkhorn tmax/n.
        Bias init is not fully specified; we make it explicit and stable:

        - H_pre = sigmoid(H~_pre). To make initial aggregation sum to 1 (uniform),
          set b_pre = logit(1/n). Then sigmoid(b_pre)=1/n.
        - H_post = 2*sigmoid(H~_post). To make initial distribution sum to 1 (uniform),
          we want H_post_i = 1/n => 2*sigmoid(b_post)=1/n => sigmoid(b_post)=1/(2n).
        - H_res is Sinkhorn(exp(...)). To approximate identity mapping at init:
          set b_res diagonal high and off-diagonal low. After Sinkhorn, this becomes close to a permutation.
        """
        n = self.n
        if self.cfg.init_hpre == "uniform_sum1":
            p = 1.0 / n
            self.b_pre.data.fill_(math.log(p / (1 - p)))
        elif self.cfg.init_hpre == "sigmoid_half":
            self.b_pre.data.zero_()

        if self.cfg.init_hpost == "uniform_sum1":
            p = 1.0 / (2 * n)
            self.b_post.data.fill_(math.log(p / (1 - p)))

        if self.cfg.init_hres == "approx_identity":
            # diag = +2, off = -2 (tunable)
            self.b_res.data.fill_(-2.0)
            self.b_res.data.diagonal().fill_(2.0)
        elif self.cfg.init_hres == "uniform":
            self.b_res.data.zero_()

        # init linear weights
        for lin in [self.phi_pre, self.phi_post, self.phi_res]:
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, n, C = X.shape
        assert n == self.n and C == self.d_model, f"Expected X shape [B,T,{self.n},{self.d_model}]"
        x_flat = X.reshape(B, T, n * C)
        x_norm = self.norm(x_flat)

        hpre_tilde = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre  # [B,T,n]
        hpost_tilde = self.alpha_post * self.phi_post(x_norm) + self.b_post  # [B,T,n]
        hres_tilde = self.alpha_res * self.phi_res(x_norm).reshape(B, T, n, n) + self.b_res  # [B,T,n,n]

        H_pre = torch.sigmoid(hpre_tilde)  # (0,1)
        H_post = 2.0 * torch.sigmoid(hpost_tilde)  # (0,2)
        H_res = sinkhorn_knopp(
            hres_tilde,
            tmax=self.cfg.sinkhorn_tmax,
            eps=self.cfg.sinkhorn_eps,
            clamp_min=self.cfg.sinkhorn_clamp_min,
            clamp_max=self.cfg.sinkhorn_clamp_max,
        )
        return H_pre, H_post, H_res


class MHCResidual(nn.Module):
    """
    Wrap a sublayer F (attention or MLP) with mHC residual update.

    Update:
      X_next = H_res * X + H_post^T * F( H_pre * X )

    Where:
      - X is the n-stream residual: [B,T,n,C]
      - H_pre aggregates to [B,T,C]
      - F outputs [B,T,C]
      - H_post distributes back to streams [B,T,n,C]
      - H_res mixes streams [B,T,n,n] -> [B,T,n,C]
    """
    def __init__(self, d_model: int, cfg: MHCConfig, sublayer: nn.Module):
        super().__init__()
        self.mapping = MHCMapping(d_model=d_model, cfg=cfg)
        self.sublayer = sublayer

    def forward(self, X: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        H_pre, H_post, H_res = self.mapping(X)  # shapes above

        # aggregate: x_in = sum_i H_pre_i * X_i
        x_in = torch.einsum("btn,btnc->btc", H_pre, X)  # [B,T,C]

        # apply sublayer
        if attn_mask is None:
            y = self.sublayer(x_in)
        else:
            # attention sublayer expects attn_mask
            y = self.sublayer(x_in, attn_mask=attn_mask)

        # residual mixing: res = H_res @ X
        res = torch.einsum("btnm,btmc->btnc", H_res, X)  # [B,T,n,C] with n==m

        # distribute: upd_i = H_post_i * y
        upd = torch.einsum("btn,btc->btnc", H_post, y)
        return res + upd

```



## `src/llm_mhc_sdft_tttd/model/transformer.py`

```python
from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .layers import RMSNorm, CausalSelfAttention, MLP
from .mhc import MHCResidual


class AttentionSublayer(nn.Module):
    """Pre-norm attention sublayer."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            max_seq_len=cfg.max_seq_len,
            rope_theta=cfg.rope_theta,
            attn_dropout=cfg.attn_dropout,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        return self.attn(x, attn_mask=attn_mask)


class MLPSublayer(nn.Module):
    """Pre-norm MLP sublayer."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, act=cfg.mlp_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.mlp(x)


class MHCTransformerBlock(nn.Module):
    """
    Transformer block with mHC residual stream for both attention and MLP.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_res = MHCResidual(cfg.d_model, cfg.mhc, AttentionSublayer(cfg))
        self.mlp_res = MHCResidual(cfg.d_model, cfg.mhc, MLPSublayer(cfg))

    def forward(self, X: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        X = self.attn_res(X, attn_mask=attn_mask)
        X = self.mlp_res(X, attn_mask=None)
        return X


class MHCTransformerLM(nn.Module):
    """
    Decoder-only LM with mHC residual streams.

    Forward inputs:
      input_ids: [B,T]
    Output:
      logits: [B,T,V]
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.n_streams = cfg.mhc.n_streams
        # stream embeddings to break symmetry across streams (learnable)
        self.stream_emb = nn.Parameter(torch.zeros(self.n_streams, cfg.d_model))

        self.blocks = nn.ModuleList([MHCTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.stream_emb, mean=0.0, std=0.02)
        # output head weight is tied; no init needed.

    def _init_streams(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C] -> X: [B,T,n,C]
        """
        B, T, C = x.shape
        n = self.n_streams
        X = x.unsqueeze(2).repeat(1, 1, n, 1)
        X = X + self.stream_emb.view(1, 1, n, C)
        return X

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)  # [B,T,C]
        X = self._init_streams(x)

        for blk in self.blocks:
            X = blk(X, attn_mask=attn_mask)

        # readout: mean across streams (stable due to doubly-stochastic H_res)
        x_out = X.mean(dim=2)  # [B,T,C]
        x_out = self.final_norm(x_out)
        logits = self.lm_head(x_out)  # [B,T,V]
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        *,
        prompt_lens: Optional[List[int]] = None,
        pad_token_id: int = 0,
        return_lens: bool = False,
        forbid_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Nucleus sampling generation that correctly handles right-padded prompts.

        Important: the *old* implementation assumed `input_ids` had no padding and always
        sampled from the last column (`logits[:, -1]`). That is **wrong** when prompts in
        a batch have different lengths and are right-padded: the last column may be a PAD
        token for shorter prompts, causing the model to generate after PAD instead of after
        the real prompt.

        This implementation supports a padded prompt batch by requiring `prompt_lens`.

        Args:
          input_ids: [B, T_prompt_max] with prompts right-padded by `pad_token_id`.
          prompt_lens: true prompt lengths (excluding padding). If None, assumes no padding.
          pad_token_id: id used for padding.
          return_lens: if True, also return per-sample output lengths.
          forbid_token_ids: token ids that must never be sampled (defaults to [pad_token_id]).

        Returns:
          ids: [B, T_out_max] padded on the right.
          (optional) lens: List[int] output lengths per sample.
        """

        self.eval()
        device = input_ids.device
        B, T = input_ids.shape

        if prompt_lens is None:
            prompt_lens = [T] * B
        if len(prompt_lens) != B:
            raise ValueError(f"prompt_lens must have length B={B}, got {len(prompt_lens)}")
        if max(prompt_lens) <= 0:
            raise ValueError("prompt_lens must be >= 1 for all samples")

        # Default: never sample PAD.
        if forbid_token_ids is None:
            forbid_token_ids = [pad_token_id]
        # de-duplicate while preserving order
        forbid_token_ids = list(dict.fromkeys(forbid_token_ids))

        max_prompt = max(int(x) for x in prompt_lens)
        total_cap = max_prompt + max_new_tokens
        out = torch.full((B, total_cap), pad_token_id, dtype=input_ids.dtype, device=device)
        for i in range(B):
            L = int(prompt_lens[i])
            out[i, :L] = input_ids[i, :L]

        cur_lens = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Build a conditional batch by taking up to the last `max_seq_len` tokens
            # for each sample, and right-padding to a common window length.
            cur_max = int(cur_lens.max().item())
            window_len = min(cur_max, int(self.cfg.max_seq_len))
            if window_len <= 0:
                raise RuntimeError("window_len <= 0; check prompt_lens")

            ids_cond = torch.full((B, window_len), pad_token_id, dtype=input_ids.dtype, device=device)
            last_pos = torch.zeros(B, dtype=torch.long, device=device)
            for i in range(B):
                Li = int(cur_lens[i].item())
                start = max(0, Li - window_len)
                seg = out[i, start:Li]
                seg_len = int(seg.numel())
                if seg_len <= 0:
                    raise RuntimeError("empty segment in generate(); check prompt_lens")
                ids_cond[i, :seg_len] = seg
                last_pos[i] = seg_len - 1

            logits = self(ids_cond)  # [B, window_len, V]
            next_logits = logits[torch.arange(B, device=device), last_pos, :]
            # Forbid certain tokens from being generated.
            if forbid_token_ids:
                next_logits[:, forbid_token_ids] = -1e9

            next_logits = next_logits / max(temperature, 1e-6)
            probs = F.softmax(next_logits, dim=-1)

            # top-p (nucleus) sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[:, 0] = False  # keep at least one token
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_idx).squeeze(1)  # [B]

            # Only extend active (not-finished) sequences.
            active = ~finished
            if active.any():
                rows = torch.arange(B, device=device)[active]
                cols = cur_lens[active]
                out[rows, cols] = next_token[active]
                if eos_token_id is not None:
                    finished = finished | (active & (next_token == eos_token_id))
                cur_lens = cur_lens + active.long()

            if eos_token_id is not None and finished.all():
                break

        max_out = int(cur_lens.max().item())
        out = out[:, :max_out]
        if return_lens:
            return out, cur_lens.tolist()
        return out

```



## `src/llm_mhc_sdft_tttd/training/pretrain.py`

```python
from __future__ import annotations

import os
import math
import time
import json
from dataclasses import asdict
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import ModelConfig, PretrainConfig
from ..model.transformer import MHCTransformerLM
from ..data.dataset import PackedTokenDataset


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(out_dir: str, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfgs: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"ckpt_step_{step:07d}.pt")
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfgs": cfgs,
    }
    torch.save(payload, ckpt_path)
    # Always write a stable 'latest' checkpoint file (do NOT rely on symlinks).
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    torch.save(payload, latest)
    return ckpt_path


@torch.no_grad()
def eval_loss(model: torch.nn.Module, dl: DataLoader, device: str, max_batches: int = 20) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train_pretrain(
    model_cfg: ModelConfig,
    train_data_path: str,
    val_data_path: Optional[str],
    out_dir: str,
    pre_cfg: PretrainConfig,
    train_dtype: str = "uint16",
    val_dtype: str = "uint16",
) -> None:
    set_seed(pre_cfg.seed)
    device = pre_cfg.device

    model = MHCTransformerLM(model_cfg).to(device)

    # dtype
    if pre_cfg.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif pre_cfg.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pre_cfg.lr,
        betas=pre_cfg.betas,
        eps=pre_cfg.eps,
        weight_decay=pre_cfg.weight_decay,
    )

    train_ds = PackedTokenDataset(train_data_path, seq_len=pre_cfg.seq_len, dtype=train_dtype, seed=pre_cfg.seed)
    train_dl = DataLoader(train_ds, batch_size=pre_cfg.micro_batch_size, num_workers=2, pin_memory=True)

    if val_data_path:
        val_ds = PackedTokenDataset(val_data_path, seq_len=pre_cfg.seq_len, dtype=val_dtype, seed=pre_cfg.seed + 1)
        val_dl = DataLoader(val_ds, batch_size=pre_cfg.micro_batch_size, num_workers=2, pin_memory=True)
    else:
        val_dl = None

    # optional compile
    if pre_cfg.use_compile:
        model = torch.compile(model)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        f.write(model_cfg.to_json())
    with open(os.path.join(out_dir, "pretrain_config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(pre_cfg), indent=2))

    step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x, y) in enumerate(train_dl):
        if step >= pre_cfg.total_steps:
            break

        # lr schedule
        lr = cosine_lr(step, pre_cfg.total_steps, pre_cfg.lr, pre_cfg.warmup_steps) if pre_cfg.lr_decay == "cosine" else pre_cfg.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / pre_cfg.grad_accum_steps

        loss.backward()

        if (batch_idx + 1) % pre_cfg.grad_accum_steps == 0:
            if pre_cfg.grad_clip is not None and pre_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), pre_cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # logging
            if step % pre_cfg.log_every == 0:
                dt = time.time() - t0
                toks = pre_cfg.micro_batch_size * pre_cfg.grad_accum_steps * pre_cfg.seq_len
                toks_per_s = toks / max(1e-6, dt)
                print(f"[step {step}] loss={loss.item()*pre_cfg.grad_accum_steps:.4f} lr={lr:.3e} toks/s={toks_per_s:.1f}")
                t0 = time.time()

            # eval
            if val_dl is not None and step % pre_cfg.eval_every == 0:
                vloss = eval_loss(model, val_dl, device=device, max_batches=20)
                print(f"[step {step}] val_loss={vloss:.4f} ppl={math.exp(min(20, vloss)):.2f}")

            # checkpoint
            if step % pre_cfg.save_every == 0:
                ckpt = save_checkpoint(
                    out_dir=out_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    cfgs={"model": asdict(model_cfg), "pretrain": asdict(pre_cfg)},
                )
                print(f"[step {step}] saved {ckpt}")

    # Always write a final checkpoint + ckpt_latest, even for very short runs
    # (e.g., smoke tests with total_steps < save_every).
    ckpt = save_checkpoint(
        out_dir=out_dir,
        step=step,
        model=model,
        optimizer=optimizer,
        cfgs={"model": asdict(model_cfg), "pretrain": asdict(pre_cfg)},
    )
    print(f"[final step {step}] saved {ckpt}")

```



## `src/llm_mhc_sdft_tttd/training/sdft.py`

```python
from __future__ import annotations

import os
import math
import time
import json
from dataclasses import asdict
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import ModelConfig, SDFTConfig
from ..model.transformer import MHCTransformerLM
from ..data.tokenizer import SpmTokenizer
from ..data.sdft_dataset import SDFTJsonlDataset, make_teacher_prompt, identity_collate


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def pad_to_max(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def compute_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Reverse KL: KL(p_s || p_t) = sum_v p_s(v) [log p_s(v) - log p_t(v)]

    student_logits, teacher_logits: [B, L, V]
    Returns:
      kl: [B, L] (per position)
    """
    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_t = F.log_softmax(teacher_logits.detach(), dim=-1)
    p_s = log_p_s.exp()
    kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
    return kl


def train_sdft(
    base_model_ckpt: str,
    tokenizer_path: str,
    sdft_data_path: str,
    out_dir: str,
    cfg: SDFTConfig,
    model_cfg: Optional[ModelConfig] = None,
) -> None:
    """
    SDFT continual learning training loop.

    Inputs:
      base_model_ckpt: path to a checkpoint saved by pretraining (ckpt_step_*.pt) OR a HF-like state_dict.
      tokenizer_path: sentencepiece model file
      sdft_data_path: jsonl dataset path
    """
    set_seed(cfg.seed)
    device = cfg.device
    tok = SpmTokenizer(tokenizer_path)

    # load model config from checkpoint dir if not provided
    if model_cfg is None:
        # expect sibling file model_config.json
        ckpt_dir = os.path.dirname(base_model_ckpt)
        cfg_path = os.path.join(ckpt_dir, "model_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                model_cfg = ModelConfig.from_json(f.read())
        else:
            raise ValueError("model_cfg not provided and model_config.json not found")

    model = MHCTransformerLM(model_cfg).to(device)
    ema_teacher = MHCTransformerLM(model_cfg).to(device)

    # load weights
    payload = torch.load(base_model_ckpt, map_location="cpu")
    if "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
        ema_teacher.load_state_dict(payload["model"], strict=True)
    else:
        model.load_state_dict(payload, strict=True)
        ema_teacher.load_state_dict(payload, strict=True)

    ema_teacher.eval()
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    # dtype
    if cfg.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    ds = SDFTJsonlDataset(sdft_data_path)
    dl = DataLoader(
        ds,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=identity_collate,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sdft_config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(cfg), indent=2))
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        f.write(model_cfg.to_json())

    # IMPORTANT: `total_steps`, `log_every`, and `save_every` are defined in
    # *optimizer steps* (i.e., after gradient accumulation), not micro-batches.
    opt_step = 0
    micro_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    # NOTE: for simplicity, we do not implement replay buffer here; see docs for extension.

    while opt_step < cfg.total_steps:
        for batch in dl:
            if opt_step >= cfg.total_steps:
                break

            # build contexts
            prompts = [ex.prompt for ex in batch] if isinstance(batch, list) else [batch.prompt]
            demos = [ex.demonstration for ex in batch] if isinstance(batch, list) else [batch.demonstration]

            # student context: just the question/prompt
            student_ctx = [tok.encode(p, add_bos=True, add_eos=False) for p in prompts]
            # teacher context: question + demonstration in template
            teacher_ctx = [tok.encode(make_teacher_prompt(p, d), add_bos=True, add_eos=False) for p, d in zip(prompts, demos)]

            # generate from student
            prompt_lens = [len(s) for s in student_ctx]
            student_ctx_tensor = pad_to_max(student_ctx, pad_id=tok.pad_id()).to(device)
            # IMPORTANT: prompts are right-padded; pass `prompt_lens` so generation starts
            # immediately after the true prompt, not after PAD.
            gen, gen_lens = model.generate(
                student_ctx_tensor,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=tok.eos_id(),
                prompt_lens=prompt_lens,
                pad_token_id=tok.pad_id(),
                return_lens=True,
            )
            # split out generated tokens (remove context)
            gen_tokens = []
            for i in range(gen.shape[0]):
                ctx_len = prompt_lens[i]
                out_len = int(gen_lens[i])
                gen_tokens.append(gen[i, ctx_len:out_len].tolist())

            # build full sequences for forward passes
            student_full = [student_ctx[i] + gen_tokens[i] for i in range(len(prompts))]
            teacher_full = [teacher_ctx[i] + gen_tokens[i] for i in range(len(prompts))]

            student_full_tensor = pad_to_max(student_full, pad_id=tok.pad_id()).to(device)
            teacher_full_tensor = pad_to_max(teacher_full, pad_id=tok.pad_id()).to(device)

            # We compute logits for all positions; then select the positions that predict the generated tokens.
            # For student: positions [ctx_len-1 : ctx_len+L-1]
            # For teacher: positions [teacher_ctx_len-1 : teacher_ctx_len+L-1]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                student_logits_all = model(student_full_tensor)  # [B, S, V]
                with torch.no_grad():
                    teacher_logits_all = ema_teacher(teacher_full_tensor)  # [B, S2, V]

            # collect per-sample aligned logits slices (ragged -> pad)
            student_slices = []
            teacher_slices = []
            max_L = 0
            for i in range(len(prompts)):
                L = len(gen_tokens[i])
                max_L = max(max_L, L)
            V = student_logits_all.size(-1)

            for i in range(len(prompts)):
                L = len(gen_tokens[i])
                # if generation ended early, skip empty
                if L == 0:
                    # create dummy one token to avoid empty tensors
                    L = 1
                    gen_tokens[i] = [tok.eos_id()]

                s_ctx_len = len(student_ctx[i])
                t_ctx_len = len(teacher_ctx[i])

                s_start = s_ctx_len - 1
                s_end = s_start + L
                t_start = t_ctx_len - 1
                t_end = t_start + L

                s_logits = student_logits_all[i, s_start:s_end, :]  # [L,V]
                t_logits = teacher_logits_all[i, t_start:t_end, :]  # [L,V]

                # pad to max_L
                if s_logits.size(0) < max_L:
                    pad = torch.zeros(max_L - s_logits.size(0), V, device=device, dtype=s_logits.dtype)
                    s_logits = torch.cat([s_logits, pad], dim=0)
                    t_logits = torch.cat([t_logits, pad], dim=0)

                student_slices.append(s_logits)
                teacher_slices.append(t_logits)

            student_logits = torch.stack(student_slices, dim=0)  # [B,max_L,V]
            teacher_logits = torch.stack(teacher_slices, dim=0)

            kl = compute_reverse_kl(student_logits, teacher_logits)  # [B,max_L]
            loss = cfg.kl_coef * kl.mean() / cfg.grad_accum_steps

            loss.backward()
            micro_step += 1

            if micro_step % cfg.grad_accum_steps == 0:
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # update EMA teacher
                update_ema(ema_teacher, model, decay=cfg.teacher_ema_decay)

                opt_step += 1

                if opt_step % cfg.log_every == 0:
                    dt = time.time() - t0
                    print(
                        f"[sdft step {opt_step}] loss={loss.item()*cfg.grad_accum_steps:.4f} "
                        f"micro_step={micro_step} dt={dt:.2f}s"
                    )
                    t0 = time.time()

                if opt_step % cfg.save_every == 0:
                    os.makedirs(out_dir, exist_ok=True)
                    ckpt_path = os.path.join(out_dir, f"sdft_step_{opt_step:07d}.pt")
                    torch.save(
                        {
                            "step": opt_step,
                            "micro_step": micro_step,
                            "model": model.state_dict(),
                            "ema_teacher": ema_teacher.state_dict(),
                        },
                        ckpt_path,
                    )
                    print(f"[sdft step {opt_step}] saved {ckpt_path}")

    # Always write a final checkpoint + a stable 'latest' pointer, so short runs
    # (e.g., smoke tests with total_steps < save_every) still produce artifacts.
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"sdft_step_{opt_step:07d}.pt")
    payload = {
        "step": opt_step,
        "micro_step": micro_step,
        "model": model.state_dict(),
        "ema_teacher": ema_teacher.state_dict(),
    }
    torch.save(payload, final_path)
    latest_path = os.path.join(out_dir, "sdft_latest.pt")
    torch.save(payload, latest_path)
    print(f"[sdft] final checkpoint: {final_path}")
    print(f"[sdft] latest checkpoint: {latest_path}")

```



## `src/llm_mhc_sdft_tttd/training/ttt_discover.py`

```python
from __future__ import annotations

import os
import math
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, TTTDiscoverConfig
from ..model.transformer import MHCTransformerLM
from ..data.tokenizer import SpmTokenizer


# -----------------------------
# Environment interface
# -----------------------------

class DiscoveryEnv:
    """
    Abstract environment for TTT-Discover.

    - State s: a candidate solution (string; could be code)
    - Action a: what the model outputs; typically includes reasoning + code
    - Transition: parse(action) -> next_state
    - Reward: continuous, verifiable (returns 0 if invalid)
    """
    def __init__(self, problem_description: str):
        self.d = problem_description

    def initial_state(self) -> str:
        return "<empty>"

    def context_from_archive(self, state: str, archive: "Archive") -> str:
        """
        Convert archive information into natural language context c_i
        (state-action reuse). Default: empty.
        """
        return ""

    def transition(self, action: str) -> str:
        """
        Parse action into new state. Default: action itself.
        Override for code parsing.
        """
        return action

    def reward(self, state: str) -> float:
        """
        Compute R(state). Must be continuous and return 0 if invalid.
        """
        raise NotImplementedError


class ToyStringMatchEnv(DiscoveryEnv):
    """
    Simple toy environment: model must guess a hidden string.
    Reward = fraction of matching characters. Purely for testing the TTT loop.
    """
    def __init__(self, problem_description: str, target: str):
        super().__init__(problem_description)
        self.target = target

    def reward(self, state: str) -> float:
        if not state:
            return 0.0
        # take first len(target) characters
        s = state[: len(self.target)]
        # pad
        s = s.ljust(len(self.target))
        match = sum(1 for a, b in zip(s, self.target) if a == b)
        return match / len(self.target)


# -----------------------------
# Archive + PUCT reuse
# -----------------------------

@dataclass
class ArchiveState:
    state_id: int
    state: str
    reward: float
    parent_id: Optional[int] = None
    n_visits: int = 0
    m_best_child_reward: float = float("-inf")
    is_seed: bool = False


class Archive:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.states: Dict[int, ArchiveState] = {}
        self.next_id = 0
        self.total_expansions = 0  # T in paper

    def add_state(self, state: str, reward: float, parent_id: Optional[int] = None, is_seed: bool = False) -> int:
        sid = self.next_id
        self.next_id += 1
        self.states[sid] = ArchiveState(state_id=sid, state=state, reward=reward, parent_id=parent_id, is_seed=is_seed)
        self._enforce_size()
        return sid

    def _enforce_size(self) -> None:
        if len(self.states) <= self.max_size:
            return
        # keep seeds always, and keep top by reward
        items = list(self.states.values())
        seeds = [s for s in items if s.is_seed]
        non = [s for s in items if not s.is_seed]
        non.sort(key=lambda x: x.reward, reverse=True)
        keep = seeds + non[: max(0, self.max_size - len(seeds))]
        keep_ids = set(s.state_id for s in keep)
        self.states = {sid: self.states[sid] for sid in keep_ids}

    def best(self) -> ArchiveState:
        return max(self.states.values(), key=lambda x: x.reward)

    def __len__(self) -> int:
        return len(self.states)

    def as_list_sorted(self) -> List[ArchiveState]:
        return sorted(self.states.values(), key=lambda x: x.reward, reverse=True)

    def update_after_expand(self, parent_id: int, child_rewards: List[float]) -> None:
        """
        After expanding a parent, update:
          m(parent) <- max(m(parent), max(child_rewards))
          n(a) <- n(a)+1 for parent and its ancestors
          T <- T + 1
        See Appendix A.2.
        """
        if parent_id not in self.states:
            return
        parent = self.states[parent_id]
        y = max(child_rewards) if child_rewards else parent.reward
        parent.m_best_child_reward = max(parent.m_best_child_reward, y)

        # increment visits for parent + ancestors
        cur = parent
        while True:
            cur.n_visits += 1
            if cur.parent_id is None or cur.parent_id not in self.states:
                break
            cur = self.states[cur.parent_id]
        self.total_expansions += 1


def puct_select_start_state(archive: Archive, c: float = 1.0) -> int:
    """
    PUCT-inspired prioritization over archived states (Appendix A.2).
    score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T/(1+n(s)))

    - Q(s): m(s) if visited else R(s)
    - P(s): linear rank prior
    - scale: Rmax - Rmin
    """
    states = archive.as_list_sorted()
    if not states:
        raise ValueError("Archive is empty")

    rewards = [s.reward for s in states]
    Rmax, Rmin = max(rewards), min(rewards)
    scale = max(1e-6, Rmax - Rmin)
    T = archive.total_expansions

    # rank prior
    # rank 0 = best reward
    denom = sum((len(states) - rank) for rank in range(len(states)))
    priors = []
    for rank, s in enumerate(states):
        priors.append((len(states) - rank) / denom)

    best_sid = states[0].state_id
    best_score = float("-inf")
    for rank, s in enumerate(states):
        P = priors[rank]
        n = s.n_visits
        Q = s.m_best_child_reward if n > 0 else s.reward
        score = Q + c * scale * P * math.sqrt(1.0 + (T / (1.0 + n)))
        if score > best_score:
            best_score = score
            best_sid = s.state_id
    return best_sid


# -----------------------------
# Entropic objective + adaptive beta
# -----------------------------

def _kl_q_u(q: torch.Tensor) -> torch.Tensor:
    """
    KL(q || u) where u is uniform over N entries.
      KL(q||u) = sum_n q_n log(N q_n)
    """
    N = q.numel()
    return (q * (math.log(N) + torch.log(q + 1e-12))).sum()


def solve_beta_by_kl(rewards: torch.Tensor, gamma: float, max_beta: float = 1000.0, iters: int = 40) -> float:
    """
    Solve for beta >= 0 such that KL(q_beta || u) = gamma, where
      q_beta(n) = exp(beta r_n) / sum_m exp(beta r_m)
    using bisection.

    rewards: [N] tensor
    """
    # shift for stability (doesn't change q)
    r = rewards - rewards.max()
    lo, hi = 0.0, max_beta
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q = torch.softmax(mid * r, dim=0)
        kl = _kl_q_u(q).item()
        if kl < gamma:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def entropic_advantages_loo(rewards: torch.Tensor, beta: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Leave-one-out (LOO) entropic advantages from Appendix A.1.

    Given N rewards r_n, compute:
      w_n = exp(beta (r_n - r_max))
      Z_-n = (sum_m w_m - w_n) / (N-1)
      A_n = w_n / (Z_-n + eps) - 1
    """
    N = rewards.numel()
    if N < 2:
        return torch.zeros_like(rewards)
    rmax = rewards.max()
    w = torch.exp(beta * (rewards - rmax))
    sum_w = w.sum()
    Z_minus = (sum_w - w) / (N - 1)
    A = (w / (Z_minus + eps)) - 1.0
    return A


# -----------------------------
# Logprob utilities
# -----------------------------

def pad_to_max(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def sequence_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    ctx_lens: List[int],
    seq_lens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Compute log-prob of the continuation tokens (after ctx_len) for each sequence in batch.

    input_ids: [B, S] token ids
    ctx_lens: list of context lengths for each sample (int)
    seq_lens: (optional) list of full sequence lengths for each sample; if provided,
      logprob is computed only up to seq_lens[i]. This is required when sequences are
      padded to a common width but have different effective lengths.

    Returns:
      logp: [B] sum logprob over continuation tokens
    """
    device = input_ids.device
    logits = model(input_ids)  # [B,S,V]
    log_probs = F.log_softmax(logits, dim=-1)

    B, S, V = logits.shape
    out = torch.zeros(B, device=device)
    for i in range(B):
        ctx = int(ctx_lens[i])
        end = int(seq_lens[i]) if seq_lens is not None else S
        end = min(end, S)
        # continuation tokens start at ctx (token index), predicted by positions ctx-1 .. S-2
        # but simplest: iterate over t in [ctx, end-1] and take log_probs at pos=t-1 for token t
        if ctx <= 0:
            raise ValueError("ctx_lens must be >= 1 (prompt should include BOS)")
        if end <= ctx:
            out[i] = 0.0
            continue
        tokens = input_ids[i, ctx:end]
        prev_positions = torch.arange(ctx - 1, end - 1, device=device)
        out[i] = log_probs[i, prev_positions, tokens].sum()
    return out


# -----------------------------
# TTT-Discover main loop
# -----------------------------

def make_prompt(problem_description: str, state: str, context: str) -> str:
    """
    Default prompt format for action generation.
    Override per domain if needed.
    """
    return (
        "### Problem\n"
        f"{problem_description}\n\n"
        "### Current best solution (state)\n"
        f"{state}\n\n"
        "### Context (past attempts / hints)\n"
        f"{context}\n\n"
        "### Task\n"
        "Propose an improved solution. Output ONLY the solution (state) as plain text.\n"
    )


def run_ttt_discover(
    base_ckpt: str,
    tokenizer_path: str,
    env: DiscoveryEnv,
    cfg: TTTDiscoverConfig,
    model_cfg: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """
    Run TTT-Discover for a single problem/environment.

    Returns a dict containing:
      - best_state
      - best_reward
      - archive_dump (json-serializable)
    """
    device = cfg.device
    tok = SpmTokenizer(tokenizer_path)

    # load model config
    if model_cfg is None:
        ckpt_dir = os.path.dirname(base_ckpt)
        cfg_path = os.path.join(ckpt_dir, "model_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                model_cfg = ModelConfig.from_json(f.read())
        else:
            raise ValueError("model_cfg not provided and model_config.json not found")

    base_model = MHCTransformerLM(model_cfg).to(device)
    payload = torch.load(base_ckpt, map_location="cpu")
    if "model" in payload:
        base_model.load_state_dict(payload["model"], strict=True)
    else:
        base_model.load_state_dict(payload, strict=True)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    # LoRA-adapted model (trainable)
    model = MHCTransformerLM(model_cfg).to(device)
    if "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    else:
        model.load_state_dict(payload, strict=True)

    # Apply minimal LoRA to selected linear layers.
    from ..model.lora import apply_lora, mark_only_lora_trainable, save_lora

    replaced = apply_lora(
        model,
        target_module_suffixes=cfg.lora_target_modules,
        r=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
    )
    mark_only_lora_trainable(model)
    print(f"[LoRA] instrumented {len(replaced)} Linear layers")
    # dtype
    if cfg.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    # init archive
    archive = Archive(max_size=min(cfg.buffer_max_size, 1000))
    s0 = env.initial_state()
    r0 = env.reward(s0)
    s0_id = archive.add_state(s0, r0, parent_id=None, is_seed=True)

    best = archive.best()
    print(f"[init] reward={best.reward:.6f}")

    os.makedirs(cfg.out_dir, exist_ok=True)

    for step in range(cfg.ttt_steps):
        # select start state
        if cfg.reuse_enabled and len(archive) > 1:
            start_id = puct_select_start_state(archive, c=cfg.puct_c)
        else:
            start_id = s0_id
        start_state = archive.states[start_id].state
        context = env.context_from_archive(start_state, archive)

        # generate rollouts
        prompts = []
        ctx_lens = []
        for _ in range(cfg.rollouts_per_step):
            ptxt = make_prompt(env.d, start_state, context)
            ids = tok.encode(ptxt, add_bos=True, add_eos=False)
            prompts.append(ids)
            ctx_lens.append(len(ids))

        prompt_tensor = pad_to_max(prompts, pad_id=tok.pad_id()).to(device)

        # sample actions
        with torch.no_grad():
            # IMPORTANT: prompts may be padded; pass `ctx_lens` so generation starts right
            # after the true prompt length for each sample.
            gen, gen_lens = model.generate(
                prompt_tensor,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=tok.eos_id(),
                prompt_lens=ctx_lens,
                pad_token_id=tok.pad_id(),
                return_lens=True,
            )
        # decode and evaluate rewards
        actions = []
        next_states = []
        rewards = []
        for i in range(gen.shape[0]):
            full = gen[i].tolist()
            ctx_len = int(ctx_lens[i])
            out_len = int(gen_lens[i])
            action_ids = full[ctx_len:out_len]
            action_text = tok.decode(action_ids)
            actions.append(action_text)
            s_next = env.transition(action_text)
            r = env.reward(s_next)
            next_states.append(s_next)
            rewards.append(r)

        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

        # adaptive beta based on KL(q||u) = gamma
        if cfg.adaptive_beta:
            beta = solve_beta_by_kl(rewards_t.detach().cpu(), gamma=cfg.target_kl_gamma)
        else:
            beta = cfg.beta

        A_ent = entropic_advantages_loo(rewards_t, beta=beta)  # [N]

        # compute logprobs under current model and base model for KL shaping
        # IMPORTANT: use the token-ids returned by `generate` directly.
        # Do NOT decode -> re-encode, because SentencePiece round-trips are not guaranteed
        # to be perfectly token-identical.
        full_tensor = gen  # [N, S] (prompt + sampled continuation)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logp_theta = sequence_logprob(model, full_tensor, ctx_lens, seq_lens=gen_lens)  # [N]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                logp_base = sequence_logprob(base_model, full_tensor, ctx_lens, seq_lens=gen_lens)

        A = A_ent - cfg.kl_penalty_lambda * (logp_theta.detach() - logp_base.detach())

        # policy gradient loss (REINFORCE): minimize -A * logp
        loss = -(A.detach() * logp_theta).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # archive update: keep top-2 children from this batch
        idx_sorted = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
        topk = idx_sorted[:2]
        child_rewards = [rewards[i] for i in topk]
        for i in topk:
            archive.add_state(next_states[i], rewards[i], parent_id=start_id, is_seed=False)

        archive.update_after_expand(start_id, child_rewards)

        best = archive.best()
        print(f"[step {step+1}/{cfg.ttt_steps}] beta={beta:.3f} loss={loss.item():.4f} best_reward={best.reward:.6f}")

        # save checkpoint of adapter
        if (step + 1) % 10 == 0 or (step + 1) == cfg.ttt_steps:
            adapter_path = os.path.join(cfg.out_dir, f"adapter_step_{step+1:03d}")
            save_lora(model, os.path.join(adapter_path, 'lora.pt'))

    best = archive.best()
    result = {
        "best_state": best.state,
        "best_reward": best.reward,
        "archive": [asdict(s) for s in archive.as_list_sorted()],
    }
    with open(os.path.join(cfg.out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

```



## `tests/test_model_shapes.py`

```python
import torch
from llm_mhc_sdft_tttd.config import ModelConfig, MHCConfig
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM


def test_forward_shapes():
    cfg = ModelConfig(
        vocab_size=1000,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_head=32,
        d_ff=256,
        max_seq_len=64,
        mhc=MHCConfig(n_streams=4),
    )
    model = MHCTransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size)

```



## `tests/test_sinkhorn.py`

```python
import torch
from llm_mhc_sdft_tttd.model.mhc import sinkhorn_knopp


def test_sinkhorn_doubly_stochastic():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 4)
    P = sinkhorn_knopp(x, tmax=50)
    # rows and cols sum ~1
    row = P.sum(dim=-1)
    col = P.sum(dim=-2)
    assert torch.allclose(row, torch.ones_like(row), atol=1e-2, rtol=1e-2)
    assert torch.allclose(col, torch.ones_like(col), atol=1e-2, rtol=1e-2)
    # positivity
    assert (P >= 0).all()

```



## `tests/conftest.py`

```python
"""Test bootstrap.

The repo is intended to be installed via `pip install -e .[dev]`.
However, to keep tests runnable in minimal environments (or when editable
installs are unavailable), we also add `<repo_root>/src` to PYTHONPATH.

This makes `pytest -q` work even if the package is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

```



## `tests/test_sdft_dataloader.py`

```python
import json

from torch.utils.data import DataLoader


def test_sdft_dataloader_identity_collate(tmp_path):
    """Regression test: SDFTJsonlDataset yields dataclass objects.

    PyTorch's default collate_fn cannot collate arbitrary dataclass instances.
    We therefore must use `identity_collate` in the SDFT training loop.
    """
    from llm_mhc_sdft_tttd.data.sdft_dataset import SDFTJsonlDataset, identity_collate

    p = tmp_path / "demo.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "hi", "demonstration": "hello"}) + "\n")
        f.write(json.dumps({"prompt": "bye", "demonstration": "goodbye"}) + "\n")

    ds = SDFTJsonlDataset(str(p))
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=identity_collate)

    batch = next(iter(dl))
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert hasattr(batch[0], "prompt")
    assert hasattr(batch[0], "demonstration")

```



## `tests/test_generate_padding.py`

```python
import torch


def test_generate_respects_prompt_lens_and_avoids_mid_pads():
    """Regression test for a subtle but critical bug:

    If prompts in a batch have different lengths and are right-padded, generation must
    start immediately after each sample's *true* prompt length, not after the padded
    max length.

    The failure mode (old code): for shorter prompts, the last token in the padded
    tensor is PAD, so the model generates after PAD and PAD tokens remain in the middle
    of the produced sequence.
    """

    from llm_mhc_sdft_tttd.config import ModelConfig, MHCConfig
    from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM

    torch.manual_seed(0)

    cfg = ModelConfig(
        vocab_size=50,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_ff=64,
        max_seq_len=64,
        mhc=MHCConfig(n_streams=2),
    )
    m = MHCTransformerLM(cfg)

    pad_id = 0
    # two prompts, different lengths
    p0 = torch.tensor([2, 5, 6, 7, 8], dtype=torch.long)  # len 5
    p1 = torch.tensor([2, 9, 10, 11, 12, 13, 14, 15], dtype=torch.long)  # len 8

    T = 8
    x = torch.full((2, T), pad_id, dtype=torch.long)
    x[0, : p0.numel()] = p0
    x[1, : p1.numel()] = p1

    # generate enough tokens to fill the original padding region for sample 0
    out, lens = m.generate(
        x,
        max_new_tokens=3,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=None,
        prompt_lens=[int(p0.numel()), int(p1.numel())],
        pad_token_id=pad_id,
        return_lens=True,
    )

    # prompts preserved
    assert torch.equal(out[0, : p0.numel()], p0)
    assert torch.equal(out[1, : p1.numel()], p1)

    # the padding region between len(p0) and len(p1) must be filled by generated tokens,
    # not left as PAD (i.e., no PAD tokens in the middle of sample 0's sequence).
    assert (out[0, p0.numel() : p1.numel()] != pad_id).all()

    assert lens[0] == int(p0.numel()) + 3
    assert lens[1] == int(p1.numel()) + 3

```
