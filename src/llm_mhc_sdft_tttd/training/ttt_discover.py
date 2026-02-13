from __future__ import annotations

import os
import math
import time
import json
import random
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, TTTDiscoverConfig
from ..model.transformer import MHCTransformerLM
from ..data.tokenizer import SpmTokenizer
from ..tracking import RunTracker, GracefulStopper


def grad_global_norm(parameters) -> float:
    sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq += float(torch.sum(g * g).item())
    return math.sqrt(max(0.0, sq))


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[ttt] CUDA not available; falling back to CPU.")
        return "cpu"
    return device


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
    Reward = normalized character-overlap with the target (case-insensitive).
    Purely for testing the TTT loop.
    """
    def __init__(self, problem_description: str, target: str):
        super().__init__(problem_description)
        self.target = target

    def initial_state(self) -> str:
        return ""

    def reward(self, state: str) -> float:
        if not state or not self.target:
            return 0.0
        from collections import Counter

        s = state.lower()
        t = self.target.lower()
        overlap = Counter(s) & Counter(t)
        n_match = sum(overlap.values())
        return n_match / len(t)


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
        # but simplest: iterate over t in [ctx, S-1] and take log_probs at pos=t-1 for token t
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


def archive_to_payload(archive: Archive) -> Dict[str, Any]:
    return {
        "max_size": archive.max_size,
        "next_id": archive.next_id,
        "total_expansions": archive.total_expansions,
        "states": [asdict(s) for s in archive.states.values()],
    }


def archive_from_payload(payload: Dict[str, Any], fallback_max_size: int) -> Archive:
    archive = Archive(max_size=int(payload.get("max_size", fallback_max_size)))
    archive.next_id = int(payload.get("next_id", 0))
    archive.total_expansions = int(payload.get("total_expansions", 0))
    states = payload.get("states", [])
    if isinstance(states, list):
        for row in states:
            if not isinstance(row, dict):
                continue
            s = ArchiveState(
                state_id=int(row.get("state_id", 0)),
                state=str(row.get("state", "")),
                reward=float(row.get("reward", 0.0)),
                parent_id=(None if row.get("parent_id") is None else int(row.get("parent_id"))),
                n_visits=int(row.get("n_visits", 0)),
                m_best_child_reward=float(row.get("m_best_child_reward", float("-inf"))),
                is_seed=bool(row.get("is_seed", False)),
            )
            archive.states[s.state_id] = s
    if archive.states:
        archive.next_id = max(archive.next_id, max(archive.states.keys()) + 1)
    return archive


def run_ttt_discover(
    base_ckpt: str,
    tokenizer_path: str,
    env: DiscoveryEnv,
    cfg: TTTDiscoverConfig,
    model_cfg: Optional[ModelConfig] = None,
    resume: bool = False,
    tracker_command: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run TTT-Discover for a single problem/environment.

    Returns a dict containing:
      - best_state
      - best_reward
      - archive_dump (json-serializable)
    """
    device = resolve_device(cfg.device)
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
    from ..model.lora import apply_lora, mark_only_lora_trainable, save_lora, lora_state_dict, load_lora_state_dict

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
    amp_device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    if cfg.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    if amp_device_type != "cuda" and amp_dtype == torch.float16:
        amp_dtype = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    os.makedirs(cfg.out_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    state_path = os.path.join(cfg.out_dir, "ttt_state_latest.pt")
    result_path = os.path.join(cfg.out_dir, "result.json")

    tracker = RunTracker(
        out_dir=cfg.out_dir,
        run_type="ttt_discover",
        total_steps=cfg.ttt_steps,
        command=tracker_command,
        resume=resume,
    )
    tracker.register_artifact("metrics", metrics_path, required=True)
    tracker.register_artifact("state_checkpoint", state_path, required=False)
    tracker.register_artifact("result", result_path, required=True)
    tracker.register_artifact("final_adapter", os.path.join(cfg.out_dir, f"adapter_step_{cfg.ttt_steps:03d}", "lora.pt"), required=True)

    # init archive
    archive = Archive(max_size=min(cfg.buffer_max_size, 1000))
    s0 = env.initial_state()
    r0 = env.reward(s0)
    s0_id = archive.add_state(s0, r0, parent_id=None, is_seed=True)

    start_step = 0
    if resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        if isinstance(state, dict):
            if "lora" in state and isinstance(state["lora"], dict):
                load_lora_state_dict(model, state["lora"])
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "archive" in state and isinstance(state["archive"], dict):
                archive = archive_from_payload(state["archive"], fallback_max_size=min(cfg.buffer_max_size, 1000))
            s0_id = int(state.get("seed_state_id", s0_id))
            start_step = int(state.get("step_completed", 0))
            torch_state = state.get("torch_rng_state")
            if torch_state is not None:
                torch.random.set_rng_state(torch_state)
            cuda_state = state.get("cuda_rng_state")
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)
            py_state = state.get("python_rng_state")
            if py_state is not None:
                random.setstate(py_state)
            tracker.event("resume_state", start_step=start_step, archive_size=len(archive))
            print(f"[ttt] resumed from {state_path} at step={start_step} archive={len(archive)}")

    best = archive.best()
    print(f"[init] reward={best.reward:.6f}")

    metrics_mode = "a" if (start_step > 0 and os.path.exists(metrics_path)) else "w"
    metrics_f = open(metrics_path, metrics_mode, encoding="utf-8")
    stopper = GracefulStopper()
    stopper.install()

    try:
        for step in range(start_step, cfg.ttt_steps):
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
                    min_new_tokens=1,
                    prompt_lens=ctx_lens,
                    pad_token_id=tok.pad_id(),
                    return_lens=True,
                    forbid_token_ids=[tok.pad_id()],
                )
            # decode and evaluate rewards
            next_states = []
            rewards = []
            for i in range(gen.shape[0]):
                full = gen[i].tolist()
                ctx_len = int(ctx_lens[i])
                out_len = int(gen_lens[i])
                action_ids = full[ctx_len:out_len]
                action_text = tok.decode(action_ids)
                s_next = env.transition(action_text)
                r = env.reward(s_next)
                next_states.append(s_next)
                rewards.append(r)

            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

            # objective advantages
            if cfg.objective == "entropic":
                # adaptive beta based on KL(q||u) = gamma
                if cfg.adaptive_beta:
                    beta = solve_beta_by_kl(rewards_t.detach().cpu(), gamma=cfg.target_kl_gamma)
                else:
                    beta = cfg.beta
                A_obj = entropic_advantages_loo(rewards_t, beta=beta)  # [N]
            elif cfg.objective == "expected":
                beta = cfg.beta
                # REINFORCE baseline: centered rewards.
                A_obj = rewards_t - rewards_t.mean()
            else:
                raise ValueError(f"Unknown TTT objective: {cfg.objective}")

            # compute logprobs under current model and base model for KL shaping
            # IMPORTANT: use the token-ids returned by `generate` directly.
            # Do NOT decode -> re-encode, because SentencePiece round-trips are not guaranteed
            # to be perfectly token-identical.
            full_tensor = gen  # [N, S] (prompt + sampled continuation)

            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                logp_theta = sequence_logprob(model, full_tensor, ctx_lens, seq_lens=gen_lens)  # [N]
            with torch.no_grad():
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    logp_base = sequence_logprob(base_model, full_tensor, ctx_lens, seq_lens=gen_lens)

            kl_term = (logp_theta.detach() - logp_base.detach())
            A = A_obj - cfg.kl_penalty_lambda * kl_term

            # policy gradient loss (REINFORCE): minimize -A * logp
            loss = -(A.detach() * logp_theta).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
            else:
                grad_norm = grad_global_norm(model.parameters())
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

            metrics_row = {
                "step": step + 1,
                "objective": cfg.objective,
                "beta": float(beta),
                "loss": float(loss.item()),
                "best_reward": float(best.reward),
                "mean_reward": float(rewards_t.mean().item()),
                "max_reward": float(rewards_t.max().item()),
                "min_reward": float(rewards_t.min().item()),
                "kl_term_mean": float(kl_term.mean().item()),
                "grad_norm": float(grad_norm),
                "archive_size": len(archive),
            }
            metrics_f.write(json.dumps(metrics_row) + "\n")
            metrics_f.flush()
            tracker.heartbeat(
                step=step + 1,
                total_steps=cfg.ttt_steps,
                metrics={
                    "loss": float(loss.item()),
                    "best_reward": float(best.reward),
                    "beta": float(beta),
                    "archive_size": len(archive),
                },
            )

            # save resumable state every step.
            state_payload = {
                "step_completed": step + 1,
                "seed_state_id": s0_id,
                "lora": lora_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "archive": archive_to_payload(archive),
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python_rng_state": random.getstate(),
            }
            torch.save(state_payload, state_path)

            # save checkpoint of adapter
            if (step + 1) % 10 == 0 or (step + 1) == cfg.ttt_steps:
                adapter_path = os.path.join(cfg.out_dir, f"adapter_step_{step+1:03d}")
                adapter_file = os.path.join(adapter_path, "lora.pt")
                save_lora(model, adapter_file)
                tracker.register_artifact(f"adapter_step_{step+1:03d}", adapter_file, required=False)

            if stopper.stop_requested:
                tracker.finalize(
                    status="paused",
                    step=step + 1,
                    total_steps=cfg.ttt_steps,
                    message="Graceful pause requested by signal.",
                )
                raise KeyboardInterrupt("Graceful pause requested")
    except KeyboardInterrupt:
        if not stopper.stop_requested:
            tracker.finalize(
                status="paused",
                step=None,
                total_steps=cfg.ttt_steps,
                message="Interrupted by user.",
            )
        raise
    except Exception as exc:
        tracker.finalize(
            status="failed",
            step=None,
            total_steps=cfg.ttt_steps,
            message=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        metrics_f.close()
        stopper.uninstall()

    best = archive.best()
    result = {
        "best_state": best.state,
        "best_reward": best.reward,
        "archive": [asdict(s) for s in archive.as_list_sorted()],
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    tracker.finalize(
        status="completed",
        step=cfg.ttt_steps,
        total_steps=cfg.ttt_steps,
        message="TTT discovery completed.",
        metrics={"best_reward": float(best.reward)},
    )
    return result
