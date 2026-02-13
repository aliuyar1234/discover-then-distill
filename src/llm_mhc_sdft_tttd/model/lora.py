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
        # Keep LoRA A/B tensors on the same device/dtype as the wrapped base layer.
        wrapped = wrapped.to(device=base.weight.device, dtype=base.weight.dtype)
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


def extract_lora_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA A/B matrices from a full model state_dict.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.endswith(".A.weight") or k.endswith(".B.weight"):
            out[k] = v.detach().cpu()
    return out


def checkpoint_to_lora_state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert a checkpoint payload into a LoRA-only state_dict.
    Supports:
      - direct LoRA dict (keys ending with .A.weight/.B.weight)
      - full checkpoint with payload["model"] state_dict
      - plain full model state_dict
    """
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")

    if any(k.endswith(".A.weight") or k.endswith(".B.weight") for k in payload.keys()):
        return {k: v.detach().cpu() for k, v in payload.items() if k.endswith(".A.weight") or k.endswith(".B.weight")}

    if "model" in payload and isinstance(payload["model"], dict):
        out = extract_lora_from_state_dict(payload["model"])
    else:
        out = extract_lora_from_state_dict(payload)  # type: ignore[arg-type]

    if not out:
        raise ValueError("No LoRA A/B weights found in checkpoint payload")
    return out


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
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(lora_state_dict(model), path)


def load_lora(model: nn.Module, path: str) -> None:
    sd = torch.load(path, map_location="cpu")
    load_lora_state_dict(model, sd)


def merge_lora_linears(model: nn.Module, unload: bool = True) -> List[str]:
    """
    Merge LoRA weights into base Linear weights.

    For each LoRALinear:
      W <- W + scale * (B @ A)

    If unload=True, replace LoRALinear with its merged nn.Linear.
    Returns list of merged module names.
    """
    merged: List[str] = []
    names = [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    for name in names:
        parent, attr = _get_parent_module(model, name)
        module = getattr(parent, attr)
        if not isinstance(module, LoRALinear):
            continue

        delta = torch.matmul(module.B.weight, module.A.weight) * module.scale
        delta = delta.to(dtype=module.base.weight.dtype, device=module.base.weight.device)
        module.base.weight.data.add_(delta)

        if unload:
            setattr(parent, attr, module.base)
        merged.append(name)
    return merged
