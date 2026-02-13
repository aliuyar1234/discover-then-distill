from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch

from ..data.tokenizer import SpmTokenizer
from ..model.transformer import MHCTransformerLM


@dataclass(frozen=True)
class ProbeCase:
    prompt: str
    # Optional checks. If empty, only `min_chars` is enforced.
    contains_any: tuple[str, ...] = ()
    regex_any: tuple[str, ...] = ()
    min_chars: int = 1


def default_probes() -> List[ProbeCase]:
    """Small, tolerant probe set for regression checks."""
    return [
        ProbeCase(
            prompt="Repeat exactly: BLUEPRINT_OK",
            contains_any=("BLUEPRINT_OK",),
            min_chars=1,
        ),
        ProbeCase(
            prompt="What is 2 + 2? Answer with just a number.",
            contains_any=("4",),
            min_chars=1,
        ),
        ProbeCase(
            prompt="Write a one-word greeting.",
            min_chars=1,
        ),
    ]


def _matches_probe(text: str, probe: ProbeCase) -> bool:
    if len(text.strip()) < probe.min_chars:
        return False

    contains_ok = True
    if probe.contains_any:
        contains_ok = any(s in text for s in probe.contains_any)

    regex_ok = True
    if probe.regex_any:
        regex_ok = any(re.search(pattern, text) is not None for pattern in probe.regex_any)

    return contains_ok and regex_ok


@torch.no_grad()
def run_probes(
    model: MHCTransformerLM,
    tokenizer: SpmTokenizer,
    probes: Sequence[ProbeCase],
    device: str = "cuda",
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    """Run generation-based capability probes and return score/details."""
    model.eval()
    passed = 0
    details: List[Dict[str, Any]] = []

    for probe in probes:
        prompt_ids = tokenizer.encode(probe.prompt, add_bos=True, add_eos=False)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        out, lens = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_id(),
            prompt_lens=[len(prompt_ids)],
            pad_token_id=tokenizer.pad_id(),
            return_lens=True,
        )
        out_ids = out[0, len(prompt_ids) : lens[0]].tolist()
        output_text = tokenizer.decode(out_ids)
        ok = _matches_probe(output_text, probe)
        passed += int(ok)
        details.append(
            {
                "prompt": probe.prompt,
                "output": output_text,
                "passed": ok,
            }
        )

    n = max(1, len(probes))
    return {
        "score": passed / n,
        "passed": passed,
        "total": len(probes),
        "details": details,
    }


def probes_from_prompt_list(prompts: Iterable[str]) -> List[ProbeCase]:
    """Utility for custom probe prompts when no strict expected output is provided."""
    return [ProbeCase(prompt=p, min_chars=1) for p in prompts]
