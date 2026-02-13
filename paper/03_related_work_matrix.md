# Related Work Matrix

Track comparable papers and explicitly position this work.

| Paper | Core Idea | Similarity To Ours | Key Difference | What We Must Cite |
|---|---|---|---|---|
| mHC paper | residual-stream manifold constraints | architecture axis | we integrate continual + test-time adaptation | method equations and setup |
| SDFT paper | self-distillation continual learning | continual axis | we integrate with mHC and TTT pipeline | reverse-KL objective and protocol |
| TTT-Discover paper | entropic test-time RL with reuse | test-time axis | we add LoRA-only safety and consolidation path | objective, beta solver, reuse rule |
| TBD | TBD | TBD | TBD | TBD |

## Positioning Draft

- Prior work treats these components mostly in isolation.
- This work focuses on integration and stability policy:
- `pretrain -> TTT (LoRA-only) -> controlled SDFT consolidation`.
