# Figures

Results (2026-07-07/09; all runs in wandb project `delta-attention`, entity singhh5050-stanford-university):

- `fig_wp3_staleness_cliff.png` — decode accuracy vs gamma_dec at 32K: cliff at first reuse (78 -> 29); no useful stride.
- `fig_wp1_frontier.png` — accuracy vs effective sparsity at 32K; adaptive-0.95 on the dense ceiling (fixed gamma16/32 controls pending).
- `fig_wp2_trajectories.png` — CE training pilot, 3 arms x 2000 steps: triple null on drift.
- `fig_wp3_decay.png` — cached-delta vs applied-output fidelity by steps-since-anchor (gamma_dec=16, 32K).
- `drift_curves_32k{,_raw}.png` — position-resolved inter-anchor cosine across PG19 books (cliff at sink+window=3072, flat plateau ~0.6).
- `drift_by_layer_group.png` — layer-group plateaus 0.70/0.57/0.62 (per-layer-gamma opportunity).
- `drift_ruler_vs_pg19.png` — RULER haystacks (0.755) vs prose (0.60), 32K and 65K.

Raw data: wandb artifacts `drift_probe_curves:v0-v2`, per-run `decode_drift_points` tables, `wp2_adapter_{delta,dense,detach}`, `box_final_state` archives.
