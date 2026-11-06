"""Lever 3c (4 Nov 2026): is the crack wait a property of the level or of the run?
Six same-agent FB25 runs. For each (game, level) cleared in at least 2 runs: log generated tokens to clear it. One-way
random-effects decomposition: pooled variance of log tokens = between-(game, level) + within (run-to-run). Cleared
attempts only (cut-off attempts excluded, so hard levels are under-represented; read as a lower bound on persistence).
Context: the pooled cure-lognormal spread is s = 1.05; a homogeneous renewal model with that spread predicts a run-total
SD of ~13 levels (lever5_scheduler_sim.py) against an observed 5.3."""
import numpy as np
from v5_budget_prediction import load, attempts
RUNS = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3', 'b32_spec2_r2', 'b32_spec2_r3']
cells = {}
for r in RUNS:
    for g in load(r):
        for k, t, d in attempts(g)[0]:
            if d and t > 0: cells.setdefault((g['game_id'].split('-')[0], k), []).append(np.log(t))
use = {c: np.array(v) for c, v in cells.items() if len(v) >= 2}
allv = np.concatenate(list(use.values()))
within = np.concatenate([v - v.mean() for v in use.values()])
n_c = len(use); n = len(allv)
ms_w = (within ** 2).sum() / (n - n_c)
grand = allv.mean()
ms_b = sum(len(v) * (v.mean() - grand) ** 2 for v in use.values()) / (n_c - 1)
n0 = (n - sum(len(v) ** 2 for v in use.values()) / n) / (n_c - 1)
var_b = max(0.0, (ms_b - ms_w) / n0)
print(f'(game, level) cells cleared in >= 2 runs: {n_c}; observations {n}')
print(f'pooled SD of log tokens {allv.std(ddof=1):.2f}; within-cell (run-to-run) SD {np.sqrt(ms_w):.2f}; between-cell SD {np.sqrt(var_b):.2f}')
print(f'share of variance that is the level itself (ICC): {var_b / (var_b + ms_w):.2f}')
# by level index
for k in range(0, 4):
    vs = [v for (g, kk), v in use.items() if kk == k]
    if vs:
        w = np.concatenate([v - v.mean() for v in vs]); print(f'  level {k + 1}: cells {len(vs)}, within SD {np.sqrt((w ** 2).sum() / (len(w) - len(vs))):.2f}')
