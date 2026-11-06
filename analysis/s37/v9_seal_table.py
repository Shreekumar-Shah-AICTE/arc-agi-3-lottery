"""V9 seal table (S37, 4 Nov 2026 ~16:45 UTC, before the ARC-AGI-2 private board is published). One-parameter family:
for each per-half-board noise SD sigma, fit the lottery model with a skill gain per doubling of submissions (HB, beta
free on a grid) to the final public ARC-AGI-2 board, then simulate the private board. Also H (beta = 0) at 0.45 and 0.6.
The sealed text (sealed/v9_arc2_private_lb_prediction.txt) reads its numbers from this script's output."""
import importlib.util, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v9', HERE / 'v9_arc2_sealed_model.py'); v9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v9)
BETAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
KEYS = ('spearman', 'retain', 'grad', 'excess_drop', 'ours_pct')
rows = [('HB', s) for s in (0.3, 0.45, 0.6, 0.81, 1.0, 1.3)] + [('H', 0.45), ('H', 0.6), ('H', 0.81)]
print(f'cluster [{v9.LO:.0f},{v9.HI:.0f}): {len(v9.S)} teams; public gradient {v9.S[v9.G64].mean() - v9.S[v9.G1].mean():.2f} '
      f'({int(v9.G64.sum())} vs {int(v9.G1.sum())}); top-decile cut {np.quantile(v9.S, 0.9):.2f}; ours {v9.S[v9.iours]}')
print('model sigma  loglik   mu    tau   f    beta | ' + ' | '.join(f'{k} mean [5%, 95%]' for k in KEYS))
for lab, s in rows:
    ll, mu, tau, f, beta = v9.fit(s, beta_grid=BETAS if lab == 'HB' else (0.0,))
    pr = v9.predictive(mu, tau, f, s, beta, R=800)
    cells = ' | '.join(f'{np.mean(pr[k]):.3f} [{np.quantile(pr[k], 0.05):.3f}, {np.quantile(pr[k], 0.95):.3f}]' for k in KEYS)
    print(f'{lab:2s}  {s:4.2f}  {ll:8.1f} {mu:5.2f} {tau:5.2f} {f:4.2f} {beta:4.1f} | {cells}', flush=True)
