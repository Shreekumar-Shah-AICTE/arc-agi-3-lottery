"""Lever 3b (4 Nov 2026, before the V5 readout): of the level attempts still open at 121 minutes, how many are
walls and how many are slow cracks? Crack model (cure-lognormal) fitted to the four FB25 control runs; for an attempt
that has survived c tokens, P(wall | c) = q / S(c). V5 (twice the time) tests the split."""
import math
import numpy as np
from v5_budget_prediction import FB, load, attempts, fit, surv
runs = {r: load(r) for r in FB}
A = [a for r in FB for g in runs[r] for a in attempts(g)[0]]
ll, par = fit(A, 'lognormal'); q = par[0]
print('fit q %.2f median %.1fk s %.2f' % (q, math.exp(par[1]) / 1e3, par[2]))
open_att = [a for a in A if not a[2]]
pw = np.array([q / float(surv(max(a[1], 200), par, 'lognormal')) for a in open_att])
print('open attempts at 121 min: %d (per run %.1f); expected walls %.1f (%.0f%%), slow cracks %.1f' % (
    len(open_att), len(open_att) / len(FB), pw.sum(), 100 * pw.mean(), len(pw) - pw.sum()))
tok = np.array([a[1] for a in open_att])
for lo, hi in ((0, 30e3), (30e3, 60e3), (60e3, 100e3), (100e3, 150e3), (150e3, 1e9)):
    m = (tok >= lo) & (tok < hi)
    if m.any(): print(f'  open with {lo/1e3:.0f}-{min(hi, 999e3)/1e3:.0f}k tokens on the level: n {m.sum():2d}, mean P(wall) {pw[m].mean():.2f}')
# level-index split: first level vs later levels
first = [a for a in open_att if a[0] == 0]; later = [a for a in open_att if a[0] > 0]
for name, S in (('level 1', first), ('later levels', later)):
    p = np.array([q / float(surv(max(a[1], 200), par, 'lognormal')) for a in S])
    print(f'  {name}: open {len(S)}, expected walls {p.sum():.1f}')
