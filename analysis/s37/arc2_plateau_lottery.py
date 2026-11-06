"""ARC-AGI-2 plateau lottery (exploratory, CPU; the SD comes from our own draws, nothing is fitted to the board).

Our 7 scored draws of one public TTT notebook lineage (mikelou1 'lb33-89 perfpatch' forks: 29.72, 29.31, 28.06, 30.14,
29.72, 30.14, 30.56; mean 29.66, SD 0.81) define a same-code score distribution. If the big ARC-AGI-2 plateau is mostly
teams resubmitting near-identical public notebooks, a team's public score is about the best of its n draws, so the
plateau mean should rise with SubmissionCount like E[max of n] of N(29.66, 0.81^2). Compared on the final public board.
The byte-identical column (SD 0.34) uses our four PP resubmissions only. Correction (5 Nov 2026): the three A0 draws are
unchanged resubmissions too (A0 v1 three times), so the run-luck floor is the pooled within-notebook SD of both sets
(0.61; V9 uses 0.6), printed last; the 0.81 column also includes the 1.11-point gap between the two notebooks."""
import csv, gzip
import numpy as np
from common import HERE
p = HERE / 'data' / 'lb' / 'arc-agi-2_public_final_20261104T1116.csv.gz'
rows = list(csv.DictReader(gzip.open(p, 'rt', encoding='utf-8-sig')))
S = np.array([float(r['Score']) for r in rows]); N = np.array([int(r['SubmissionCount']) for r in rows])
draws = np.array([29.72, 29.31, 28.06, 30.14, 29.72, 30.14, 30.56]); mu, sd = draws.mean(), draws.std(ddof=1)
print(f'final public ARC-AGI-2 board: {len(S)} teams; our same-lineage draws mean {mu:.2f}, SD {sd:.2f} (n = 7)')
m = (S >= 27) & (S < 35)
print(f'plateau [27, 35): {m.sum()} teams ({m.mean():.0%} of the board), mean {S[m].mean():.2f}, SD {S[m].std():.2f}')
rng = np.random.default_rng(0)
def emax(n, s=sd, k=4000): return (mu + s * rng.standard_normal((k, n))).max(1).mean()
rng2 = np.random.default_rng(1)   # separate stream for the byte-identical column, so the older columns keep their values
SD_SAME = 0.34                     # our four byte-identical ARC-AGI-2 resubmissions (Section 4.8)
def emax2(n, s, k=4000): return (mu + s * rng2.standard_normal((k, n))).max(1).mean()
print('entries   teams  plateau mean (SD)  model E[max of n], SD 0.81   (SD 1.0)  (SD 0.34)')
obs_bins = {}
for lo, hi in [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 6)]:
    sel = m & (N >= lo) & (N <= hi)
    nm = int(np.median(N[sel]))
    print(f'{lo:>3}-{(str(hi) if hi < 10 ** 6 else "inf"):>4}  {sel.sum():5d}   {S[sel].mean():5.2f} ({S[sel].std():4.2f})        '
          f'{emax(min(nm, 500)):5.2f} (n = {nm})            {emax(min(nm, 500), 1.0):5.2f}    {emax2(min(nm, 500), SD_SAME):5.2f}')
    obs_bins[(lo, hi)] = (S[sel].mean(), nm)
(o1, n1), (oT, nT) = obs_bins[(1, 1)], obs_bins[(64, 10 ** 6)]; rise = oT - o1
r81 = emax2(nT, sd, 20000) - emax2(n1, sd, 20000); r34 = emax2(nT, SD_SAME, 20000) - emax2(n1, SD_SAME, 20000)
print(f'rise from 1 to {nT} entries: observed {rise:.2f} points; best of n at SD {sd:.2f} gives {r81:.2f} ({r81 / rise:.0%}), '
      f'at the byte-identical SD {SD_SAME} {r34:.2f} ({r34 / rise:.0%})')
# 5 Nov: both groups are unchanged resubmissions (A0 v1 three times, PP v1 four times), so the run-luck floor is their
# pooled within-notebook SD, not PP's alone (V9 uses the same pooled SD, 0.6). Separate stream: older lines keep their values.
a0, pp = draws[:3], draws[3:]
SD_POOL = float(np.sqrt((((a0 - a0.mean()) ** 2).sum() + ((pp - pp.mean()) ** 2).sum()) / (len(draws) - 2)))
rng3 = np.random.default_rng(2)
def emax3(n, s, k=20000): return (mu + s * rng3.standard_normal((k, n))).max(1).mean()
r60 = emax3(nT, SD_POOL) - emax3(n1, SD_POOL); r86 = emax3(nT, a0.std(ddof=1)) - emax3(n1, a0.std(ddof=1))
rk = lambda x: np.argsort(np.argsort(x))
print(f'Spearman(score, entries) inside the plateau: {np.corrcoef(rk(S[m]), rk(N[m]))[0, 1]:.2f}')
# share of plateau variance the lottery model accounts for (model mean per team vs observed)
pred = np.array([emax(min(int(n), 300), k=600) for n in N[m]])
r2 = 1 - np.mean((S[m] - pred) ** 2) / S[m].var()
print(f'per-team prediction from entries alone: R^2 {r2:.2f}, residual SD {np.sqrt(np.mean((S[m] - pred) ** 2)):.2f}')
print(f'unchanged resubmissions: A0 (3 draws) mean {a0.mean():.2f} SD {a0.std(ddof=1):.2f}; PP (4 draws) mean {pp.mean():.2f} '
      f'SD {pp.std(ddof=1):.2f}; pooled within-notebook SD {SD_POOL:.2f}; notebook means differ by {pp.mean() - a0.mean():.2f}')
print(f'run-luck floor, rise from 1 to {nT} entries at the pooled SD {SD_POOL:.2f}: {r60:.2f} ({r60 / rise:.0%}); '
      f"at A0's SD {a0.std(ddof=1):.2f}: {r86:.2f} ({r86 / rise:.0%})")
