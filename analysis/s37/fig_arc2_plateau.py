"""Figure 6: the ARC-AGI-2 plateau is a resubmission lottery (4 Nov 2026; numbers as in out/arc2_plateau_lottery.out).
Left: final public ARC-AGI-2 board (3,627 teams), plateau [27, 35) shaded, our 7 same-lineage draws as ticks.
Right: plateau mean score by number of entries vs E[max of n draws] from our draws' N(29.66, 0.81^2); no free parameters."""
import csv, gzip
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import HERE, ROOT
p = HERE / 'data' / 'lb' / 'arc-agi-2_public_final_20261104T1116.csv.gz'
rows = list(csv.DictReader(gzip.open(p, 'rt', encoding='utf-8-sig')))
S = np.array([float(r['Score']) for r in rows]); N = np.array([int(r['SubmissionCount']) for r in rows])
draws = np.array([29.72, 29.31, 28.06, 30.14, 29.72, 30.14, 30.56]); mu, sd = draws.mean(), draws.std(ddof=1)
m = (S >= 27) & (S < 35)
rng = np.random.default_rng(0)
def emax(n, s, k=20000): return (mu + s * rng.standard_normal((k, n))).max(1).mean()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1, 1.15]})
ax = axes[0]
ax.hist(S, bins=np.arange(0, max(S) + 1, 0.5), color='#888', edgecolor='none')
ax.axvspan(27, 35, color='#1f77b4', alpha=0.12, label=f'plateau 27-35: {m.sum():,} teams ({m.mean():.0%})')
for d in draws: ax.axvline(d, ymin=0, ymax=0.06, color='#d62728', lw=1.5)
ax.plot([], [], color='#d62728', lw=1.5, label=f'our 7 draws of one public notebook lineage (mean {mu:.2f}, SD {sd:.2f})')
ax.set_xlabel('Public score (final ARC-AGI-2 board)'); ax.set_ylabel('Teams per 0.5-point bin')
ax.set_title(f'ARC-AGI-2 final public board: {len(S):,} teams', fontsize=11); ax.legend(fontsize=8.5, loc='upper right')
ax.set_xlim(0, min(60, max(S) + 1))
ax = axes[1]
buckets = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 6)]
xs, ys, es, ns = [], [], [], []
for lo, hi in buckets:
    sel = m & (N >= lo) & (N <= hi)
    xs.append(np.median(N[sel])); ys.append(S[sel].mean()); es.append(S[sel].std() / np.sqrt(sel.sum())); ns.append(sel.sum())
ax.errorbar(xs, ys, yerr=1.96 * np.array(es), fmt='o', color='#1f77b4', capsize=3, label='plateau teams: mean score (95% CI)')
for x, y, n in zip(xs, ys, ns): ax.annotate(f'{n}', (x, y), xytext=(6, -12), textcoords='offset points', fontsize=7.5, color='#1f77b4')
grid = np.unique(np.round(np.logspace(0, np.log10(150), 25)).astype(int))
ax.plot(grid, [emax(n, sd) for n in grid], color='#d62728', lw=1.8, label=f'best of n draws, N({mu:.2f}, {sd:.2f}²): no fitting')
ax.plot(grid, [emax(n, 1.0) for n in grid], color='#d62728', lw=1.2, ls='--', label='same with SD 1.00')
ax.set_xscale('log'); ax.set_xlabel('Entries per team (bucket median; numbers = teams)'); ax.set_ylabel('Public score')
ax.set_title('Score rises with entries as best-of-n of one notebook predicts', fontsize=11); ax.legend(fontsize=8.5, loc='upper left')
ax.grid(alpha=0.3)
fig.text(0.01, 0.005, 'Sources: Kaggle ARC-AGI-2 public leaderboard download (final, fetched 4 Nov 2026 11:16 UTC); our 7 scored '
         'draws of mikelou1 perfpatch forks. Script analysis/s37/fig_arc2_plateau.py.', fontsize=7.5, color='#555')
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(ROOT / 'paper' / 'figs' / 's37_arc2_plateau.png', dpi=150)
print('saved', [round(y, 2) for y in ys], [int(x) for x in xs])
