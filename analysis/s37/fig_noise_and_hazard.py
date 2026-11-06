"""Figure 1: rank spread of 5 identical B12 runs on the final public board (2 Nov close, fetched 4 Nov), and the level-clear hazard (analysis/s7)."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, B12_LB, load_lb
sc = np.array(load_lb('arc-agi-3'))
# Our own team's row (best draw 33.37, official rank 197) is removed before placing each draw; ties go against the draw,
# which reproduces the official rank of the best draw.
_others = np.delete(sc, int(np.where(np.isclose(sc, max(B12_LB)))[0][-1]))
rank = lambda x: 1 + int((_others >= x - 1e-9).sum())
b12 = B12_LB
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.35, 1]})
ax = axes[0]
r = np.arange(1, len(sc) + 1)
ax.plot(r, sc, color='#555', lw=1.6, label=f'ARC-AGI-3 final public leaderboard ({len(sc):,} teams, 2 Nov 2026 close)')
ax.set_xscale('log'); ax.set_xlim(1, 6000); ax.set_ylim(0, 80)
labpos = [None, (2300, 58), (2300, 52), (2300, 46), (2300, 40)]
for i, x in enumerate(b12):
    k = rank(x); ax.scatter([k], [x], s=55, color='#d62728', zorder=5)
    if labpos[i] is None:
        ax.annotate(f'#{k:,} ({x})', (k, x), xytext=(-30, 12), textcoords='offset points', fontsize=8.5, color='#d62728')
    else:
        ax.annotate(f'#{k:,} ({x})', (k, x), xytext=labpos[i], textcoords='data', fontsize=8.5, color='#d62728',
                    arrowprops=dict(arrowstyle='-', color='#d62728', lw=0.6, alpha=0.7))
    print('B12 run', x, 'rank', k)
m = np.mean(b12); s = np.std(b12, ddof=1)
ax.axhspan(m - s, m + s, color='#d62728', alpha=0.10, label=f'Our B12 notebook: mean {m:.1f}, SD {s:.1f} (n = 5 identical runs)')
ax.set_xlabel('Public leaderboard rank (log scale)'); ax.set_ylabel('Public score')
_rk = sorted(rank(x) for x in b12)
ax.set_title(f'One notebook, five identical runs: ranks #{_rk[0]:,} to #{_rk[-1]:,}', fontsize=11)
ax.legend(loc='lower left', fontsize=8.5, frameon=False)
ax.grid(alpha=0.25, which='both')
A = json.load(open(ROOT / 'analysis' / 's7' / 'level_attempts.json')); B = [0, 10e3, 20e3, 40e3, 60e3]
ax = axes[1]
for nm, runs, col in [('dev games (10, tuned on)', {'b1', 'b1c', 'b3v2', 'b4', 'b4v2'}, '#1f77b4'), ('holdout games (15, never tuned on)', {'b5', 'b5c', 'b7', 'b7c'}, '#ff7f0e')]:
    S = [x for x in A if x[0] in runs]; hz = []; mid = []
    for lo, hi in zip(B, B[1:]):
        risk = [x for x in S if x[3] > lo]; ev = [x for x in risk if x[5] and x[3] <= hi]; expo = sum(min(x[3], hi) - lo for x in risk)
        hz.append(len(ev) / expo * 1e4); mid.append((lo + hi) / 2e3)
    print(nm, 'hazard per 10k tokens by bin', [round(h, 3) for h in hz])
    ax.plot(mid, hz, 'o-', color=col, lw=2, label=f'{nm}, {len(S)} level attempts')
ax.set_xlabel('Tokens already spent on the level (thousands)'); ax.set_ylabel('Level clears per 10k tokens (hazard)')
ax.set_title('Level-clear hazard: holdout games start slow, then speed up', fontsize=11)
ax.set_ylim(0, 0.8); ax.grid(alpha=0.25); ax.legend(fontsize=8.5, frameon=False, loc='upper right')
fig.text(0.01, 0.005, 'Sources: Kaggle public leaderboard and our 5 scored runs of arc3-b12-x6dprime v1 (Kaggle CLI; final board fetched 4 Nov 2026); 415 level attempts from 9 commit runs (repo analysis/s7). Hazard bins 0-10k, 10-20k, 20-40k, 40-60k tokens.', fontsize=7.5, color='#444')
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(ROOT / 'paper' / 'figs' / 's37_noise_and_hazard.png', dpi=150)
print('ok')
