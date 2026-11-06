"""Fig. 3: levels completed per game across the 6 valid runs of the same agent (the lottery). spec2 A is left out:
it played 24 of 25 games (Appendix A row 31)."""
import statistics as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, CTL, SPEC2_VALID, load_run
runs = CTL + SPEC2_VALID
data = {r: load_run(r) for r in runs}
games = sorted(data[runs[0]])
nl = {g: data[runs[0]][g][2] for g in games}
M = np.array([[data[r][g][1] / nl[g] for r in runs] for g in games])
L = np.array([[data[r][g][1] for r in runs] for g in games])
order = np.argsort(-(M.std(axis=1)))
M = M[order]; L = L[order]; gs = [games[i] for i in order]
fig, ax = plt.subplots(figsize=(7.2, 7.8))
im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{L[i, j]}/{nl[gs[i]]}', ha='center', va='center', fontsize=7, color='white' if M[i, j] < 0.6 else 'black')
labels = ['ctl A', 'ctl B', 'ctl C', 'ctl D', 'spec2 B', 'spec2 C']
ax.set_xticks(range(len(runs))); ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks(range(len(gs))); ax.set_yticklabels(gs, fontsize=7.5)
means = [st.mean([data[r][g][0] for g in games]) for r in runs]
for j, m in enumerate(means): ax.text(j, -0.9, f'{m:.1f}', ha='center', fontsize=7.5)
ax.set_title('Same agent, 6 valid runs, 25 public games: levels completed per game\n(top numbers: run mean score; rows sorted by run-to-run swing)', fontsize=9.5, pad=22)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("share of the game's levels completed", fontsize=8)
fig.text(0.01, 0.005, 'Source: benchmark.json of 6 dev runs (B12 harness; spec2 = speculative depth 2, judged noise). spec2 A left out: 24 of 25 games played.', fontsize=6.8, color='#444')
plt.tight_layout(rect=(0, 0.02, 1, 1))
plt.savefig(ROOT / 'paper' / 'figs' / 's37_pergame_lottery.png', dpi=150)
print('ok')
