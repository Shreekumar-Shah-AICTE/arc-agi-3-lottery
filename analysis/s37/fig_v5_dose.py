"""Fig 4: budget dose-response. Levels cleared by minute for the four FB25 runs (121 min per game), the V5 run
(242 min per game) when its output exists, the sealed prediction (127, 90% 110-145) and the stated alternatives."""
import gzip, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
FB = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']
def times(p):
    G = json.load(gzip.open(p, 'rt'))['game_runs']; out = []
    for g in G:
        h = g['history']; apl = g['actions_per_level']; i = 0
        for k in range(g['levels_completed']): i += apl[k]; out.append(float(h[i - 1]['wallclock_seconds']) / 60)
    return np.sort(out)
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
grid = np.linspace(0, 121, 243)
for j, r in enumerate(FB):
    t = times(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz')
    ax.plot(grid, [np.sum(t <= x) for x in grid], color='grey', lw=1, alpha=0.8, label='four runs at 121 min per game' if j == 0 else None)
v5 = HERE / 'data' / 'runs' / 'v5_fb50.benchmark.json.gz'
if v5.exists():
    t = times(v5); g2 = np.linspace(0, 244, 489)
    ax.plot(g2, [np.sum(t <= x) for x in g2], color='black', lw=1.8, label=f'V5 run at 242 min per game ({len(t)} levels)')
ax.errorbar([242], [127], yerr=[[127 - 110], [145 - 127]], fmt='o', color='tab:blue', capsize=4, lw=1.6,
            label='sealed prediction: 127 (90% 110-145)')
ax.scatter([242], [103], marker='v', color='tab:red', zorder=3, label='alternative A, saturation ~103')
ax.scatter([242], [140], marker='D', color='tab:purple', zorder=3, s=18, label='alternative B, log-linear ~140')
ax.plot([242, 242], [150, 175], color='tab:green', lw=4, alpha=0.6, label='alternative C, linear 150-175')
ax.axvline(121, color='grey', lw=0.7, ls=':'); ax.axhline(183, color='grey', lw=0.7, ls='--'); ax.text(126, 185, 'all 183 levels', fontsize=7, color='grey')
ax.set_xlim(0, 250); ax.set_ylim(0, 195); ax.set_xlabel('minutes into the run (per-game budget)'); ax.set_ylabel('levels cleared, 25 public games')
ax.set_title('What twice the play time buys (prediction sealed before the run)', fontsize=10)
ax.legend(fontsize=7.2, frameon=False, loc='upper left'); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig('/data/arc/paper/figs/s37_v5_dose_response.png'); print('saved; V5 present:', v5.exists())
