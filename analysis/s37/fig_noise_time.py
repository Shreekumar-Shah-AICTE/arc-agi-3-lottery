"""Fig 5: where run-to-run noise comes from, six same-agent FB25 runs (lever3_noise.py logic).
Left: per-game variance with a fixed token budget per game divided by per-game variance at a fixed wall clock, by minute
(game-bootstrap 90% band). Right: run-to-run SD of the total, observed vs if games were independent (sum of per-game
variances), by minute."""
import gzip, json, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
RUNS = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3', 'b32_spec2_r2', 'b32_spec2_r3']
B1 = 121 * 60
def curve(g):
    h = g['history']; apl = g['actions_per_level']
    T = int(re.search(r'tokens=(\d+)', g.get('solver_note') or 'tokens=0').group(1))
    ct = np.cumsum([int(x.get('generated_tokens') or 0) for x in h]) if h else np.zeros(0)
    wc = np.array([float(x['wallclock_seconds']) for x in h]) if h else np.zeros(0)
    lt, lw, i = [], [], 0
    for k in range(g['levels_completed']): i += apl[k]; lt.append(ct[i - 1]); lw.append(wc[i - 1])
    return dict(T=max(T, int(ct[-1]) if len(ct) else 0), ct=ct, wc=wc, lt=np.array(lt), lw=np.array(lw),
                fin=g['levels_completed'] >= g['number_of_levels'])
D = {r: {g['game_id'].split('-')[0]: curve(g) for g in json.load(gzip.open(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz', 'rt'))['game_runs']} for r in RUNS}
games = sorted(set.intersection(*[set(v) for v in D.values()]))
def tok_by(c, t):
    if t >= B1: return c['T']
    k = np.searchsorted(c['wc'], t, side='right'); return int(c['ct'][k - 1]) if k else 0
rng = np.random.default_rng(5)
F = np.linspace(0.1, 1.0, 10); rat, rlo, rhi, sd_obs, sd_ind, cens = [], [], [], [], [], []
for f in F:
    t = f * B1
    W = np.array([[(D[r][g]['lw'] <= t).sum() for g in games] for r in RUNS], float)
    tau = {g: np.mean([tok_by(D[r][g], t) for r in RUNS]) for g in games}
    K = np.array([[(D[r][g]['lt'] <= tau[g]).sum() for g in games] for r in RUNS], float)
    cens.append(sum(1 for r in RUNS for g in games if not D[r][g]['fin'] and tau[g] > D[r][g]['T']))
    vw, vk = W.var(0, ddof=1), K.var(0, ddof=1); rat.append(vk.sum() / vw.sum())
    bs = [vk[p].sum() / vw[p].sum() for p in (rng.integers(0, len(games), len(games)) for _ in range(2000))]
    rlo.append(np.quantile(bs, .05)); rhi.append(np.quantile(bs, .95))
    sd_obs.append(W.sum(1).std(ddof=1)); sd_ind.append(np.sqrt(vw.sum()))
mins = F * 121
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=160)
a1.fill_between(mins, rlo, rhi, color='tab:blue', alpha=0.2); a1.plot(mins, rat, 'o-', color='tab:blue', ms=3.5)
a1.axhline(1, color='black', lw=0.8, ls=':')
for m, c, rr in zip(mins, cens, rat):
    if c > 10: a1.annotate(f'{c} cut', (m, rr), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=6.5, color='grey')
a1.set_xlabel('minutes into the run'); a1.set_ylabel('per-game variance: fixed tokens / fixed clock')
a1.set_title('Fixing tokens per game removes early noise, not late noise', fontsize=9)
a1.set_ylim(0, 1.6); a1.grid(alpha=0.25)
a2.plot(mins, sd_obs, 'o-', color='black', ms=3.5, label='observed SD of the run total')
a2.plot(mins, sd_ind, 's--', color='tab:orange', ms=3.5, label='SD if games were independent')
a2.set_xlabel('minutes into the run'); a2.set_ylabel('levels'); a2.legend(fontsize=7.5, frameon=False)
a2.set_title('Shared server steadies the total mid-run (p = 0.021 at 60 min; 4 times tested)', fontsize=9); a2.grid(alpha=0.25)
fig.suptitle('Six same-agent runs, 25 public games x 121 min, 10 streams', fontsize=9.5)
fig.tight_layout(); fig.savefig('/data/arc/paper/figs/s37_noise_over_time.png')
for m, r_, o, i, c in zip(mins, rat, sd_obs, sd_ind, cens): print(f'{m:5.1f} min ratio {r_:.2f} SD obs {o:.1f} indep {i:.1f} censored {c}')
