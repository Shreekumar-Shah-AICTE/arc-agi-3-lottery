"""Lever 3d (4 Nov 2026): which version of the crack model reproduces the observed run-to-run noise?
Same simulator as lever5_scheduler_sim.py (25 games, FB25 level counts, 10 streams, 4.5M tokens, FAIR sharing).
Models: H0 homogeneous (every level median 22.1k, s 1.05, walls q 0.06 drawn per run);
H1 level-specific medians fixed per 'world' (between-level SD 0.80, run luck s 0.49, from lever3c), walls drawn per run;
H2 as H1 but walls fixed per world. For each model: 20 worlds x 30 runs; mean total and the run-to-run SD within a world.
Observed (six same-agent FB25 runs): mean 102.5, SD 5.3."""
import math
import numpy as np
from v5_budget_prediction import FB, load
NL = [g['number_of_levels'] for g in load(FB[0])]
STEP, STREAMS, TOTAL = 2000.0, 10, 4.5e6
STEPS = int(TOTAL / STREAMS / STEP)
def sim(med, wall, s, rng):
    """med[i][k] log-median for game i level k; wall[i][k] True if wall (None -> draw per run with q 0.06)"""
    G = len(NL); lvl = np.zeros(G, int); c = np.zeros(G); fin = np.zeros(G, bool)
    def need(i):
        k = lvl[i]; w = (rng.random() < 0.06) if wall is None else wall[i][k]
        return math.inf if w else math.exp(med[i][k] + s * rng.standard_normal())
    nd = np.array([need(i) for i in range(G)])
    for _ in range(STEPS):
        live = np.where(~fin)[0]
        if len(live) == 0: break
        sh = min(1.0, STREAMS / len(live))
        for i in live:
            c[i] += STEP * sh
            if c[i] >= nd[i]:
                lvl[i] += 1; c[i] = 0.0
                if lvl[i] >= NL[i]: fin[i] = True
                else: nd[i] = need(i)
    return lvl.sum()
if __name__ == '__main__':
    rng = np.random.default_rng(3)
    MU = math.log(22.1e3)
    for name in ('H0', 'H1', 'H2'):
        means, sds = [], []
        for w in range(20):
            if name == 'H0': med = [[MU] * n for n in NL]; wall = None; s = 1.05
            else:
                # centre so that the pooled median stays ~22k: mean of log-medians = MU
                med = [[MU + 0.80 * rng.standard_normal() for _ in range(n)] for n in NL]; s = 0.49
                wall = None if name == 'H1' else [[rng.random() < 0.06 for _ in range(n)] for n in NL]
            tot = np.array([sim(med, wall, s, rng) for _ in range(30)])
            means.append(tot.mean()); sds.append(tot.std(ddof=1))
        print(f'{name}: mean total {np.mean(means):6.1f} (between worlds SD {np.std(means):4.1f}); run-to-run SD within a world: '
              f'mean {np.mean(sds):4.1f}, 10-90% {np.quantile(sds, .1):4.1f}-{np.quantile(sds, .9):4.1f}')
    print('observed: mean 102.5, run-to-run SD 5.3 (six runs; 90% CI for an SD from 6 runs ~ x0.67 to x2.1)')
