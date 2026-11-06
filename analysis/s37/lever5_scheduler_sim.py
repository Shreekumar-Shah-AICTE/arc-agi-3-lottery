"""Lever 5 candidate (4 Nov 2026, exploratory, before the V5 readout): what would a crack-model scheduler gain?
Simulated 25-game run in token time under the fitted crack model (q 0.06, median 22k, s 1.05; FB25 level counts), 10
streams, total tokens = an FB25 run (4.5M over 121 min). Schedulers: FAIR (streams shared equally by unfinished games,
idle streams wasted when fewer than 10 remain), HAZARD (each step the 10 unfinished games with the highest posterior crack
hazard on their current level get a stream), ORACLE (knows walls; skips walled games). Also a heterogeneous variant: each
game's median multiplied by exp(N(0, 0.8^2)) (games differ in difficulty) where the hazard scheduler uses the game's
own cleared-level history (empirical-Bayes median)."""
import math
import numpy as np
from v5_budget_prediction import FB, load, ninv, Phi
runs = {r: load(r) for r in FB}
NL = [g['number_of_levels'] for g in runs[FB[0]]]
Q, MU, S = 0.06, math.log(22.1e3), 1.05
STEP, STREAMS, TOTAL = 2000.0, 10, 4.5e6
STEPS = int(TOTAL / STREAMS / STEP)
def hazard(c, mu):
    """posterior crack hazard per STEP tokens for a level survived c tokens, cure-lognormal"""
    z0 = (np.log(np.maximum(c, 1.0)) - mu) / S; z1 = (np.log(c + STEP) - mu) / S
    s0 = Q + (1 - Q) * (1 - Phi(z0)); s1 = Q + (1 - Q) * (1 - Phi(z1))
    return (s0 - s1) / s0
def run(policy, rng, hetero=0.0):
    G = len(NL); gm = MU + hetero * rng.standard_normal(G)          # per-game log-median
    lvl = np.zeros(G, int); c = np.zeros(G); fin = np.zeros(G, bool)
    def draw(i):
        if rng.random() < Q: return math.inf
        return math.exp(gm[i] + S * rng.standard_normal())
    need = np.array([draw(i) for i in range(G)])
    hist = [[] for _ in range(G)]                                       # cleared-level costs per game (for EB median)
    for _ in range(STEPS):
        live = np.where(~fin)[0]
        if len(live) == 0: break
        if policy == 'fair':
            share = min(1.0, STREAMS / len(live)); give = {i: share for i in live}
        else:
            if policy == 'oracle': cand = [i for i in live if math.isfinite(need[i])] or list(live)
            else: cand = list(live)
            if policy == 'hazard' and hetero > 0:
                mus = np.array([(np.log(hist[i]).sum() + MU * 2) / (len(hist[i]) + 2) if hist[i] else MU for i in cand])
            else: mus = np.full(len(cand), MU)
            h = hazard(c[cand], mus) if policy != 'oracle' else -c[cand]
            top = [cand[k] for k in np.argsort(-h)[:STREAMS]]; give = {i: 1.0 for i in top}
        for i, sh in give.items():
            c[i] += STEP * sh
            if c[i] >= need[i]:
                hist[i].append(need[i]); lvl[i] += 1; c[i] = 0.0
                if lvl[i] >= NL[i]: fin[i] = True
                else: need[i] = draw(i)
    return lvl.sum(), fin.sum()
rng = np.random.default_rng(42)
for het in (0.0, 0.8):
    print(f'--- heterogeneity SD {het} (log-median per game)')
    res = {}
    for pol in ('fair', 'hazard', 'oracle'):
        out = np.array([run(pol, rng, het) for _ in range(150)])
        res[pol] = out[:, 0]
        print(f'  {pol:6s}: levels {out[:, 0].mean():6.1f} (SD {out[:, 0].std():4.1f}), games completed {out[:, 1].mean():4.1f}')
    for pol in ('hazard', 'oracle'):
        d = res[pol].mean() / res['fair'].mean() - 1
        print(f'  {pol} vs fair: {100 * d:+.1f}%')
