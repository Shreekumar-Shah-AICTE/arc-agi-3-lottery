"""Does the sealed V5 calculation fall short inside the scheduler simulator, and is the end-of-run rule the reason?
(exploratory; written 5 Nov 2026 ~16:30 UTC, before any V11 output was read)
For each simulated world (sched_endgame_sim.py: H2 levels, the slot rule D', 10 slots), 4 runs at 121 minutes stand in for
the 4 control runs, once with and once without the end-of-run rule. The sealed calculation (v5_budget_prediction.py: fit,
simulate; imported unchanged; its states() rebuilt from the simulated end states with the same formula) is fitted on the
4 runs' level attempts (coarse grid) and continues their end states to 242 minutes. The worlds are those of
sched_endgame_sim.py (the same seed, 10 runs drawn per world; the first 4 are used here). Truth: the same 4 runs (same random
numbers) played at 242 minutes. Also: the sealed calculation from the 242-minute runs' own state at minute 121 (the
V5-like state; the paper's 42.9 further levels, against 36 observed).
Usage: python3 sched_sealedcalc_check.py [worlds] [drain tokens, default 58000] [R1 scale, default 1] (sensitivity runs)."""
import math, sys
import numpy as np
from v5_budget_prediction import fit, simulate
import sched_endgame_sim as S
from sched_endgame_sim import worlds, NL, GID

def sealed_states(lvl, t, spent, fin):
    """the tuples of v5_budget_prediction.states(): (game, done, levels, tokens on the open level, used * ratio, finished)"""
    used = [spent[i] + t[i] for i in range(len(NL))]
    ua = sum(used); uu = sum(u for u, f in zip(used, fin) if not f); ratio = ua / max(1, uu)
    return [(GID[i] + str(i), lvl[i], NL[i], 0.0 if fin[i] else t[i], used[i] * ratio, fin[i]) for i in range(len(NL))]

def attempts(need, lvl, t, fin):
    A = []
    for i in range(len(NL)):
        A += [(k, need[i][k], True) for k in range(lvl[i])]
        if not fin[i]: A.append((lvl[i], t[i], False))
    return A

if __name__ == '__main__':
    NW = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    NR, R = 4, 200
    D = float(sys.argv[2]) if len(sys.argv) > 2 else 58e3
    if len(sys.argv) > 3: S.R1 *= float(sys.argv[3])
    run = lambda need, b, eg, info=None: S.run(need, b, eg, D=D, info=info)
    rng = np.random.default_rng(20261105)
    by_w = {}
    for w, need in worlds(NW, 10):
        if len(by_w.setdefault(w, [])) < NR: by_w[w].append(need)
    keys = ('1x on', '1x off', '2x on', '2x off', '2x on at 121', 'sealed from 1x on', 'sealed from 1x off',
            'sealed from 2x state at 121')
    acc = {k: [] for k in keys}; fits = {'on': [], 'off': []}
    for w, needs in by_w.items():
        ends = {'on': [], 'off': []}; marks = []; A = {'on': [], 'off': []}; r = {k: [] for k in keys}
        for need in needs:
            for k, eg in (('on', True), ('off', False)):
                info = {}; tot = run(need, 121, eg, info=info)[0]; r['1x ' + k].append(tot)
                ends[k].append(info['end']); A[k] += attempts(need, *[info['end'][j] for j in (0, 1, 3)])
            info = {}; tot = run(need, 242, True, info=info)[0]; r['2x on'].append(tot)
            marks.append(info['mark']); r['2x on at 121'].append(sum(info['mark'][0]))
            r['2x off'].append(run(need, 242, False)[0])
        for k in ('on', 'off'):
            par = fit(A[k], 'lognormal', coarse=True)[1]; fits[k].append(par)
            r['sealed from 1x ' + k] = [simulate(sealed_states(*e), par, 'lognormal', rng, R)[0].mean() for e in ends[k]]
            if k == 'on':
                r['sealed from 2x state at 121'] = [simulate(sealed_states(*m), par, 'lognormal', rng, R)[0].mean() for m in marks]
        for k in keys: acc[k].append(np.mean(r[k]))
    n = len(acc['1x on'])
    print(f'Sealed calculation inside the scheduler simulator: {n} worlds x {NR} runs, {R} draws per state'
          + (f'; drain {D / 1e3:.0f}k tokens, R1 {S.R1:.0f} tokens/min per slot' if len(sys.argv) > 2 else ''))
    for k in ('on', 'off'):
        f = np.array(fits[k]); print(f'  fit on the 1x {k:3s} runs: q {f[:, 0].mean():.3f}, median {math.exp(f[:, 1].mean()) / 1e3:.1f}k, '
                                     f's {f[:, 2].mean():.2f}')
    for k in keys:
        print(f'  {k:28s} {np.mean(acc[k]):6.1f} levels')
    def pd(a, b):
        d = np.array(acc[a]) - np.array(acc[b]); return d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    for lab, a, b in (('true gain, rule on (2x on - 1x on)', '2x on', '1x on'),
                      ('sealed gain from 1x on states', 'sealed from 1x on', '1x on'),
                      ('true gain, rule off (2x off - 1x off)', '2x off', '1x off'),
                      ('sealed gain from 1x off states', 'sealed from 1x off', '1x off'),
                      ('shortfall from 1x on states (sealed - true 2x on)', 'sealed from 1x on', '2x on'),
                      ('shortfall from 1x off states (sealed - true 2x off)', 'sealed from 1x off', '2x off'),
                      ('from the 2x state at 121: sealed further levels - true further levels', 'sealed from 2x state at 121', '2x on')):
        m_, se = pd(a, b); print(f'  {lab}: {m_:+.1f} (SE {se:.1f})')
    d = (np.array(acc['sealed from 1x on']) - np.array(acc['2x on'])) - (np.array(acc['sealed from 1x off']) - np.array(acc['2x off']))
    print(f'  shortfall from 1x on states minus shortfall from 1x off states: {d.mean():+.1f} (SE {d.std(ddof=1) / math.sqrt(len(d)):.1f})')
    print(f'  further levels after minute 121 in the 2x runs: true {np.mean(acc["2x on"]) - np.mean(acc["2x on at 121"]):.1f}, '
          f'sealed calculation from that state {np.mean(acc["sealed from 2x state at 121"]) - np.mean(acc["2x on at 121"]):.1f}')
