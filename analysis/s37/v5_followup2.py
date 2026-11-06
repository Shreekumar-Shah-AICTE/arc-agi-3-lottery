#!/usr/bin/env python3
"""V5 follow-up 2 (written 4 Nov 2026 ~15:20 UTC, AFTER the V5 readout; exploratory, not part of the sealed test).
v5_followup.py found that the six games stuck at 1x beat their sealed per-game prediction by 7.2 levels, which we first
read as "walls that are really slow cracks". This script checks that reading three ways.
(1) Timing: when did those games earn their levels? The slot-priority formula (D') adds a bonus for games with levels
    left and fades it to zero over the last 40% of the per-game budget (D_PRIME["fade"] 0.4, an endgame that favours games
    close to finishing). It starts at minute 72.6 at 121 min and at minute 145.2 at 242 min (V5 log: "tail fade phase
    after 145.6 minutes"; FB25-family logs: after 72.8-74.8 minutes). Nothing else differs between kaggle/b12_fb50 and
    kaggle/b12_fb25, so both runs play by the same rules only until minute 72.6.
(2) Supply: given the tokens each game actually generated after minute 121, how many levels does the pooled cure model
    used for the seal (q 0.06, median 22.1k tokens, s 1.05; walls stay walls) expect? The open attempt at minute 121 has
    survived t0 tokens, so it is a wall with posterior probability q / (q + (1 - q) S(t0)); later levels are fresh draws.
    Tokens are booked when an action completes, so t0 and the split at minute 121 are approximate.
(3) Long waits: observed vs expected clears by token band, where expected = the model's cumulative hazard summed over
    each attempt's exposure in the band (controls give O/E near 1 by construction of the fit)."""
import gzip, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
FB = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']
Q, MU, S = 0.06, math.log(22.1e3), 1.05
T1, T_FADE = 121 * 60, 0.6 * 121 * 60
rng = np.random.default_rng(20261104)
def load(r): return json.load(gzip.open(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz', 'rt'))['game_runs']
def gid(g): return g['game_id'].split('-')[0]
def clear_times(g):
    h = g.get('history') or []; cum = np.cumsum([0] + list(g.get('actions_per_level') or []))
    return [h[int(cum[i + 1]) - 1].get('wallclock_seconds') or 0.0 for i in range(g.get('levels_completed') or 0)
            if 0 <= cum[i + 1] - 1 < len(h)]
def attempts(g):
    h = g.get('history') or []; tok = [x.get('generated_tokens') or 0 for x in h]
    cum = np.cumsum([0] + list(g.get('actions_per_level') or [])); lc = g.get('levels_completed') or 0
    out = [(sum(tok[int(cum[i]):int(cum[i + 1])]), 1) for i in range(lc)]
    if lc < g['number_of_levels']: out.append((sum(tok[int(cum[lc]):]), 0))
    return out
def surv(t): return 0.5 * math.erfc((math.log(max(t, 1.0)) - MU) / (S * math.sqrt(2)))
def S_model(t): return Q + (1 - Q) * surv(t)
def H(t): return -math.log(S_model(t))
def ln_inv_surv(p):  # t with surv(t) = p (bisection on log t)
    lo, hi = -5.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if surv(math.exp(mid)) > p: lo = mid
        else: hi = mid
    return math.exp((lo + hi) / 2)
def sim_gain(t0, budget, left, n=4000):
    pw0 = Q / (Q + (1 - Q) * surv(t0)); s0 = surv(t0); out = np.zeros(n)
    for k in range(n):
        b, gain, first = budget, 0, True
        while gain < left:
            if first:
                if rng.random() < pw0: break
                need = max(ln_inv_surv(rng.random() * s0) - t0, 0.0); first = False
            else:
                if rng.random() < Q: break
                need = math.exp(MU + S * rng.standard_normal())
            if need > b: break
            b -= need; gain += 1
        out[k] = gain
    return out
C = [{gid(g): g for g in load(r)} for r in FB]
V = {gid(g): g for g in load('v5_fb50')}
one = {k: np.mean([c[k]['levels_completed'] for c in C]) for k in V}
stuck = sorted(k for k in V if one[k] <= 1.5); rest = sorted(set(V) - set(stuck))
lv = lambda g, t: sum(1 for x in clear_times(g) if x <= t)
print('(1) timing')
print(f'  endgame: the bonus for games with levels left fades over the last 40% of the per-game budget, from minute {0.6 * 121:.1f} at 121 min and from minute {0.6 * 242:.1f} at 242 min')
for name, ks in ((f'{len(stuck)} games stuck at 1x ({", ".join(stuck)})', stuck), (f'other {len(rest)} games', rest)):
    ctl_end = [sum(c[k]['levels_completed'] for k in ks) for c in C]
    print(f'  {name}: controls at their end (121 min) {ctl_end} (mean {np.mean(ctl_end):.2f}); V5 by minute {T_FADE/60:.1f} '
          f'{sum(lv(V[k], T_FADE) for k in ks)}, by minute 121 {sum(lv(V[k], T1) for k in ks)}, at the end {sum(V[k]["levels_completed"] for k in ks)}')
print(f'  all games: controls by minute {T_FADE/60:.1f} {[sum(lv(c[k], T_FADE) for k in c) for c in C]}; V5 by minute {T_FADE/60:.1f} '
      f'{sum(lv(V[k], T_FADE) for k in V)}')
print('(2) supply after minute 121 (pooled model, walls permanent)')
pred, obs = {}, {}
for k, g in V.items():
    h = g.get('history') or []; ct = clear_times(g); L1 = lv(g, T1); nl = g['number_of_levels']
    wc = [x.get('wallclock_seconds') or 0.0 for x in h]; tok = [x.get('generated_tokens') or 0 for x in h]
    last = max([i for i, x in enumerate(wc) if ct and L1 and x <= ct[L1 - 1]] or [-1])
    t0 = sum(t for j, (t, w) in enumerate(zip(tok, wc)) if j > last and w <= T1)
    after = sum(t for t, w in zip(tok, wc) if w > T1)
    pred[k] = 0.0 if L1 >= nl else sim_gain(t0, after, nl - L1).mean(); obs[k] = g['levels_completed'] - L1
for name, ks in (('all 25 games', list(V)), ('stuck games', stuck), ('other games', rest)):
    print(f'  {name}: gain after minute 121 predicted {sum(pred[k] for k in ks):.1f} given the tokens generated, observed {sum(obs[k] for k in ks)}')
print('(3) observed vs expected clears by token band (pooled model hazard over each attempt\'s exposure)')
bands = [(0, 50e3), (50e3, 100e3), (100e3, 150e3), (150e3, 1e12)]
for name, at in (('controls', sum((attempts(g) for c in C for g in c.values()), [])), ('V5', sum((attempts(g) for g in V.values()), []))):
    parts = []
    for lo, hi in bands:
        O = sum(1 for t, e in at if e and lo <= t < hi); E = sum(H(min(t, hi)) - H(lo) for t, _ in at if t > lo)
        lab = f'{lo/1e3:.0f}k+' if hi > 1e11 else f'{lo/1e3:.0f}-{hi/1e3:.0f}k'
        parts.append(f'{lab} O {O} E {E:.1f}')
    print(f'  {name}: attempts {len(at)}; ' + '; '.join(parts) + f'; all O {sum(e for _, e in at)} E {sum(H(t) for t, _ in at):.1f}')
O, E = 6, None
at = sum((attempts(g) for g in V.values()), [])
O = sum(1 for t, e in at if e and t >= 150e3); E = sum(H(t) - H(150e3) for t, _ in at if t > 150e3)
p = 1 - sum(math.exp(-E) * E ** i / math.factorial(i) for i in range(O))
print(f'  V5 attempts past 150k tokens: {sum(1 for t, _ in at if t > 150e3)}; clears {O} vs expected {E:.1f}; Poisson P(>= {O}) = {p:.3f}')
print('(4) state when the second half starts: the sealed method continued the control runs\' end states (after their endgame)')
def open_state(g, t_cut):
    h = g.get('history') or []; ct = clear_times(g); nl = g['number_of_levels']
    wc = [x.get('wallclock_seconds') or 0.0 for x in h]; tok = [x.get('generated_tokens') or 0 for x in h]
    L = sum(1 for x in ct if x <= t_cut)
    if L >= nl: return L, None, 0.0, 0
    last_t = ct[L - 1] if L else -1.0
    return L, sum(t for t, w in zip(tok, wc) if last_t < w <= t_cut), sum(t for t, w in zip(tok, wc) if w <= t_cut), nl - L
def pw(t0): return Q / (Q + (1 - Q) * surv(t0))
for name, G, t in [(r, list(c.values()), T1) for r, c in zip(FB, C)] + [('V5 at minute 121', list(V.values()), T1)]:
    st = [open_state(g, t) for g in G]; op = [x for x in st if x[1] is not None]
    done_fade = sum(1 for g in G if open_state(g, T_FADE)[1] is None)
    gain = sum(sim_gain(x[1], 1.35 * x[2], x[3], 1500).mean() for x in op)
    print(f'  {name}: games complete at minute {T_FADE/60:.1f} {done_fade}, at minute 121 {len(st) - len(op)}; open attempts '
          f'{len(op)}, mean age {np.mean([x[1] for x in op])/1e3:.1f}k tokens, expected walls {sum(pw(x[1]) for x in op):.1f}; '
          f'sealed method from this state (second-half tokens 1.35x first-half): +{gain:.1f} levels')
print('(5) the 1x endgame window (minute 72.6 to 121): levels and tokens of the stuck and other games')
def tok_between(g, a, b): return sum((x.get('generated_tokens') or 0) for x in (g.get('history') or []) if a < (x.get('wallclock_seconds') or 0.0) <= b)
for name, ks in (('stuck', stuck), ('other', rest)):
    ctl = [(sum(lv(c[k], T_FADE) for k in ks), sum(lv(c[k], T1) for k in ks), sum(tok_between(c[k], T_FADE, T1) for k in ks)) for c in C]
    v = (sum(lv(V[k], T_FADE) for k in ks), sum(lv(V[k], T1) for k in ks), sum(tok_between(V[k], T_FADE, T1) for k in ks))
    print(f'  {name} games: controls levels at {T_FADE/60:.1f} -> 121 ' + ', '.join(f'{a}->{b}' for a, b, _ in ctl) +
          f'; tokens in the window {", ".join(f"{t/1e6:.2f}M" for *_, t in ctl)} | V5 {v[0]}->{v[1]}, tokens {v[2]/1e6:.2f}M')
