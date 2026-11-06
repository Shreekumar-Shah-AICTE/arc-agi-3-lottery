"""V5 (pre-registered, written 4 Nov 2026 ~10:09 UTC, before the run): what does twice the play time buy?
Run under test: B12-FB25 byte-identical except bm.solver.max_runtime_s_per_game = 242*60 (FB25: 121*60), 25 public games,
25 concurrent games on 10 SGLang streams, n_passes = 1 (kernel shreekumarshah/arc3-v5-fb50).
Model (the crack model): a level attempt is a wait in generated-token time T. With probability q the agent never cracks it
(a wall); otherwise log T ~ N(mu, s^2). One parameter set for all games and levels, fitted by maximum likelihood to the
480 level attempts of the 4 FB25 control runs (411 clears, 69 censored at the budget). The 2x run = each 1x end state
continued: the current level conditional on its survival so far, then fresh draws for the next levels, with the second
half's server tokens shared by the games still playing (stream redistribution, ratio = all tokens / unfinished-game tokens).
Check before use (pseudo-prospective): the same procedure fitted on the runs cut at half time predicts their full-time level
counts. Prints the prediction for total levels at 2x, games completed, and per-game expected levels."""
import gzip, json, math, re
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
FB = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']
B1 = 121 * 60
def load(r): return json.load(gzip.open(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz', 'rt'))['game_runs']
def attempts(g, frac=None):
    """level attempts as (level_index, generated tokens, cleared), optionally cut at wall-clock frac * 121 min"""
    h = g['history']; apl = g['actions_per_level']
    tot = int(re.search(r'tokens=(\d+)', g.get('solver_note') or 'tokens=0').group(1))
    cut = None if frac is None else frac * B1
    out, i, used = [], 0, 0
    for k in range(g['number_of_levels']):
        if k < g['levels_completed']:
            seg = h[i:i + apl[k]]; i += apl[k]
            tk = sum(int(x.get('generated_tokens') or 0) for x in seg)
            if cut is not None and seg and float(seg[-1]['wallclock_seconds']) > cut:
                part = sum(int(x.get('generated_tokens') or 0) for x in seg if float(x['wallclock_seconds']) <= cut)
                out.append((k, part, False)); return out, used + part
            out.append((k, tk, True)); used += tk
        else:
            if cut is not None:
                part = sum(int(x.get('generated_tokens') or 0) for x in h[i:] if float(x['wallclock_seconds']) <= cut)
                out.append((k, part, False)); return out, used + part
            out.append((k, max(0, tot - used), False)); return out, tot
    return out, used
def Phi(x):
    x = np.asarray(x, float); z = np.abs(x) / math.sqrt(2.0); t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)
def ninv(p):   # Acklam's inverse normal CDF
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p = min(max(p, 1e-12), 1 - 1e-12)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
# --- two cure models: lognormal (primary) and log-logistic (robustness)
def surv(t, par, kind):
    q, m, s = par; t = np.maximum(np.asarray(t, float), 1.0)
    if kind == 'lognormal': return q + (1 - q) * (1 - Phi((np.log(t) - m) / s))
    return q + (1 - q) / (1 + np.exp((np.log(t) - m) / s))
def dens(t, par, kind):
    q, m, s = par; t = np.maximum(np.asarray(t, float), 1.0); z = (np.log(t) - m) / s
    if kind == 'lognormal': return (1 - q) * np.exp(-0.5 * z * z) / (s * t * math.sqrt(2 * math.pi))
    e = np.exp(z); return (1 - q) * e / (s * t * (1 + e) ** 2)
def tinv(u, par, kind):
    """T with S(T) = u (u > q)"""
    q, m, s = par; p = (u - q) / (1 - q)
    if kind == 'lognormal': return math.exp(m + s * ninv(1 - p))
    return math.exp(m + s * math.log((1 - p) / p))
def fit(A, kind, coarse=False):
    t = np.array([max(a[1], 200) for a in A], float); d = np.array([a[2] for a in A])
    qs = np.arange(0.0, 0.40, 0.02 if coarse else 0.01); ms = np.arange(math.log(4e3), math.log(1.5e5), 0.06 if coarse else 0.03)
    ss = np.arange(0.3, 2.6, 0.1 if coarse else 0.05) if kind == 'lognormal' else np.arange(0.2, 1.6, 0.06 if coarse else 0.03)
    best = None
    for q in qs:
        for m in ms:
            for s in ss:
                ll = np.where(d, np.log(dens(t, (q, m, s), kind) + 1e-300), np.log(surv(t, (q, m, s), kind) + 1e-300)).sum()
                if best is None or ll > best[0]: best = (float(ll), (float(q), float(m), float(s)))
    return best
def states(run, frac=None):
    sts, ua, uu = [], 0, 0
    for g in run:
        at, used = attempts(g, frac)
        done = sum(1 for a in at if a[2]); c = at[-1][1] if (at and not at[-1][2]) else 0
        fin = done >= g['number_of_levels']; sts.append((g['game_id'].split('-')[0], done, g['number_of_levels'], c, used, fin))
        ua += used; uu += 0 if fin else used
    ratio = ua / max(1, uu)
    return [(gid, done, nl, c, used * ratio, fin) for gid, done, nl, c, used, fin in sts]
def simulate(sts, par, kind, rng, R):
    """returns (R,) total levels and dict game -> mean levels"""
    tot = np.zeros(R); per = {}
    for gid, done, nl, c, delta, fin in sts:
        lv = np.full(R, done, float)
        if not fin:
            sc = float(surv(c, par, kind))
            for rr in range(R):
                budget, first, n = delta, True, done
                while n < nl:
                    u = rng.random() * sc if first else rng.random()
                    if u <= par[0]: break
                    need = tinv(u, par, kind) - (c if first else 0.0)
                    if need > budget: break
                    budget -= need; n += 1; first = False
                lv[rr] = n
        tot += lv; per[gid] = float(lv.mean())
    return tot, per
if __name__ == '__main__':
    rng = np.random.default_rng(20261104)
    runs = {r: load(r) for r in FB}
    actual = [sum(g['levels_completed'] for g in runs[r]) for r in FB]
    print('FB25 actual levels', actual, 'mean %.2f SD %.2f; games completed' % (np.mean(actual), np.std(actual, ddof=1)),
          [sum(g['levels_completed'] == g['number_of_levels'] for g in runs[r]) for r in FB])
    out = {}
    for kind in ('lognormal', 'loglogistic'):
        Ah = [a for r in FB for g in runs[r] for a in attempts(g, 0.5)[0]]; fh = fit(Ah, kind)
        ph = [simulate(states(runs[r], 0.5), fh[1], kind, rng, 400)[0].mean() for r in FB]
        A = [a for r in FB for g in runs[r] for a in attempts(g)[0]]; ff = fit(A, kind)
        sims = [simulate(states(runs[r]), ff[1], kind, rng, 1000) for r in FB]
        tot = np.concatenate([s[0] for s in sims]); means = [s[0].mean() for s in sims]
        per = {gid: np.mean([s[1][gid] for s in sims]) for gid in sims[0][1]}
        nl = {g['game_id'].split('-')[0]: g['number_of_levels'] for g in runs[FB[0]]}
        wins = sum(1 for gid in per if per[gid] >= nl[gid] - 0.5)
        out[kind] = (np.mean(means), tot)
        print(f'--- {kind}: half-fit q {fh[1][0]:.2f} median {math.exp(fh[1][1])/1e3:.0f}k s {fh[1][2]:.2f}; half->full predicted '
              f'{np.round(ph, 1).tolist()} mean {np.mean(ph):.1f} vs actual mean {np.mean(actual):.2f}')
        print(f'    full-fit q {ff[1][0]:.2f} median {math.exp(ff[1][1])/1e3:.0f}k s {ff[1][2]:.2f} (loglik {ff[0]:.1f}); 2x total levels by base run '
              f'{np.round(means, 1).tolist()} -> mean {np.mean(means):.1f}; model-only 5-95% {np.quantile(tot, .05):.0f}-{np.quantile(tot, .95):.0f}')
        print('    per-game expected levels at 2x vs 1x mean:', {gid: (round(per[gid], 1), round(np.mean([next(g['levels_completed'] for g in runs[r] if g['game_id'].startswith(gid)) for r in FB]), 2)) for gid in sorted(per)})
        print(f'    games expected complete at 2x (mean levels within 0.5 of all): {wins}')
    # bootstrap over games (parameter uncertainty), lognormal, coarse grid
    gids = [g['game_id'] for g in runs[FB[0]]]; bs = []
    for b in range(int(__import__('os').environ.get('V5_BOOT', '30'))):
        pick = rng.choice(len(gids), len(gids)); sel = [gids[i] for i in pick]
        A = [a for r in FB for gid in sel for g in runs[r] if g['game_id'] == gid for a in attempts(g)[0]]
        fb_ = fit(A, 'lognormal', coarse=True)[1]
        bs.append(np.mean([simulate(states(runs[r]), fb_, 'lognormal', rng, 150)[0].mean() for r in FB]))
    print('bootstrap (games, coarse grid) 2x mean levels: 5/50/95%', np.quantile(bs, [.05, .5, .95]).round(1), 'n', len(bs))
