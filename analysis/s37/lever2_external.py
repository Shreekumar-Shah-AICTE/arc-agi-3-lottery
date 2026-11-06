"""Lever 2 (4 Nov 2026): does the crack model describe other teams' runs?
Six public save-run logs of other teams (/data/paperdata/external; five Duck-harness forks at ~132 min per game, one
10-game run at 25 min) and our four FB25 control runs. Per run: level attempts in generated tokens (cleared or cut off),
maximum-likelihood fit of the cure-lognormal (wall share q, median tokens to crack, spread s), and the fit check
(expected levels under the fitted model given each game's tokens vs actual)."""
import gzip, json, re, math, glob
import numpy as np
from pathlib import Path
from v5_budget_prediction import fit, surv, dens, tinv
HERE = Path(__file__).resolve().parent
def tot_tokens(g):
    m = re.search(r'tokens=(\d+)', g.get('solver_note') or '')
    s = sum(int(x.get('generated_tokens') or 0) for x in g['history'])
    return max(int(m.group(1)) if m else 0, s)
def attempts(g):
    h = g['history']; apl = g['actions_per_level']; tot = tot_tokens(g); out, i, used = [], 0, 0
    for k in range(g['number_of_levels']):
        if k < g['levels_completed']:
            seg = h[i:i + apl[k]]; i += apl[k]; tk = sum(int(x.get('generated_tokens') or 0) for x in seg)
            out.append((k, tk, True)); used += tk
        else:
            out.append((k, max(0, tot - used), False)); break
    return out
def expected_levels(g, par, rng, R=300, T_fin=None):
    """levels expected under the model if the game had its actual total tokens, played from level 0.
    A game that completed stopped because it was done, not because tokens ran out: it gets T_fin (the median tokens of
    the run's unfinished games) when that is larger."""
    T = tot_tokens(g); nl = g['number_of_levels']; lv = np.zeros(R)
    if g['levels_completed'] >= nl and T_fin: T = max(T, T_fin)
    for r in range(R):
        b, n = T, 0
        while n < nl:
            u = rng.random()
            if u <= par[0]: break
            need = tinv(u, par, 'lognormal')
            if need > b: break
            b -= need; n += 1
        lv[r] = n
    return lv.mean()
rng = np.random.default_rng(11)
sets = [(Path(f).name.split('.benchmark')[0][:44], f) for f in sorted(glob.glob('/data/paperdata/external/*.benchmark.json.gz'))]
sets += [(r, str(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz')) for r in ('b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3')]
print('run | games levels | tokens/game median | attempts cleared/cut | fit q median(k) s | model-expected levels vs actual')
for name, f in sets:
    G = json.load(gzip.open(f, 'rt'))['game_runs']
    A = [a for g in G for a in attempts(g)]
    ll, par = fit(A, 'lognormal')
    unf = [tot_tokens(g) for g in G if g['levels_completed'] < g['number_of_levels']]; T_fin = float(np.median(unf)) if unf else None
    exp = sum(expected_levels(g, par, rng, T_fin=T_fin) for g in G); act = sum(g['levels_completed'] for g in G)
    print(f'{name:44s} | {len(G):2d} {act:3d} | {np.median([tot_tokens(g) for g in G])/1e3:6.0f}k | {sum(a[2] for a in A):3d}/{sum(not a[2] for a in A):2d} | '
          f'q {par[0]:.2f} med {math.exp(par[1])/1e3:5.1f}k s {par[2]:.2f} | {exp:6.1f} vs {act}')
