"""Lever 3 (4 Nov 2026): is run-to-run noise from the clock or from the agent?
Same-agent runs at 121 min per game (25 public games, 10 streams). Speculative decoding (spec2) changes server speed only.
For each run and game we know when (wall clock) and after how many generated tokens each level was cleared.
Wall-clock design (what ran): levels by minute t.  Token design (counterfactual): levels by a fixed per-game token budget,
the same in every run (= the mean tokens that game had used by minute t). The agent's moves depend on tokens, not on the
clock (no time-based yields: LOCAL_ANALYZER_YIELD_SECONDS 0), so levels by a token count are what any run with that
budget would have shown. If the token design has much lower run-to-run variance, the noise is allocation/throughput;
if not, it is the agent's own sampling."""
import gzip, json, re, math
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
RUNS = {'b12_fb25': 'ctl', 'b28_ctl_fb25': 'ctl', 'b32_ctl_r2': 'ctl', 'b32_ctl_r3': 'ctl',
        'b32_spec2_r2': 'spec2', 'b32_spec2_r3': 'spec2'}
B1 = 121 * 60
def load(r): return json.load(gzip.open(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz', 'rt'))['game_runs']
def game_curve(g):
    h = g['history']; apl = g['actions_per_level']
    T = int(re.search(r'tokens=(\d+)', g.get('solver_note') or 'tokens=0').group(1))
    ct = np.cumsum([int(x.get('generated_tokens') or 0) for x in h]) if h else np.zeros(0)
    wc = np.array([float(x['wallclock_seconds']) for x in h]) if h else np.zeros(0)
    lv_tok, lv_t, i = [], [], 0
    for k in range(g['levels_completed']):
        i += apl[k]; lv_tok.append(ct[i - 1]); lv_t.append(wc[i - 1])
    return dict(nl=g['number_of_levels'], done=g['levels_completed'], T=max(T, int(ct[-1]) if len(ct) else 0),
                ct=ct, wc=wc, lv_tok=np.array(lv_tok), lv_t=np.array(lv_t), fin=g['levels_completed'] >= g['number_of_levels'])
def tokens_by(c, t):
    if t >= B1: return c['T']
    k = np.searchsorted(c['wc'], t, side='right'); return int(c['ct'][k - 1]) if k else 0
def levels_by_time(c, t): return int((c['lv_t'] <= t).sum())
def levels_by_tokens(c, tau): return int((c['lv_tok'] <= tau).sum())
data = {r: {g['game_id'].split('-')[0]: game_curve(g) for g in load(r)} for r in RUNS}
games = sorted(set.intersection(*[set(v) for v in data.values()]))
runs = list(RUNS)
print('runs', runs, 'games', len(games))
print('\nrun-level: levels, games won, total tokens (M), tokens of unfinished games (M)')
L = []; TT = []
for r in runs:
    lv = sum(data[r][g]['done'] for g in games); won = sum(data[r][g]['fin'] for g in games)
    tt = sum(data[r][g]['T'] for g in games) / 1e6; tu = sum(data[r][g]['T'] for g in games if not data[r][g]['fin']) / 1e6
    L.append(lv); TT.append(tt); print(f'  {r:14s} {RUNS[r]:5s} levels {lv:3d} won {won:2d} tokens {tt:.2f}M unfinished {tu:.2f}M')
print('  corr(levels, total tokens) over runs: %.2f' % np.corrcoef(L, TT)[0, 1])
# per-game token share variability
cv = []
for g in games:
    t = np.array([data[r][g]['T'] for r in runs], float)
    if not all(data[r][g]['fin'] for r in runs): cv.append(t.std(ddof=1) / max(1, t.mean()))
print('  per-game tokens across runs, CV median %.2f (games not won in every run)' % np.median(cv))
def design(f, rset, how='mean'):
    """returns per-run totals (wall, token), per-game arrays, censored count.
    how='mean': budget = mean tokens used by minute t (cells where a run used fewer are censored = lower bounds);
    how='min': budget = the smallest tokens any run used by minute t (never censored, slightly fewer tokens)."""
    t = f * B1
    W = np.array([[levels_by_time(data[r][g], t) for g in games] for r in rset], float)
    agg = np.mean if how == 'mean' else np.min
    tau = {g: agg([tokens_by(data[r][g], t) for r in rset]) for g in games}
    K = np.zeros_like(W); cens = 0
    for i, r in enumerate(rset):
        for j, g in enumerate(games):
            c = data[r][g]; K[i, j] = levels_by_tokens(c, tau[g])
            if not c['fin'] and tau[g] > c['T']: cens += 1
    return W, K, cens
rng = np.random.default_rng(4)
print('\nwall-clock vs fixed-token design (same mean tokens per game), runs =', len(runs))
print('  budget f | wall: mean  SD  sum per-game var | token: mean  SD  sum per-game var | ratio var(token)/var(wall) [game bootstrap 5-95%] | censored cells')
for how, f in [('mean', x) for x in (0.25, 0.5, 0.75, 0.9, 1.0)] + [('min', x) for x in (0.25, 0.5, 0.75, 0.9, 1.0)]:
    W, K, cens = design(f, runs, how)
    vw, vk = W.var(axis=0, ddof=1), K.var(axis=0, ddof=1)
    sw, sk = W.sum(1), K.sum(1)
    rat = vk.sum() / max(1e-9, vw.sum())
    bs = []
    for b in range(2000):
        p = rng.integers(0, len(games), len(games)); bs.append(vk[p].sum() / max(1e-9, vw[p].sum()))
    lo, hi = np.quantile(bs, [.05, .95])
    print(f'  {how:4s} {f:4.2f} | {sw.mean():6.1f} {sw.std(ddof=1):5.1f} {vw.sum():6.1f} | {sk.mean():6.1f} {sk.std(ddof=1):5.1f} {vk.sum():6.1f} | {rat:.2f} [{lo:.2f}-{hi:.2f}] | {cens}')
# covariance between games inside a run (shared streams): total var vs sum of per-game var
W, K, _ = design(1.0, runs)
print('\nat 121 min: var(total) %.1f vs sum per-game var %.1f (between-game covariance share %.0f%%)' % (
    W.sum(1).var(ddof=1), W.var(axis=0, ddof=1).sum(), 100 * (W.sum(1).var(ddof=1) - W.var(axis=0, ddof=1).sum()) / max(1e-9, W.sum(1).var(ddof=1))))
# agent-only noise: per-game SD at fixed tokens, by game
print('\nper-game levels across runs at 121 min (wall) and at the fixed token budget (token):')
for j, g in enumerate(games):
    print(f'  {g} nl {data[runs[0]][g]["nl"]}: wall {W[:, j].astype(int).tolist()} token {K[:, j].astype(int).tolist()}')
# permutation test for between-game covariance: shuffle each game's run labels independently (breaks the shared-server
# link, keeps every game's own spread); how often is var(total) as small as observed?
print('\nshared-server covariance test (each game\'s runs shuffled independently, 20000 shuffles):')
for f in (0.25, 0.5, 0.75, 1.0):
    W, _, _ = design(f, runs)
    obs = W.sum(1).var(ddof=1); null = []
    for b in range(20000):
        P = np.stack([rng.permutation(W[:, j]) for j in range(W.shape[1])], 1); null.append(P.sum(1).var(ddof=1))
    null = np.array(null)
    print(f'  f {f:4.2f}: var(total) observed {obs:5.1f}, null median {np.median(null):5.1f}, P(null <= observed) = {(null <= obs).mean():.3f}')
