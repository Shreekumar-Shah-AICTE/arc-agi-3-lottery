"""Lever 2b (4 Nov 2026, exploratory, CPU): the open 4-pass puzzle (2 Nov, lb_cluster.py). dfranzen's public
milestone-2 run plays 25 games x 4 concurrent passes on one server. Its pass means vary far less than independent games
predict. Lever 3 found the same sign in our own runs at 60 min (shared-server covariance, p = 0.021), fading by 121 min.
Test: the same permutation test (shuffle each game's passes independently) on the 4 passes at several time fractions,
with levels as the unit, plus the per-pass token supply."""
import gzip, json
from collections import defaultdict
import numpy as np
from common import HERE
d = json.load(gzip.open(HERE / 'data' / 'external' / 'dfranzen_milestone2_4pass.benchmark.json.gz', 'rt'))
print('label', d.get('label'), 'n_passes', d.get('n_passes'), 'game_runs', len(d['game_runs']), d.get('start_time'), d.get('end_time'))
runs = defaultdict(dict)            # pass -> game -> (times of level clears, tokens list, final wall)
cnt = defaultdict(int)
for g in d['game_runs']:
    game = g['game_id'].split('-')[0]; p = cnt[game]; cnt[game] += 1
    h = g.get('history') or []; apl = g.get('actions_per_level') or []; L = g.get('levels_completed') or 0
    cum = np.cumsum([0] + list(apl)); wc = [x.get('wallclock_seconds') or 0 for x in h]
    clears = [wc[cum[i + 1] - 1] for i in range(L) if cum[i + 1] - 1 < len(wc)]
    runs[p][game] = (clears, sum(x.get('generated_tokens') or 0 for x in h), max(wc) if wc else 0, g.get('final_score') or 0)
P = sorted(runs); games = sorted(set.intersection(*[set(runs[p]) for p in P]))
Tend = max(v[2] for p in P for v in runs[p].values())
print(f'passes {len(P)}, games in all passes {len(games)}, longest game wall {Tend / 60:.1f} min')
for p in P:
    print(f'  pass {p}: levels {sum(len(runs[p][g][0]) for g in games)}, tokens {sum(runs[p][g][1] for g in games) / 1e6:.2f}M, '
          f'mean score {np.mean([runs[p][g][3] for g in games]):.2f}')
rng = np.random.default_rng(0)
for f in [0.25, 0.5, 0.75, 1.0]:
    t = f * Tend
    M = np.array([[sum(1 for c in runs[p][g][0] if c <= t) for g in games] for p in P], float)   # passes x games
    tot = M.sum(1); obs = tot.var(ddof=1); sumv = M.var(0, ddof=1).sum()
    null = []
    for _ in range(20000):
        S = np.array([rng.permutation(M[:, j]) for j in range(M.shape[1])]).T
        null.append(S.sum(1).var(ddof=1))
    null = np.array(null)
    print(f'f {f:.2f} ({t / 60:5.1f} min): pass totals {tot.astype(int).tolist()}, var(total) {obs:6.1f} vs sum per-game var {sumv:6.1f}; '
          f'null median {np.median(null):6.1f}, P(null <= obs) = {(null <= obs).mean():.3f}')
# score-based version (what lb_cluster.py compared): pass means of final scores
S = np.array([[runs[p][g][3] for g in games] for p in P])
print(f'final scores: pass means {np.round(S.mean(1), 2).tolist()}, SD {S.mean(1).std(ddof=1):.2f} vs independent-games prediction '
      f'{np.sqrt(S.var(0, ddof=1).sum()) / len(games):.2f}')
