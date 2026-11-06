"""Lever 5b (4 Nov 2026, exploratory, CPU only, before the V5 readout).

Question: can the agent's own behaviour flag a 'wall' (a level attempt that will not crack soon) better than elapsed
tokens alone? The lever 5 simulator says elapsed-token scheduling gains 0% while wall knowledge gains 13-17%.

Data: every downloaded dev-run benchmark.json (data/runs). actions_per_level partitions each game's history exactly, so
level attempt i = history[cum[i-1]:cum[i]]; attempts i < levels_completed cracked, attempt i == levels_completed is open.
At token checkpoints C in {5k, 10k, 20k, 40k} (tokens spent inside the attempt) we take every attempt that survived to
C and had either cracked later or kept playing for at least H more tokens (censoring rule). Target: no crack within the
next H = 30k tokens ('wall-like'). Features use only the attempt's history up to C plus the game's earlier levels, which
the agent could know at run time (no human baselines, no game identity). Evaluation: leave-one-game-out logistic
regression, AUC against the target. Elapsed tokens are constant within a checkpoint, so AUC 0.5 is the elapsed-only bar.
"""
import gzip, json, math
from collections import Counter, defaultdict
import numpy as np
from common import HERE, dev_run_files
H = 30_000
CHECKS = [5_000, 10_000, 20_000, 40_000]

def attempts(game):
    h = game.get('history') or []; apl = game.get('actions_per_level') or []; L = game.get('levels_completed') or 0
    cum = np.cumsum([0] + list(apl))
    prev = []
    for i in range(min(len(apl), L + 1)):
        seg = h[cum[i]:cum[i + 1]]
        if not seg: break
        yield i, seg, i < L, list(prev)
        prev.append(sum(x.get('generated_tokens') or 0 for x in seg))

def feats(seg_upto, i, prev):
    ids = [x['action']['id'] for x in seg_upto]
    tok = [x.get('generated_tokens') or 0 for x in seg_upto]
    calls = [t for t in tok if t > 0]
    n = len(ids)
    c = Counter(ids)
    p = np.array(list(c.values()), float) / max(n, 1)
    ent = float(-(p * np.log(p)).sum()) if n else 0.0
    tri = [tuple(ids[k:k + 3]) for k in range(max(n - 2, 0))]
    rep3 = 1 - len(set(tri)) / len(tri) if tri else 0.0
    clicks = [(x['action'].get('data') or {}) for x in seg_upto if x['action']['id'] == 'ACTION6']
    uclick = len({(d.get('x'), d.get('y')) for d in clicks})
    tot = sum(tok)
    return dict(
        log_actions=math.log1p(n),                       # actions taken within the C tokens (tempo)
        log_calls=math.log1p(len(calls)),
        log_max_call=math.log1p(max(calls) if calls else 0),
        frac_tokens_last_call=(calls[-1] / tot) if calls and tot else 0.0,
        actions_per_call=n / max(len(calls), 1),
        n_reset=c.get('RESET', 0),
        action_entropy=ent,
        n_distinct_actions=len(c),
        rep3=rep3,
        frac_click=len(clicks) / max(n, 1),
        click_novelty=uclick / max(len(clicks), 1) if clicks else 0.0,
        level_index=i,
        log_prev_mean=math.log1p(np.mean(prev)) if prev else math.log(22_000),
        has_prev=1.0 if prev else 0.0,
    )

rows = []
for p in dev_run_files():
    run = p.name.split('.')[0]
    d = json.load(gzip.open(p, 'rt'))
    for g in d['game_runs']:
        game = g['game_id'].split('-')[0]
        for i, seg, cracked, prev in attempts(g):
            tok = np.cumsum([x.get('generated_tokens') or 0 for x in seg]); total = tok[-1] if len(tok) else 0
            for C in CHECKS:
                if total < C and not (cracked and total >= C): continue
                if total <= C: continue                              # cracked or stopped before reaching C
                k = int(np.searchsorted(tok, C, side='right'))       # steps fully inside the first C tokens
                if k == 0: continue
                rem = total - tok[k - 1]
                if cracked:
                    y = 1 if rem > H else 0                          # cracked, but only after more than H tokens
                elif rem >= H:
                    y = 1                                            # open attempt that ran H more without a crack
                else:
                    continue                                         # censored: unknown
                rows.append(dict(run=run, game=game, C=C, y=y, **feats(seg[:k], i, prev)))

FEATS = [k for k in rows[0] if k not in ('run', 'game', 'C', 'y')]

def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def logit_fit(X, y, l2=1.0, it=50):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        z = X @ w; pr = 1 / (1 + np.exp(-z)); g = X.T @ (pr - y) + l2 * np.r_[0, w[1:]]
        Hm = (X * (pr * (1 - pr))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(Hm, g)
    return w

print(f'level-attempt checkpoints: {len(rows)} rows, {len({r["run"] for r in rows})} runs, {len({r["game"] for r in rows})} games; H = {H}')
for C in CHECKS:
    R = [r for r in rows if r['C'] == C]
    y = np.array([r['y'] for r in R]); games = np.array([r['game'] for r in R])
    X = np.array([[r[f] for f in FEATS] for r in R], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = np.c_[np.ones(len(R)), (X - mu) / sd]
    pred = np.zeros(len(R))
    for gname in sorted(set(games)):
        tr, te = games != gname, games == gname
        w = logit_fit(Xs[tr], y[tr]); pred[te] = Xs[te] @ w
    uni = sorted(((auc(X[:, j], y), f) for j, f in enumerate(FEATS)), key=lambda t: -abs(t[0] - 0.5))
    print(f'\nC = {C // 1000}k tokens into the attempt: n {len(R)}, wall-like share {y.mean():.2f}')
    print(f'  leave-one-game-out logistic AUC {auc(pred, y):.3f}')
    print('  univariate AUC (0.5 = no signal; <0.5 = higher value means crack soon):')
    for a, f in uni[:8]:
        print(f'    {f:22s} {a:.3f}')
