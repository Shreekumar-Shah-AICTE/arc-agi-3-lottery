"""V7 preview (exploratory, 2 Nov): discovery vs execution. For every game run in data/runs, compare the
first level with later levels: actions used / human baseline actions on cleared levels, and the chance of
clearing a level given the previous one was cleared. Dev = 10 original dev games, hold = 15 holdout games.
f1_forklab is left out (4 Nov): its 256 plays are forks from saved mid-game states, so their first level is not a first try."""
import gzip, json, statistics as st, collections
from v3_hazard import DEV, HOLD
from common import HERE, dev_run_files
R = collections.defaultdict(lambda: collections.defaultdict(list))
for p in [f for f in dev_run_files() if not f.name.startswith('f1_forklab')]:  # forks start mid-game
    for g in json.load(gzip.open(p, 'rt'))['game_runs']:
        gid = g['game_id'].split('-')[0]; s = 'dev' if gid in DEV else 'hold' if gid in HOLD else None
        if not s: continue
        a, b, k, n = g['actions_per_level'], g['base_actions_per_level'], g['levels_completed'], g['number_of_levels']
        for i in range(min(k, n)):
            if b[i]: R[s]['r1' if i == 0 else 'r2'].append(a[i] / b[i])
        R[s]['att1'].append(1); R[s]['clr1'].append(int(k >= 1))
        for i in range(1, n):
            if k >= i: R[s]['attN'].append(1); R[s]['clrN'].append(int(k >= i + 1))
for s in ('dev', 'hold'):
    d = R[s]
    print(f"{s}: game runs {len(d['att1'])}; P(clear level 1) {st.mean(d['clr1']):.2f}; P(clear next | previous cleared) {st.mean(d['clrN']):.2f}; "
          f"median actions/human on cleared level 1 {st.median(d['r1']):.2f} (n={len(d['r1'])}) vs later levels {st.median(d['r2']):.2f} (n={len(d['r2'])})")
