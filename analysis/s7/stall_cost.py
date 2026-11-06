# A3 test (round 4): how often does the LLM still clear a level after the frontier stall threshold, and how efficiently?
# Input: analysis/s7/level_attempts.json rows (run, game, level, tokens_on_level, actions_on_level, cleared), 9 logged runs, 415 attempts.
# Level score of an LLM clear = min(115, (h/a)^2 * 100) with h = public baseline actions (arc_agi game metadata).
import json, glob, statistics as st
A = json.load(open('/data/arc/analysis/s7/level_attempts.json'))
META = {}
for p in glob.glob('/data/a3local/environment_files/*/*/metadata.json'):
    try:
        m = json.load(open(p)); META[m['game_id'][:4]] = m
    except Exception:
        pass
DEV = {'b1', 'b1c', 'b3v2', 'b4', 'b4v2'}; HO = {'b5', 'b5c', 'b7', 'b7c'}
def score(g, L, a):
    h = META.get(g, {}).get('baseline_actions', [None] * 20)[L - 1] if g in META else None
    return None if not h or not a else min(115.0, (h / a) ** 2 * 100)
for name, runs in [('dev', DEV), ('holdout', HO), ('all', DEV | HO)]:
    for thr in (20000, 40000, 60000, 90000):
        S = [x for x in A if x[0] in runs and x[3] >= thr]
        C = [x for x in S if x[5]]
        sc = [score(x[1], x[2], x[4]) for x in C]; sc = [s for s in sc if s is not None]
        print(f'{name:8s} tokens>={thr//1000:3d}k: attempts {len(S):3d}  cleared later {len(C):3d} ({(len(C)/len(S)*100 if S else 0):4.1f}%)'
              f'  median level score of those clears {st.median(sc) if sc else float("nan"):6.1f}  (n scored {len(sc)})')
