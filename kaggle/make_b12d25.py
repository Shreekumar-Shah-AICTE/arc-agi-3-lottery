# B12-D25 = B12 (b12_x6d) unchanged except the save run: dev probe on all 25 public games, 40 min each,
# 10 slots (B12 serving). Matched baseline for B13 (same probe, 16 slots) and every later B-arm (D39).
import json, os, sys
d, kid = sys.argv[1:3]
n = json.load(open('/data/arc/kaggle/b12_x6d/b12_x6d.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)
s = src(17)
a = "demo_excluded_games = [] if TRUE_SUBMISSION else ["; i = s.index(a); j = s.index(']', i + len(a))
s = s[:i] + "demo_excluded_games = []" + s[j + 1:]
s = rep(s, 'bm.solver.max_runtime_s_per_game = 5*60  # save run only proves the pipeline completes',
           'bm.solver.max_runtime_s_per_game = 40*60  # B12-D25 dev probe: 25 games compete for 10 slots')
assert 'bm.solver.concurrency = 120' in s and 'bm.solver.max_runtime_s_per_game = 532*60' in s
put(17, s)
n['cells'][0]['source'] = ("# B12-D25: B12 baseline dev probe\n\nPrivate experiment. Code and serving = our B12 notebook (X6 v2 + D' slot priority, 10 streams). "
    "Only the save run differs: dev probe on all 25 public games, 40 min each, matched to B13.\n\n").splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open('/data/arc/kaggle/b12_x6d/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
