# B12 = X6 v2 (b4_x6) + D' slot priority (shiiin9 Form D1 cell, verbatim) + minimal save run (ft09, 5 min).
# TRUE_SUBMISSION settings are unchanged from X6 v2.
import json, copy, os, sys
d, kid = sys.argv[1:3]
x6 = json.load(open('/data/arc/kaggle/b4_x6/b4_x6.ipynb'))
d1 = json.load(open('/data/intel/k3/shiiin9_affectify-arc-dfranzen-form-d1/affectify-arc-dfranzen-form-d1.ipynb'))
n = copy.deepcopy(x6)
# hook cell: save run plays ft09 only for 5 minutes
hk = [c for c in n['cells'] if 'bm.solver.max_runtime_s_per_game = 25*60' in ''.join(c['source'])]; assert len(hk) == 1
hs = ''.join(hk[0]['source'])
a = "demo_excluded_games = [] if TRUE_SUBMISSION else ["; i = hs.index(a); j = hs.index(']', i + len(a))
d1hook = ''.join(d1['cells'][17]['source']); i2 = d1hook.index(a); j2 = d1hook.index(']', i2 + len(a))
hs = hs[:i] + d1hook[i2:j2] + hs[j:]
hs = hs.replace('bm.solver.max_runtime_s_per_game = 25*60 #532*60 * bm.solver.concurrency // 110',
                'bm.solver.max_runtime_s_per_game = 5*60  # save run only proves the pipeline completes')
assert "'ar25', 'bp35'" in hs and '5*60  # save run' in hs and 'bm.solver.max_runtime_s_per_game = 532*60' in hs
hk[0]['source'] = hs.splitlines(True)
# D' cell before the run cell, print cell after it
dcell = copy.deepcopy(d1['cells'][21]); pcell = copy.deepcopy(d1['cells'][23])
assert 'install_d(_fp_ta, _fp_solver)' in ''.join(dcell['source']) and '#OURS_FRESH' in ''.join(pcell['source'])
ri = [k for k, c in enumerate(n['cells']) if 'await bm.run(' in ''.join(c['source'])]; assert len(ri) == 1; ri = ri[0]
n['cells'].insert(ri + 1, pcell); n['cells'].insert(ri, dcell)
n['cells'][0]['source'] = ("# B12: X6 v2 + D' slot priority\n\nPrivate experiment. Base: our X6 v2 notebook (Franzen M2 fork + X6 patch). "
    "Adds the D' slot-priority formula from shiiin9 Form D1 (embedded verbatim). Save run plays ft09 for 5 minutes only.\n\n").splitlines(True) + (lambda s: s.splitlines(True) if isinstance(s, str) else s)(n['cells'][0]['source'])
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open('/data/arc/kaggle/b4_x6/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
