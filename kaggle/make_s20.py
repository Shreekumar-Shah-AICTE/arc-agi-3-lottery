# S20 arm on the FB25 instrument (25 public games x 121 min, D47) = kaggle/b12_fb25 + one mechanism.
#   python3 make_s20.py rb /data/arc/kaggle/b12_rb arc3-b12-rb-fb25
#       RB: level-start checkpoint rollback on a stalled level (kaggle/s20_rb.patch, ARC3_RB=1). After ARC3_RB_TOKENS
#       generated tokens on one attempt at a level, the harness RESETs the level and restores the conversation, retained
#       functions and memory sections to the level-start checkpoint (one-line notice). Max ARC3_RB_MAX per level; none in
#       the last ARC3_RB_MIN_REMAINING_FRAC of the game's budget. Patch apply is NON-fatal: on failure the run is a B12-FB25
#       replicate (logged as "#S20 RB patch FAILED").
#   python3 make_s20.py wmk /data/arc/kaggle/b12_wmk_smoke arc3-b12-wmk-smoke [games=ka59,re86,...] [minutes=33]
#       WMK: world-model toolkit (kaggle/s20_wmk.patch, ARC3_WMK=1). Preloads check_model / plan / run_plan in the python
#       sandbox (retrodiction over this level's transitions, bounded BFS over the model's own simulator, verified
#       one-action-at-a-time execution) and adds a 6-line note to the system prompt. Optional game subset + per-game
#       minutes for a smoke run. Patch apply is NON-fatal ("#S20 WMK patch FAILED" = B12 replicate on that subset).
import json, os, sys, copy
arm, d, kid = sys.argv[1:4]
opts = dict(a.split('=', 1) for a in sys.argv[4:])
base = 'b12_fb25'
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)
ENV_ANCHOR = "    ########## controls ##########\n    'EXPOSE_UNDO': 'on',\n}\n"
assert src(5).count(ENV_ANCHOR) == 1
if arm not in ('rb', 'wmk'):
    raise SystemExit('arm?')
if arm == 'wmk':
    exec(open('/data/arc/kaggle/make_s20_wmk.py').read()); raise SystemExit(0)
patch = open('/data/arc/kaggle/s20_rb.patch').read()
cell = copy.deepcopy(n['cells'][3]); cell['source'] = ('%%writefile /kaggle/rb.patch\n' + patch).splitlines(True)
n['cells'].insert(4, cell)   # cells 4.. shift by one: old 5 -> 6
s = src(6)
s = rep(s, 'print("bline patch applied")\n',
        'print("bline patch applied")\n'
        '_rb_apply = subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/rb.patch"], cwd=f"{BUNDLE_DIR}/src", capture_output=True, text=True)\n'
        'print(_rb_apply.stdout[-2000:], _rb_apply.stderr[-2000:])\n'
        'print("#S20 RB patch applied" if _rb_apply.returncode == 0 else f"#S20 RB patch FAILED rc={_rb_apply.returncode} (run = B12-FB25 replicate)", flush=True)\n')
s = rep(s, ENV_ANCHOR, ENV_ANCHOR.replace("    'EXPOSE_UNDO': 'on',\n}\n",
    "    'EXPOSE_UNDO': 'on',\n\n"
    "    ########## S20 RB: level-start checkpoint rollback on a stalled level ##########\n"
    "    'ARC3_RB': '1',\n"
    "    'ARC3_RB_TOKENS': 50000,              # generated tokens on one attempt at a level before rollback\n"
    "    'ARC3_RB_MAX': 2,                     # rollbacks per level (3 attempts at most)\n"
    "    'ARC3_RB_MIN_REMAINING_FRAC': 0.15,   # never in the last 15% of the game's budget\n"
    "    'ARC3_RB_LOG': '/kaggle/working/rb_events.jsonl',\n"
    "}\n"))
put(6, s)
title = ("# B12-RB-FB25: level-start checkpoint rollback on a stalled level\n\nPrivate experiment (S20). = arc3-b12-fb25 "
         "(B12, 25 public games x 121 min) + a harness patch (cell 4, /kaggle/rb.patch): after 50K generated tokens on one "
         "attempt at a level, RESET the level and restore the conversation, retained functions and memory sections to the "
         "level-start checkpoint, with a one-line notice; at most 2 per level, none in the last 15% of a game's budget. "
         "Events in /kaggle/working/rb_events.jsonl. Cells 0, 4 (new), 6 (old 5) differ from arc3-b12-fb25.\n\n")
n['cells'][0]['source'] = title.splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
assert m['docker_image'].endswith('32a8be4cb13c') and 'kaggle-private-byod' in m['docker_image']
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
