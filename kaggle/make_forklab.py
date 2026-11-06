# Build the Fork Lab notebook from the X6 notebook (b4_x6 = submitted config). Usage: make_forklab.py dir slug plan.json LAB_MINUTES FORK_TOKENS FORK_WALL_MIN
import json, copy, os, sys
d, kid, planf, lab_min, fork_tok, fork_wall = sys.argv[1:7]
n = json.load(open('/data/arc/kaggle/b4_x6/b4_x6.ipynb'))
patch = open('/data/arc/lab/forklab/forklab.patch').read(); plan = open(planf).read().strip()
rep = open('/data/arc/lab/forklab/report.py').read()
def code(src):
    c = copy.deepcopy(n['cells'][3]); c['source'] = src.splitlines(True); c['outputs'] = []; c['execution_count'] = None; return c
cells = n['cells']
cells.insert(4, code("# Input-path resolver: new kernels can mount inputs at a different layout than the parent notebook expects.\n# Build /kaggle/xin with the expected layout as symlinks to wherever Kaggle actually mounted each input.\nimport os, subprocess\ndef _walk(root='/kaggle/input', depth=6):\n    out = []\n    for d, subs, files in os.walk(root, followlinks=True):\n        if d.count('/') - root.count('/') >= depth: subs[:] = []\n        out.append((d, set(subs), set(files)))\n    return out\n_W = _walk()\ndef _pick(pred, label):\n    hits = sorted((d for d, s, f in _W if pred(d, s, f)), key=len)\n    print('resolve', label, '->', hits[:3]); return hits[0] if hits else None\n_MAP = {\n  'datasets/dfranzen/pennyroyal-v253': _pick(lambda d, s, f: os.path.basename(d) == 'pennyroyal-v253' or ('pennyroyal' in d and any(x.endswith('.whl') for x in f)), 'wheelhouse'),\n  'datasets/dfranzen/taaf-kaggle-source-bundle-copy': _pick(lambda d, s, f: 'taaf-kaggle-bundle.json' in f, 'bundle'),\n  'models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1': _pick(lambda d, s, f: 'config.json' in f and 'drafter' in d, 'drafter'),\n  'models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1': _pick(lambda d, s, f: 'config.json' in f and 'autoround' in d and 'drafter' not in d, 'model'),\n  'competitions/arc-prize-2026-arc-agi-3': _pick(lambda d, s, f: 'arc_agi_3_wheels' in s, 'competition'),\n}\nfor rel, real in _MAP.items():\n    dst = '/kaggle/xin/' + rel\n    os.makedirs(os.path.dirname(dst), exist_ok=True)\n    if os.path.islink(dst): os.unlink(dst)\n    if real: os.symlink(real, dst)\n    else: print('MISSING input for', rel)\nsubprocess.run('ls /kaggle/input; ls /kaggle/input/* | head -20', shell=True)\nassert all(_MAP.values()), 'an input is missing'\n"))
cells.insert(5, code('%%writefile /kaggle/forklab.patch\n' + patch))
import gzip, base64
_b64 = base64.b64encode(gzip.compress(plan.encode(), 9)).decode()
cells.insert(6, code("import gzip, base64\nopen('/kaggle/forklab_plan.json', 'wb').write(gzip.decompress(base64.b64decode('" + _b64 + "')))\nprint('forklab plan written')\n"))
cells.insert(7, code('%%writefile /kaggle/forklab_report.py\n' + rep))
c = cells[9]; s = ''.join(c['source'])
a = 'print("bline patch applied")\n'; assert s.count(a) == 1
assert s.count('!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR') == 1
s = s.replace('!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR', '!mkdir -p $BUNDLE_DIR && cp -a $ORIG_BUNDLE_DIR/. $BUNDLE_DIR/')
s = s.replace(a, a + 'subprocess.run(["git", "apply", "-v", "/kaggle/forklab.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)\nprint("forklab patch applied")\n')
a = "setup_env = {\n"; assert s.count(a) == 1
s = s.replace(a, a + "    'FORKLAB_PLAN': '/kaggle/forklab_plan.json',\n    'FORKLAB_RESULTS': '/kaggle/working/forklab_results.jsonl',\n")
a = "TRUE_SUBMISSION = os.environ.get"; assert s.count(a) == 1
c['source'] = s.splitlines(True)
hk = [cc for cc in cells if 'bm.solver.max_runtime_s_per_game = 25*60' in ''.join(cc['source'])]; assert len(hk) == 1
hs = ''.join(hk[0]['source'])
hs += f"""
# ---- Fork Lab overrides (lab notebook, never a submission) ----
assert not TRUE_SUBMISSION, 'Fork Lab notebook must never be submitted'
FORKLAB = json.load(open('/kaggle/forklab_plan.json'))
LAB_MINUTES = {int(lab_min)}
bm.solver.concurrency = 10
bm.solver.max_runtime_s_per_game = {int(fork_wall)}*60
bm.solver.max_generated_tokens_per_game = {int(fork_tok)}
bm.solver.save_request_logs = True
bm.n_passes = 1
print('forklab:', len(FORKLAB), 'forks, LAB_MINUTES', LAB_MINUTES, 'tokens/fork', bm.solver.max_generated_tokens_per_game)
"""
hk[0]['source'] = hs.splitlines(True)
rk = [cc for cc in cells if 'bm.games = _offline_games(competition_env_files)' in ''.join(cc['source'])]; assert len(rk) == 1
rs = ''.join(rk[0]['source'])
rs = rs.replace('    bm.games = _offline_games(competition_env_files)',
 """    import arc_agi, taaf.game_api
    _spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=competition_env_files)
    bm.games = [taaf.game_api.GameAPI(env_name=f['game_id'], arcade_spec=_spec) for f in FORKLAB]""")
a = "# Play the benchmark; teardown commands run even if the run raises.\n"; assert rs.count(a) == 1
rs = rs.replace(a, "soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(minutes=LAB_MINUTES - 8)\nprint('forklab soft end', soft_end)\n" + a)
rk[0]['source'] = rs.splitlines(True)
i = cells.index(rk[0])
cells.insert(i + 1, code(f"import subprocess\nsubprocess.run(['python', '/kaggle/forklab_report.py', '/kaggle/working/forklab_results.jsonl', '{int(fork_tok)}', '/kaggle/working/forklab_report.json'])\n"))
for cc in cells:
    src = ''.join(cc['source'])
    if cc['cell_type'] == 'code' and not src.startswith('%%writefile') and 'Input-path resolver' not in src and 'b64decode' not in src:
        cc['source'] = src.replace('/kaggle/input/datasets/', '/kaggle/xin/datasets/').replace('/kaggle/input/models/', '/kaggle/xin/models/').replace('/kaggle/input/competitions/', '/kaggle/xin/competitions/').splitlines(True)
for cc in cells:
    if cc['cell_type'] == 'code': cc['outputs'] = []; cc['execution_count'] = None
os.makedirs(d, exist_ok=True); json.dump(n, open(f'{d}/{os.path.basename(d)}.ipynb', 'w'), indent=1)
m = json.load(open('/data/arc/kaggle/b4_x6/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=f'{os.path.basename(d)}.ipynb')
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, 'forks', len(json.loads(plan)))
