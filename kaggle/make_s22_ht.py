# S22 HT arm = arc3-b12-sdt-fb25 build (kaggle/b12_sdt) + kaggle/s22_ht.patch (fresh-eyes analysts at every SDT fire).
#   python3 make_s22_ht.py /data/arc/kaggle/b12_ht arc3-b12-ht-fb25
# HT: at each SDT fire, ARC3_HT_K fresh-context requests (short system prompt, no history) get a level dossier (harness
# record, per-action effect table, the agent's memory notes, first and current board) through different lenses; their
# answers are appended to the SDT text. Tokens charged to the game. Non-fatal apply ("#S22 HT patch FAILED" = SDT replicate).
# Test: kaggle/tests/test_s22_ht.py.
import json, os, sys, copy
d, kid = sys.argv[1:3]
base = 'b12_sdt'
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)
assert src(4).startswith('%%writefile /kaggle/sdt.patch')
patch = open('/data/arc/kaggle/s22_ht.patch').read()
cell = copy.deepcopy(n['cells'][4]); cell['source'] = ('%%writefile /kaggle/ht.patch\n' + patch).splitlines(True)
n['cells'].insert(5, cell)   # old 6 -> 7
s = src(7)
OK = 'print("#S22 SDT patch applied" if _sdt_apply.returncode == 0 else f"#S22 SDT patch FAILED rc={_sdt_apply.returncode} (run = B12-FB25 replicate)", flush=True)\n'
s = rep(s, OK, OK +
        'if _sdt_apply.returncode == 0:\n'
        '    _ht_apply = subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/ht.patch"], cwd=f"{BUNDLE_DIR}/src", capture_output=True, text=True)\n'
        '    print(_ht_apply.stdout[-2000:], _ht_apply.stderr[-2000:])\n'
        '    print("#S22 HT patch applied" if _ht_apply.returncode == 0 else f"#S22 HT patch FAILED rc={_ht_apply.returncode} (run = SDT replicate)", flush=True)\n')
A = "    'ARC3_SDT_LOG': '/kaggle/working/sdt_events.jsonl',\n"
s = rep(s, A, A +
        "    ########## S22 HT: fresh-eyes analysts at every SDT fire ##########\n"
        "    'ARC3_HT_K': 2,                       # independent fresh-context analysts per fire (lenses: goal, controls)\n"
        "    'ARC3_HT_MAX_TOKENS': 12000,          # per analyst (thinking + answer)\n"
        "    'ARC3_HT_TIMEOUT_S': 420,\n")
put(7, s)
title = ("# B12-HT-FB25: SDT + fresh-eyes analysts\n\nPrivate experiment (S22). = arc3-b12-sdt-fb25 + a second harness patch "
         "(cell 5, /kaggle/ht.patch): at every SDT fire (60K level tokens, +50K, or a late death), two fresh-context requests "
         "(short system prompt, no conversation history) receive a level dossier (harness record, per-action effect table, the "
         "agent's memory notes, the level's first board and the current board) and each diagnoses the stall through a different "
         "lens (goal; controls and objects). Their answers (GOAL / MISSED / TEST) are appended to the SDT text. Analyst tokens are "
         "charged to the game. Events: ht lines in /kaggle/working/sdt_events.jsonl and #OURS_HT in the log. Cells 0, 5 (new), "
         "7 (old 6) differ from arc3-b12-sdt-fb25.\n\n")
n['cells'][0]['source'] = title.splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
assert m['docker_image'].endswith('32a8be4cb13c') and 'kaggle-private-byod' in m['docker_image']
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
