# S22 SDT arm on the FB25 instrument (25 public games x 121 min, D47) = kaggle/b12_fb25 + kaggle/s22_sdt.patch.
#   python3 make_s22_sdt.py /data/arc/kaggle/b12_sdt arc3-b12-sdt-fb25
# SDT (stall digest + trouble prompt, D58 amended by D60): at 60K generated tokens on the current level (then +50K, max 2 per level) or
# on a GAME_OVER after > 50 level actions, the next user message gets a harness record of the level and the 2607.15439
# section 12.5 trouble prompt. Non-destructive: game state, history, helpers and the D' clock stay as they are.
# Patch apply is NON-fatal ("#S22 SDT patch FAILED" = B12-FB25 replicate). Test: kaggle/tests/test_s22_sdt.py.
import json, os, sys, copy
d, kid = sys.argv[1:3]
base = 'b12_fb25'
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)
ENV_ANCHOR = "    ########## controls ##########\n    'EXPOSE_UNDO': 'on',\n}\n"
assert src(5).count(ENV_ANCHOR) == 1
patch = open('/data/arc/kaggle/s22_sdt.patch').read()
cell = copy.deepcopy(n['cells'][3]); cell['source'] = ('%%writefile /kaggle/sdt.patch\n' + patch).splitlines(True)
n['cells'].insert(4, cell)   # cells 4.. shift by one: old 5 -> 6
s = src(6)
s = rep(s, 'print("bline patch applied")\n',
        'print("bline patch applied")\n'
        '_sdt_apply = subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/sdt.patch"], cwd=f"{BUNDLE_DIR}/src", capture_output=True, text=True)\n'
        'print(_sdt_apply.stdout[-2000:], _sdt_apply.stderr[-2000:])\n'
        'print("#S22 SDT patch applied" if _sdt_apply.returncode == 0 else f"#S22 SDT patch FAILED rc={_sdt_apply.returncode} (run = B12-FB25 replicate)", flush=True)\n')
s = rep(s, ENV_ANCHOR, ENV_ANCHOR.replace("    'EXPOSE_UNDO': 'on',\n}\n",
    "    'EXPOSE_UNDO': 'on',\n\n"
    "    ########## S22 SDT: stall digest + trouble prompt (non-destructive) ##########\n"
    "    'ARC3_SDT': '1',\n"
    "    'ARC3_SDT_TOKENS': 60000,             # generated tokens on the level before the first prompt (P(solve) 0.43)\n"
    "    'ARC3_SDT_TOKENS2': 50000,            # further tokens before the second\n"
    "    'ARC3_SDT_MAX': 2,                    # prompts per level\n"
    "    'ARC3_SDT_DEATH_ACTIONS': 50,         # 2607.15439 trigger: GAME_OVER after > 50 level actions\n"
    "    'ARC3_SDT_MIN_REMAINING_FRAC': 0.15,  # never in the last 15% of the game's budget\n"
    "    'ARC3_SDT_LOG': '/kaggle/working/sdt_events.jsonl',\n"
    "}\n"))
put(6, s)
title = ("# B12-SDT-FB25: stall digest + trouble prompt\n\nPrivate experiment (S22, built S21). = arc3-b12-fb25 "
         "(B12, 25 public games x 121 min) + a harness patch (cell 4, /kaggle/sdt.patch): at 60K generated tokens on a "
         "level (then +50K, at most 2 per level) or on a GAME_OVER after more than 50 level actions, the next user message "
         "gets a harness record of the level (actions by name, unused actions, deaths, no-ops, clicks, tokens per action) "
         "and the 2607.15439 section 12.5 trouble prompt. No game state, history, helper or scheduler change. Events in "
         "/kaggle/working/sdt_events.jsonl and as #OURS_SDT lines in the log. Cells 0, 4 (new), 6 (old 5) differ from "
         "arc3-b12-fb25.\n\n")
n['cells'][0]['source'] = title.splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
assert m['docker_image'].endswith('32a8be4cb13c') and 'kaggle-private-byod' in m['docker_image']
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
