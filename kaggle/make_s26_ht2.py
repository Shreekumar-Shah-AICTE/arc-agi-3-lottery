# S26 HT2 arm = arc3-b12-ht-fb25 build (kaggle/b12_ht) with the analysts' thinking OFF.
#   python3 make_s26_ht2.py /data/arc/kaggle/b12_ht2 arc3-b12-ht2-fb25
# Why: S22 HT was inert (D70): 13/14 SDT fires hit the 2 x 12,000-token analyst cap after ~400 s, 0/28 answers.
# HT2 = same dossier, lenses and SDT trigger; analysts answer directly (enable_thinking=False, 1,500-token cap,
# 150 s timeout, preamble before "GOAL:" dropped). Patch: kaggle/s26_ht2.patch (replaces s22_ht.patch in cell 5).
# Test: kaggle/tests/test_s26_ht2.py. Cells 0, 5, 7 differ from b12_ht; everything else identical.
import json, os, sys, copy
d, kid = sys.argv[1:3]
base = 'b12_ht'
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)
assert src(5).startswith('%%writefile /kaggle/ht.patch')
put(5, '%%writefile /kaggle/ht.patch\n' + open('/data/arc/kaggle/s26_ht2.patch').read())
s = src(7)
s = rep(s, "    'ARC3_HT_MAX_TOKENS': 12000,          # per analyst (thinking + answer)\n    'ARC3_HT_TIMEOUT_S': 420,\n",
        "    'ARC3_HT_THINKING': '0',              # S26 HT2: analysts answer directly (S22 HT: 13/14 fires capped, 0/28 answers)\n"
        "    'ARC3_HT_MAX_TOKENS': 1500,           # per analyst answer (no thinking)\n"
        "    'ARC3_HT_TIMEOUT_S': 150,\n")
put(7, s)
title = ("# B12-HT2-FB25: SDT + fresh-eyes analysts that answer directly\n\nPrivate experiment (S26, D71). = arc3-b12-ht-fb25 "
         "with the analysts' thinking OFF. S22 HT was inert (D70: 13 of 14 SDT fires hit the 2 x 12,000-token analyst cap after "
         "~400 s, 0 of 28 answers). HT2 keeps the dossier, the two lenses (goal; controls and objects) and the SDT trigger (60K "
         "level tokens, +50K, or a late death); each analyst now answers directly (enable_thinking false, 1,500-token cap, 150 s "
         "timeout, any preamble before GOAL: dropped). Patch: cell 5 (/kaggle/ht.patch = kaggle/s26_ht2.patch). Events: ht lines "
         "(with thinking=false) in /kaggle/working/sdt_events.jsonl and #OURS_HT in the log. Cells 0, 5, 7 differ from "
         "arc3-b12-ht-fb25.\n\n")
n['cells'][0]['source'] = title.splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
assert m['docker_image'].endswith('32a8be4cb13c') and 'kaggle-private-byod' in m['docker_image']
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
