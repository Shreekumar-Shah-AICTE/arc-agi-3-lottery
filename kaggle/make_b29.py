"""B29 arms (D80/D81), built from b28_ctl_fb25 (the current control, D79).
b29_occ_fb25 = ctl + ARC3_MAX_ACTIVE_STREAMS 12 (server max_running_requests stays 10)."""
import json, copy, sys
from pathlib import Path
K = Path(__file__).resolve().parent
src = json.load(open(K/'b28_ctl_fb25/b28_ctl_fb25.ipynb')); meta = json.load(open(K/'b28_ctl_fb25/kernel-metadata.json'))
def cell(nb, i): return ''.join(nb['cells'][i]['source'])
def setcell(nb, i, s): nb['cells'][i]['source'] = s.splitlines(keepends=True)
def sub1(s, a, b):
    assert s.count(a) == 1, (a, s.count(a)); return s.replace(a, b)
def build(name, title_md, edits):
    nb = copy.deepcopy(src); m = copy.deepcopy(meta)
    for i, a, b in edits: setcell(nb, i, sub1(cell(nb, i), a, b))
    setcell(nb, 0, title_md + "\n\n" + cell(src, 0))
    m.update(id=f'shreekumarshah/arc3-{name.replace("_", "-")}', title=f'arc3-{name.replace("_", "-")}', code_file=f'{name}.ipynb')
    out = K/name; out.mkdir(exist_ok=True)
    json.dump(nb, open(out/f'{name}.ipynb', 'w'), indent=1); json.dump(m, open(out/'kernel-metadata.json', 'w'), indent=1)
    print('built', name, m['id'])
    return nb
if 'occ' in sys.argv[1:] or len(sys.argv) == 1:
    build('b29_occ_fb25', '# B29-OCC-FB25 (D80): control (B28-CTL) + priority gate 12 slots; server max_running_requests stays 10',
          [(5, "'ARC3_MAX_ACTIVE_STREAMS': 10,", "'ARC3_MAX_ACTIVE_STREAMS': 12,  # B29-OCC (D80): fill client-time gaps; server MAXREQ stays 10")])

# b29_safe (D81): submission build = b12_safe (40-min startup, ft09-only 5-min save run) + rerun-detection fix.
SAFE_FIX = '''TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
# B29 (S31, D81): promote-only structural check. A scored rerun reported 0.00 when KAGGLE_IS_COMPETITION_RERUN
# was missing (22 Oct, weitaoguo cell 22; forum 752727): the notebook then plays the offline branch and the
# gateway sees no actions. A Save & Run has neither ARC_BASE_URL nor a resolvable 'gateway' host.
def _b29_gateway_resolves(host="gateway", timeout=3.0):
    import socket, threading
    out = []
    t = threading.Thread(target=lambda: out.append(bool(socket.getaddrinfo(host, 8001))), daemon=True)
    t.start(); t.join(timeout)
    return bool(out and out[0])
try:
    _B29_SIG = {"env": TRUE_SUBMISSION, "arc_base_url": bool((os.environ.get("ARC_BASE_URL", "") or "").strip()),
                "gateway_dns": _b29_gateway_resolves()}
except Exception as _b29_exc:
    _B29_SIG = {"env": TRUE_SUBMISSION, "arc_base_url": bool((os.environ.get("ARC_BASE_URL", "") or "").strip()),
                "gateway_dns": False, "err": repr(_b29_exc)}
print("B29 rerun signals:", _B29_SIG, flush=True)
if not TRUE_SUBMISSION and (_B29_SIG["arc_base_url"] or _B29_SIG["gateway_dns"]):
    TRUE_SUBMISSION = True
    print("B29: TRUE_SUBMISSION promoted by structural signal", flush=True)'''
ORIG = 'TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}'
if 'safe' in sys.argv[1:]:
    import os as _os
    sf = [f for f in _os.listdir(K/'b12_safe') if f.endswith('.ipynb')][0]
    src = json.load(open(K/'b12_safe'/sf)); meta = json.load(open(K/'b12_safe/kernel-metadata.json'))
    build('b29_safe', '# B29-SAFE (D81): submission build = B12-safe (40-min startup, cheap ft09 save run) + rerun-detection fix',
          [(5, ORIG, SAFE_FIX)])
