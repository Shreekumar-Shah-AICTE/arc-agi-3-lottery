"""B30 safety arms (D82), built from b29_safe (= b12_safe + D81 promote-only rerun detection).
b30_safe     = b29_safe + DNS probe catches its own exception + startup wait 50 min + submission end-time cap.
b30_safe_occ = b30_safe + ARC3_MAX_ACTIVE_STREAMS 12 (B29-OCC knob, D80); build only if OCC is not KILL (D84: KILL).
b30_safe_spec2 / b30_safe_spec1 = b30_safe + SPEC_STEPS 3 -> 2 / 1 in cell 13 (B31 knob, D85); build only if that arm is GO."""
import json, copy, sys
from pathlib import Path
K = Path(__file__).resolve().parent
def cell(nb, i): return ''.join(nb['cells'][i]['source'])
def setcell(nb, i, s): nb['cells'][i]['source'] = s.splitlines(keepends=True)
def sub1(s, a, b):
    assert s.count(a) == 1, (a[:60], s.count(a)); return s.replace(a, b)
OLD_PROBE = "    t = threading.Thread(target=lambda: out.append(bool(socket.getaddrinfo(host, 8001))), daemon=True)\n"
NEW_PROBE = '''    def _probe():
        try:
            out.append(bool(socket.getaddrinfo(host, 8001)))
        except Exception:  # B30 (D82): unresolvable 'gateway' is the normal Save & Run outcome; no thread traceback
            out.append(False)
    t = threading.Thread(target=_probe, daemon=True)
'''
OLD_WAIT = "SERVER_STARTUP_TIMEOUT = 40 * 60"
NEW_WAIT = "SERVER_STARTUP_TIMEOUT = 50 * 60  # B30 (D82): 40 -> 50; with the 900 s first-request grace, tolerates READY by ~65 min"
ANCHOR = "# Play the benchmark; teardown commands run even if the run raises.\n"
END_CAP = '''# B30 (S32, D82): submission end-time cap. Competition config: maxGpuRuntimeMinutes = 540 (forum 712719:
# 'Notebook Timeout', no score). Each game may run 532 min counted from benchmark start, which follows the
# server wait and the gateway wait. B12-family reruns scored with READY near kernel minute 11, i.e. game
# deadlines near kernel minute 543. Keep every run inside that envelope: a slow start shortens play instead
# of pushing the end past it. Fast starts (READY <= ~11 min) are unchanged.
B30_END_KERNEL_MIN = 543
_b30_old = float(bm.solver.max_runtime_s_per_game or 0.0)
_b30_left = NOTEBOOK_START_TIME + B30_END_KERNEL_MIN * 60 - time.time()
_b30_cap = min(_b30_old, max(30 * 60, _b30_left)) if _b30_old > 0 else max(30 * 60, _b30_left)
print(f"B30 end cap: elapsed {time.time() - NOTEBOOK_START_TIME:.0f}s; per-game {_b30_old:.0f}s -> {_b30_cap:.0f}s "
      f"({'applied' if TRUE_SUBMISSION else 'dry run, not applied'})", flush=True)
if TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = _b30_cap

'''
OCC_A = "'ARC3_MAX_ACTIVE_STREAMS': 10,"
OCC_B = "'ARC3_MAX_ACTIVE_STREAMS': 12,  # B29-OCC (D80): fill client-time gaps; server MAXREQ stays 10"
def build(name, title_md, occ=False, spec=None):
    src = json.load(open(K/'b29_safe/b29_safe.ipynb')); meta = json.load(open(K/'b29_safe/kernel-metadata.json'))
    nb = copy.deepcopy(src); m = copy.deepcopy(meta)
    c5 = sub1(sub1(cell(nb, 5), OLD_PROBE, NEW_PROBE), OLD_WAIT, NEW_WAIT)
    if occ: c5 = sub1(c5, OCC_A, OCC_B)
    setcell(nb, 5, c5)
    setcell(nb, 22, sub1(cell(nb, 22), ANCHOR, END_CAP + ANCHOR))
    if spec is not None:
        setcell(nb, 13, sub1(cell(nb, 13), "SPEC_STEPS=3,", f"SPEC_STEPS={spec},  # B31 (D85): 3 -> {spec}; draft tokens {spec + 1}"))
    c0 = cell(src, 0); assert c0.startswith('# B29-SAFE'); setcell(nb, 0, title_md + "\n\n" + c0)
    m.update(id=f'shreekumarshah/arc3-{name.replace("_", "-")}', title=f'arc3-{name.replace("_", "-")}', code_file=f'{name}.ipynb')
    out = K/name; out.mkdir(exist_ok=True)
    json.dump(nb, open(out/f'{name}.ipynb', 'w'), indent=1); json.dump(m, open(out/'kernel-metadata.json', 'w'), indent=1)
    print('built', name, m['id']); return nb
if __name__ == '__main__':
    args = sys.argv[1:] or ['safe']
    if 'safe' in args:
        build('b30_safe', '# B30-SAFE (D82): B29-SAFE + DNS probe exception fix + 50-min startup wait + submission end-time cap (kernel min 543)')
    if 'occ' in args:
        build('b30_safe_occ', '# B30-SAFE-OCC (D82): B30-SAFE + priority gate 12 slots (B29-OCC knob, D80); server max_running_requests stays 10', occ=True)
    for n in (2, 1):
        if f'spec{n}' in args:
            build(f'b30_safe_spec{n}', f'# B30-SAFE-SPEC{n} (D85): B30-SAFE + speculative depth 3 -> {n} (draft tokens {n + 1}); B31-SPEC{n} knob', spec=n)
