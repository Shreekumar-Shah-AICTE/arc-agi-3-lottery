# B13 = B12 (b12_x6d) + one change: 16 concurrent LLM streams instead of 10 (gate, max-running-requests, CUDA graphs
# to bs 16, mamba cache 60 -> 80 so 16 x 3 running slots still leave 32 cached states, HiCache host tier so evicted
# prefixes reload over PCIe instead of re-prefilling). Boot falls back to the same 16-stream settings without HiCache
# if the HiCache server dies before health. Save run = dev probe: all 25 public games, 40 min each, 16 slots.
import json, copy, os, sys
d, kid = sys.argv[1:3]
n = json.load(open('/data/arc/kaggle/b12_x6d/b12_x6d.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)
# cell 5: priority gate slots
s = src(5); s = rep(s, "'ARC3_MAX_ACTIVE_STREAMS': 10,", "'ARC3_MAX_ACTIVE_STREAMS': 16,"); put(5, s)
# cell 13: server
s = src(13)
s = rep(s, "    MAXREQ=10,\n", "    MAXREQ=16,\n"); s = rep(s, "    CUDAGRAPH_MAXBS=10,\n", "    CUDAGRAPH_MAXBS=16,\n")
s = rep(s, "    MAMBA_CACHE=60,\n", "    MAMBA_CACHE=80,\n    HICACHE=True,\n")
s = rep(s, 'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})',
           'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, 12, 14, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})')
i = s.index('# ---- launch detached and wait for health ----'); head, tail = s[:i], s[i:]
tail = rep(tail, 'precache_model_thread.start()\n', '')
body = '\n'.join(('    ' + l if l.strip() else l) for l in tail.splitlines())
s = head + ('precache_model_thread.start()\n'
    'HICACHE_ARGS = ["--enable-hierarchical-cache", "--hicache-write-policy", "write_through"]\n'
    'def _launch_and_wait(args):\n    global proc, log_offset\n' + body + '\n    return proc.poll() is None\n\n'
    'if CFG.get("HICACHE"):\n'
    '    if not _launch_and_wait(args + HICACHE_ARGS):\n'
    '        print("B13: HiCache server died before health; relaunching without HiCache", flush=True)\n'
    '        _launch_and_wait(args)\n'
    'else:\n    _launch_and_wait(args)\n')
put(13, s)
# cell 17: dev probe on all 25 games, 40 min each
s = src(17)
a = "demo_excluded_games = [] if TRUE_SUBMISSION else ["; i = s.index(a); j = s.index(']', i + len(a))
s = s[:i] + "demo_excluded_games = []" + s[j + 1:]
s = rep(s, 'bm.solver.max_runtime_s_per_game = 5*60  # save run only proves the pipeline completes',
           'bm.solver.max_runtime_s_per_game = 40*60  # B13 dev probe: 25 games compete for 16 slots')
assert 'bm.solver.concurrency = 120' in s and 'bm.solver.max_runtime_s_per_game = 532*60' in s
put(17, s)
n['cells'][0]['source'] = ("# B13: B12 + 16 LLM streams\n\nPrivate experiment. Base: our B12 notebook (X6 v2 + D' slot priority). "
    "One change: 16 concurrent streams (gate, max-running-requests, CUDA graphs, mamba cache 80, HiCache host tier). "
    "Save run = dev probe on all 25 public games, 40 min each.\n\n").splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open('/data/arc/kaggle/b12_x6d/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
