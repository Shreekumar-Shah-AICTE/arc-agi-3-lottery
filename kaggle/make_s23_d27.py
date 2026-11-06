# S23: dense Qwen3.8-27B FP8 screens on the B12 runtime (Pennyroyal SGLang 0.5.19) and the B12 harness, FB25 instrument.
# One change vs B12-FB25: the served model + serving config (24 streams, 64K window, NEXTN MTP from the model's own head).
# usage: python3 make_s23_d27.py tc27|d27
import json, os, sys
arm = sys.argv[1]
ARMS = {
  'tc27': dict(dir='b23_tc27', kid='arc3-b23-tc27-fb25', title='ThinkingCap-Qwen3.8-27B-FP8',
               model_src='nischaydnk/thinkingcap-qwen38-27b-fp8/PyTorch/fp8/1', ds_src=None,
               roots="['/kaggle/input/models/nischaydnk/thinkingcap-qwen38-27b-fp8', '/kaggle/input/thinkingcap-qwen38-27b-fp8']",
               served='thinkingcap27b'),
  'd27':  dict(dir='b23_d27', kid='arc3-b23-d27-fb25', title='Qwen3.8-27B-FP8 (official, driessmit1 snapshot)',
               model_src=None, ds_src='driessmit1/qwen3-8-27b-fp8-hf-017b9c7a',
               roots="['/kaggle/input/datasets/driessmit1/qwen3-8-27b-fp8-hf-017b9c7a', '/kaggle/input/qwen3-8-27b-fp8-hf-017b9c7a']",
               served='qwen27b'),
}[arm]
STREAMS = 24
base = 'b12_fb25'
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)

s = src(5)
s = rep(s, "SERVER_STARTUP_TIMEOUT = 12 * 60\n", "SERVER_STARTUP_TIMEOUT = 25 * 60  # S23: new model + fallback ladder\n")
s = rep(s, "MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'\n",
        f"""def _s23_model_dir(roots):
    import glob as _g
    for root in roots:
        hits = sorted(_g.glob(root + '/config.json') + _g.glob(root + '/*/config.json') + _g.glob(root + '/*/*/config.json')
                      + _g.glob(root + '/*/*/*/config.json'))
        hits = [h for h in hits if os.path.isfile(os.path.join(os.path.dirname(h), 'model.safetensors.index.json'))]
        if hits: return os.path.dirname(hits[0])
    for h in sorted(_g.glob('/kaggle/input/**/model.safetensors.index.json', recursive=True)):
        print('candidate model dir:', os.path.dirname(h))
    raise FileNotFoundError(f'S23 model not found under {{roots}}')
MODEL_DIR         = _s23_model_dir({ARMS['roots']})
print('S23 MODEL_DIR', MODEL_DIR)
""")
s = rep(s, "DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'\n",
        "DRAFT_MODEL_DIR   = MODEL_DIR  # S23: NEXTN draft = the model's own MTP head\n")
s = rep(s, "SERVED_MODEL_NAME = 'flashnext'", f"SERVED_MODEL_NAME = {ARMS['served']!r}")
s = rep(s, "'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+12)*1024,", "'LOCAL_ANALYZER_CONTEXT_WINDOW': (52+12)*1024,  # S23: 24 streams share a ~1.5M-token pool")
s = rep(s, "'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024,", "'ARC3_CONTEXT_DRAIN_TOKENS': 32*1024,  # S23")
s = rep(s, "'ARC3_MAX_ACTIVE_STREAMS': 10,", f"'ARC3_MAX_ACTIVE_STREAMS': {STREAMS},  # S23: = --max-running-requests")
put(5, s)

c13 = src(13)
head = c13[:c13.index("# ---- validate target and create the read-only draft's SGLang view ----")]
head = rep(head, 'PREFIX = "/tmp/sgl-intel"', 'PREFIX = "/tmp/sgl-s23"')
for a, b in [("CTX=(116+12+8)*1024,", "CTX=(52+12+8)*1024,"), ("MEMFRAC=0.96,", "MEMFRAC=0.93,"),
             ("MAXREQ=10,", f"MAXREQ={STREAMS},"), ("CUDAGRAPH_MAXBS=10,", f"CUDAGRAPH_MAXBS={STREAMS},"),
             ("MAMBA_CACHE=60,", "MAMBA_CACHE=120,"), ("FRSPEC=True,", "FRSPEC=False,")]:
    head = rep(head, a, b)
put(13, head + open('/data/arc/kaggle/s23_launch_tail.py').read())
put(12, "## 5. Start serving\n\nS23: Pennyroyal SGLang serving the dense 27B FP8 model with its own MTP head (ladder: MTP -> no spec -> safe).")

c17 = src(17)
c17 = rep(c17, "bm.solver.concurrency = 10\n", f"bm.solver.concurrency = {STREAMS}\n")
put(17, c17)

n['cells'][0]['source'] = (f"# S23 {arm.upper()}-FB25: B12 harness on dense {ARMS['title']}\n\nPrivate dev screen (D65). Harness, prompts, "
    f"scheduler, sampling and the FB25 budget (25 public games, 121 min wall, all games admitted) = B12-FB25. One change: the engine. "
    f"{ARMS['title']} on the same Pennyroyal SGLang 0.5.19 runtime, NEXTN MTP from the model's own head, {STREAMS} active streams, "
    f"64K analyzer window (drain to 32K). Same wall time per game as the real run, so more streams = more tokens per game.\n\n").splitlines(True)
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
d = f"/data/arc/kaggle/{ARMS['dir']}"; os.makedirs(d, exist_ok=True); fn = ARMS['dir'] + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json'))
m.update(id='shreekumarshah/' + ARMS['kid'], title=ARMS['kid'], code_file=fn)
m['model_sources'] = [ARMS['model_src']] if ARMS['model_src'] else []
ds = [x for x in m['dataset_sources'] if 'pennyroyal' in x or 'taaf' in x]
if ARMS['ds_src']: ds.append(ARMS['ds_src'])
m['dataset_sources'] = ds
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, ARMS['kid'], m['dataset_sources'], m['model_sources'])
