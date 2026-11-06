# S24 (D68): Gemma-4-31B-it on the B12 runtime (Pennyroyal SGLang 0.5.19) and the B12 harness, D25 probe (25 games x 40 min).
# One change vs B12-D25: the engine = Google's QAT W4A16 compressed-tensors checkpoint + Google's QAT assistant
# (Frozen-KV MTP), served with Gemma 4 parsers, Google's current chat template and Google's sampling defaults.
# usage: python3 make_s24_g4.py
import json, os
DIR, KID = 'b24_g4', 'arc3-b24-g4-d25'
MODEL_SRCS = ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
              'google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized-assistant/2']
base = 'b12_fb25'
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)

s = src(5)
s = rep(s, "SERVER_STARTUP_TIMEOUT = 12 * 60\n", "SERVER_STARTUP_TIMEOUT = 25 * 60  # S24: new model + fallback ladder\n")
s = rep(s, "MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'\n",
        """def _s24_find(root, arch, quant):
    import glob as _g
    hits = []
    for h in sorted(_g.glob(root + '/**/config.json', recursive=True)) or sorted(_g.glob('/kaggle/input/**/config.json', recursive=True)):
        try: c = json.load(open(h))
        except Exception: continue
        d = os.path.dirname(h)
        has_w = os.path.isfile(os.path.join(d, 'model.safetensors')) or os.path.isfile(os.path.join(d, 'model.safetensors.index.json'))
        if has_w and (c.get('architectures') or [''])[0] == arch and (c.get('quantization_config') or {}).get('quant_method') == quant:
            hits.append(d)
    if not hits and root != '/kaggle/input':
        return _s24_find('/kaggle/input', arch, quant)
    if len(hits) != 1:
        for h in sorted(_g.glob('/kaggle/input/**/config.json', recursive=True)): print('candidate config:', h)
        raise FileNotFoundError(f'S24: expected one {arch} ({quant}) under {root}, found {hits}')
    return hits[0]
MODEL_DIR         = _s24_find('/kaggle/input/models/google/gemma-4', 'Gemma4ForConditionalGeneration', 'compressed-tensors')
print('S24 MODEL_DIR', MODEL_DIR)
""")
s = rep(s, "DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'\n",
        "DRAFT_MODEL_DIR   = _s24_find('/kaggle/input/models/google/gemma-4', 'Gemma4AssistantForCausalLM', None)  # Google's QAT assistant (BF16)\n"
        "print('S24 DRAFT_MODEL_DIR', DRAFT_MODEL_DIR)\n")
s = rep(s, "SERVED_MODEL_NAME = 'flashnext'", "SERVED_MODEL_NAME = 'gemma4'")
s = rep(s, "'LOCAL_ANALYZER_TEMPERATURE': '0.7',", "'LOCAL_ANALYZER_TEMPERATURE': '1.0',  # S24: Google's Gemma 4 defaults (generation_config)")
s = rep(s, "'LOCAL_ANALYZER_TOP_K': '20',", "'LOCAL_ANALYZER_TOP_K': '64',  # S24: Google's Gemma 4 default")
put(5, s)

c13 = src(13)
head = c13[:c13.index("# ---- validate target and create the read-only draft's SGLang view ----")]
head = rep(head, 'PREFIX = "/tmp/sgl-intel"', 'PREFIX = "/tmp/sgl-s24"')
for a, b in [("MEMFRAC=0.96,", "MEMFRAC=0.92,"), ("FRSPEC=True,", "FRSPEC=False,"),
             ("SPEC_STEPS=3,\n)", "SPEC_STEPS=3,\n    SWA_RATIO=0.08,  # S24: SWA-layer KV tokens / full-layer KV tokens (default 0.8 wastes the pool)\n)")]:
    head = rep(head, a, b)
tmpl = open('/data/arc/kaggle/s24_gemma4_chat_template.jinja').read()
tail = ("# Google's Gemma 4 chat template, published 2026-07-09 (huggingface.co/google/gemma-4-31b-it-qat-w4a16-ct)\n"
        f"GEMMA4_CHAT_TEMPLATE = {tmpl!r}\n\n" + open('/data/arc/kaggle/s24_g4_launch_tail.py').read())
put(13, head + tail)
put(12, "## 5. Start serving\n\nS24: Pennyroyal SGLang serving Google's Gemma-4-31B-it QAT W4A16 checkpoint with Google's QAT assistant "
        "(Frozen-KV MTP); ladder MTP -> no spec -> safe, each rung gated by a plain and an image + thinking + tool-call smoke test.")

c17 = src(17)
c17 = rep(c17, "bm.solver.max_runtime_s_per_game = 121*60  # FB25 (D47): 48.4 stream-min per game, = the real run (10 x 532 / 110)",
          "bm.solver.max_runtime_s_per_game = 40*60  # S24: = B12-D25 dev probe: 25 games compete for 10 slots")
put(17, c17)

n['cells'][0]['source'] = ("# S24 G4-D25: B12 harness on Gemma-4-31B-it (D68)\n\nPrivate dev screen (operator go, 11 Oct). Harness, prompts, "
    "scheduler, 10 streams, analyzer window and the D25 budget (25 public games x 40 min) = B12-D25. One change: the engine. "
    "Google's Gemma-4-31B-it QAT W4A16 compressed-tensors checkpoint with Google's QAT assistant model for multi-token "
    "prediction, on the same Pennyroyal SGLang 0.5.19 runtime, with Gemma 4 reasoning/tool parsers, Google's current chat "
    "template and Google's sampling defaults (temperature 1.0, top-p 0.95, top-k 64). Gemma 4 is Apache 2.0.\n\n").splitlines(True)
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
d = f"/data/arc/kaggle/{DIR}"; os.makedirs(d, exist_ok=True); fn = DIR + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json'))
m.update(id='shreekumarshah/' + KID, title=KID, code_file=fn)
m['model_sources'] = MODEL_SRCS
m['dataset_sources'] = [x for x in m['dataset_sources'] if 'pennyroyal' in x or 'taaf' in x]
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
assert m['docker_image'].endswith('57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c')
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, KID, m['dataset_sources'], m['model_sources'])
