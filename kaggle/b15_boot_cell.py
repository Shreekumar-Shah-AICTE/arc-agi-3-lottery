# B15: LHS vLLM runtime instead of SGLang. Only the server changes; the harness config is B12's (cell 5).
# The boot script is the Python body of lordhansolo/taaf-kaggle-source setup_commands.json (v306 stack). Its
# constants are patched to our served name and port, to 640 px boards (10x upscale of 64x64) and to B12's
# chat-template defaults (no reasoning_effort=xhigh). It runs detached; the wait below is B12's health loop.
import hashlib, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

LHS_BUNDLE_CANDIDATES = (Path('/kaggle/input/datasets/lordhansolo/taaf-kaggle-source'), Path('/kaggle/input/taaf-kaggle-source'))
LHS_BUNDLE_DIR = next((p for p in LHS_BUNDLE_CANDIDATES if (p / 'setup_commands.json').is_file()), None)
assert LHS_BUNDLE_DIR is not None, 'attach dataset lordhansolo/taaf-kaggle-source'
_cmd = json.loads((LHS_BUNDLE_DIR / 'setup_commands.json').read_text(encoding='utf-8'))[0]
_head = "<<'PYSETUP'\n"
body = _cmd[_cmd.index(_head) + len(_head):_cmd.rindex('\nPYSETUP')]
print('LHS boot body sha256', hashlib.sha256(body.encode()).hexdigest()[:16], '(built against B15_LHS_SHA)')

def _patch(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

body = _patch(body, "SERVED_MODEL_NAME = 'primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8'", f'SERVED_MODEL_NAME = {SERVED_MODEL_NAME!r}')
body = _patch(body, 'VLLM_PORT = 1234\n', f'VLLM_PORT = {SERVED_MODEL_PORT}\n')
body = _patch(body, 'VLLM_IMAGE_MAX_PIXELS = 65536\n', 'VLLM_IMAGE_MAX_PIXELS = 409600  # B15: 640x640 boards\n')
body = _patch(body, """'{"preserve_thinking": true, "reasoning_effort": "xhigh"}'""", """'{"preserve_thinking": true}'""")
# Our turns keep every board image in history (B13 logs: up to 58 per request), so pin the image limit
# rather than rely on vLLM's default. Profiling stays bounded by the 2048-token chunk (~5 images of 400 tokens).
body = _patch(body, "json.dumps({'video': 0})", "json.dumps({'image': 999, 'video': 0})")
for must in ('VLLM_MAX_NUM_SEQS = 14\n', 'VLLM_MAX_MODEL_LEN = 147072\n'):
    assert must in body, must
assert int(os.environ['ARC3_MAX_ACTIVE_STREAMS']) == 14

BOOT_PY = WORKING_DIR / 'b15_lhs_boot.py'
BOOT_LOG = WORKING_DIR / 'b15_boot.log'
BOOT_PY.write_text(body, encoding='utf-8')
boot_env = os.environ.copy()
boot_env.update(PYTHON=sys.executable, TAAF_KAGGLE_BUNDLE_DIR=str(LHS_BUNDLE_DIR),
                TAAF_KAGGLE_WORKING_DIR=str(WORKING_DIR), TAAF_KAGGLE_SETUP_ENV='/tmp/b15_lhs_setup_env.json')
with open(BOOT_LOG, 'ab', buffering=0) as logf:
    proc = subprocess.Popen([sys.executable, str(BOOT_PY)], env=boot_env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
print(f'pid {proc.pid} -> {BOOT_LOG}')

def show_log_tail(n=60):
    for p in (BOOT_LOG, WORKING_DIR / 'vllm-openai-server.log', WORKING_DIR / 'vllm-watchdog.log'):
        if p.exists():
            print(f'--- {p.name}'); print('\n'.join(p.read_text(errors='replace').splitlines()[-n:]))

server_ready = False
url, t0 = f'http://127.0.0.1:{SERVED_MODEL_PORT}/health', time.time()
last_report = -30
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                server_ready = True
                print(f"\nREADY after {int(time.time()-t0)}s -> {url.replace('/health','/v1')}")
                break
    except Exception:
        pass
    if proc.poll() not in (None, 0):  # the script exits 0 only after readiness; non-zero = failed boot
        print(f'boot script exited {proc.returncode} before the server was healthy')
        show_log_tail()
        break
    elapsed = int(time.time()-t0)
    if elapsed-last_report >= 30:
        print(f'  {elapsed}s ...', flush=True); last_report = elapsed
    time.sleep(3)
else:
    print(f"\nDEADLINE at {int(time.time()-NOTEBOOK_START_TIME)}s from notebook start; "
          f"releasing the benchmark with the server still loading")

# Probe: does the chat template render reasoning sent back in history (preserve_thinking)? B12's harness
# sends it under ARC3_REASONING_HISTORY_KEY. Keep B12's key when it renders; otherwise switch to one that does.
def _tok_count(messages):
    req = dict(model=SERVED_MODEL_NAME, messages=messages, add_generation_prompt=True,
               chat_template_kwargs={'preserve_thinking': True})
    rq = urllib.request.Request(f'http://127.0.0.1:{SERVED_MODEL_PORT}/tokenize', data=json.dumps(req).encode(),
                                headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(rq, timeout=60) as r:
        return int(json.loads(r.read())['count'])

if server_ready:
    try:
        marker = ' '.join(['zebra quartz'] * 40)
        hist = lambda key: [{'role': 'user', 'content': 'Board A.'},
                            dict({'role': 'assistant', 'content': 'Moving up.'}, **({key: marker} if key else {})),
                            {'role': 'user', 'content': 'Board B.'}]
        counts = {k or 'none': _tok_count(hist(k)) for k in (None, 'reasoning_content', 'reasoning')}
        renders = [k for k in ('reasoning_content', 'reasoning') if counts[k] >= counts['none'] + 40]
        if os.environ.get('ARC3_REASONING_HISTORY_KEY', '') not in renders and renders:
            os.environ['ARC3_REASONING_HISTORY_KEY'] = renders[0]
        print(f"#B15_REASONING_PROBE counts={counts} renders={renders} key={os.environ.get('ARC3_REASONING_HISTORY_KEY')}"
              + ('' if renders else ' WARNING: history reasoning not rendered'), flush=True)
    except Exception as exc:
        print(f'#B15_REASONING_PROBE failed: {exc!r}', flush=True)

for line in BOOT_LOG.read_text(errors='replace').splitlines():
    if any(k in line for k in ('KV cache', 'concurrency', 'Image runtime ready', 'Runtime patch', 'Traceback', 'Error')):
        print('boot>', line[:300])
