# P1: Dots3-Note-Preview IQ2_XXS (bartowski GGUF, public Kaggle dataset grennwolf/dots3-note-prev-iq2-xxs) on L4x4 with
# llama.cpp b10569 (first release with Dots3 text inference). Private probe, never submitted.
# P1c (session 11) = P1b + build fix (-DGGML_CUDA_NO_VMM=ON: P1b died on CUDA::cuda_driver) + pre-registered operating point
# (D37): the accuracy box runs at the thinking budget and worker count a 259-input, 12-h rerun could afford; KILL early if <12k.
# Questions: (1) build time, load time, VRAM fit; (2) aggregate decode tok/s vs concurrency for -sm layer and -sm tensor;
# (3) pass@1 at a forced 30k thinking budget (llama.cpp reasoning_budget_tokens), natural stop lengths and finish rate on the
#     E2.1 bench order (Random(2026) shuffle of public-eval ids) in a time box.
import os, sys, json, glob, re, time, random, subprocess, threading, hashlib, shutil, urllib.request
T0 = time.time(); IN = os.environ.get('P1_IN', '/kaggle/input'); OUT = os.environ.get('P1_OUT', '/kaggle/working')
FAKE = os.environ.get('P1_FAKE_SRV', '')   # local dry run: path of a fake llama-server script, skips the build
RERUN = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'1', 'true'}
END_ALL = T0 + float(os.environ.get('P1_END_S', str(2.0 * 3600)))       # hard stop for the whole probe
MAXTOK = int(os.environ.get('P1_MAXTOK', '36864'))      # thinking budget + answer room
BUDGET = int(os.environ.get('P1_BUDGET', '30000'))      # llama.cpp per-request reasoning budget; forced end of thinking
BUDGET_MSG = '\n\nI have used my thinking budget. I will stop here and write the final test output grid now.\n'
NTASK = int(os.environ.get('P1_NTASK', '20'))
PORT = 8080; URL = f'http://127.0.0.1:{PORT}'
RES = {'t0': T0, 'events': []}
def log(*a): print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
def note(k, v): RES[k] = v; RES['events'].append([round(time.time() - T0), k]); json.dump(RES, open(f'{OUT}/p1_result.json', 'w'), indent=1, default=str)
def sh(c, timeout=None):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)
    log(f'$ {c[:160]} -> rc={r.returncode}\n' + (r.stdout + r.stderr)[-3000:]); return r
def first(pat):
    g = sorted(glob.glob(f'{IN}/**/{pat}', recursive=True)); return g[0] if g else None
if RERUN:   # never meant to be submitted: write a valid all-zero file and stop
    t = json.load(open(first('arc-agi_test_challenges.json')))
    json.dump({k: [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in t[k]['test']] for k in t}, open(f'{OUT}/submission.json', 'w')); sys.exit(0)
sh('nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; nproc; free -g; df -h /tmp /kaggle/working | tail -2; which cmake nvcc gcc g++; nvcc --version | tail -2; cmake --version | head -1; ls /usr/local | head -20')
PRE = None if FAKE else first('lcpp-bin/llama-server')   # prebuilt binary from an attached P1 output, if runnable
if PRE:
    shutil.copy(PRE, '/tmp/llama-server'); os.chmod('/tmp/llama-server', 0o755)
    if sh('/tmp/llama-server --version').returncode != 0: PRE = None
if FAKE: SRV = FAKE; note('build_s', 0)
elif PRE: SRV = '/tmp/llama-server'; note('build_s', 0); log('using prebuilt', PRE)
else:
    # ---- 1. build llama.cpp (static, CUDA sm_89) ----
    src = first('llama.cpp-b10569.tar.gz.bin'); assert src, 'llama.cpp source dataset missing'
    h = hashlib.sha256(open(src, 'rb').read()).hexdigest(); log('source', src, h)
    assert h == '10c1055619b431a634f7ae82e2d79ede74fde2f2a110cb926a4b12368577d0e8'
    sh('mkdir -p /tmp/lcpp && tar -xzf ' + src + ' -C /tmp/lcpp'); root = glob.glob('/tmp/lcpp/llama.cpp-*')[0]
    if not shutil.which('cmake'): sh(f'{sys.executable} -m pip install -q cmake')
    nvcc = shutil.which('nvcc') or ('/usr/local/cuda/bin/nvcc' if os.path.exists('/usr/local/cuda/bin/nvcc') else None)
    log('nvcc', nvcc)
    tb = time.time()
    r = sh(f'cd {root} && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release '
           f'-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA_NO_VMM=ON ' + (f'-DCMAKE_CUDA_COMPILER={nvcc} ' if nvcc else '') + '2>&1 | tail -30')
    r = sh(f'cd {root} && cmake --build build -j{os.cpu_count()} --target llama-server 2>&1 | tail -25')
    SRV = f'{root}/build/bin/llama-server'; assert os.path.exists(SRV), 'build failed'
    note('build_s', round(time.time() - tb)); log('build done', RES['build_s'])
    os.makedirs(f'{OUT}/lcpp-bin', exist_ok=True); shutil.copy(SRV, f'{OUT}/lcpp-bin/llama-server')
gguf = first('*IQ2_XXS-00001-of-00002.gguf'); assert gguf, 'gguf missing'; log('gguf', gguf)
# ---- 2. server helpers ----
proc = None
def get(path, t=5):
    try:
        with urllib.request.urlopen(URL + path, timeout=t) as r: return r.status, r.read().decode()
    except Exception as e: return None, str(e)
def stop():
    global proc
    if proc and proc.poll() is None:
        proc.terminate()
        try: proc.wait(60)
        except Exception: proc.kill()
    proc = None; time.sleep(5)
def start(sm, np_, ctx, tag, ub=512):
    global proc
    base = [SRV, '-m', gguf, '-sm', sm, '-fa', 'on', '-np', str(np_), '-kvu', '-b', '2048', '-ub', str(ub), '--host', '127.0.0.1',
            '--port', str(PORT), '--jinja', '--metrics', '-t', '8', '--reasoning-format', 'deepseek']
    # ctx None = let llama.cpp --fit pick the largest context that fits (all layers on GPU first; ngl unset)
    cmd = base + (['-fitc', '131072'] if ctx is None else ['-ngl', '999', '-c', str(ctx)])
    log('START', tag, ' '.join(cmd)); lf = open(f'{OUT}/server_{tag}.log', 'w')
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT); t = time.time()
    while time.time() - t < 1800 and time.time() < END_ALL:
        if proc.poll() is not None: log('server exited', proc.returncode); sh(f'tail -40 {OUT}/server_{tag}.log'); return None
        s, _ = get('/health')
        if s == 200:
            dt = round(time.time() - t); log('READY', tag, dt, 's'); sh('nvidia-smi --query-gpu=memory.used,memory.total --format=csv')
            sh(f'grep -E -i "fit|n_ctx|offload|buffer size|n_seq|kv_unified|split|error|warn" {OUT}/server_{tag}.log | head -60'); return dt
        time.sleep(5)
    log('load timeout'); sh(f'tail -40 {OUT}/server_{tag}.log'); stop(); return None
def log_ctx(tag):
    try:
        m = re.findall(r'n_ctx\s*=\s*(\d+)', open(f'{OUT}/server_{tag}.log').read()); return int(m[-1]) if m else None
    except Exception: return None
FIT_CTX = [None]
def boot(mode):
    sm, ub = mode
    tag = f'{sm}_ub{ub}'
    dt = start(sm, 16, FIT_CTX[0], tag + '_fit' if FIT_CTX[0] is None else tag, ub)
    if dt is not None:
        if FIT_CTX[0] is None: FIT_CTX[0] = log_ctx(tag + '_fit')
        return 16, FIT_CTX[0], dt
    stop()
    for np_, ctx in ((16, 393216), (12, 262144), (8, 131072)):
        dt = start(sm, np_, ctx, f'{tag}_{np_}_{ctx}', ub)
        if dt is not None: return np_, ctx, dt
        stop()
    return None
STOP = threading.Event()
def chat(prompt, max_tokens, rec, temp=1.0, timeout=4 * 3600, budget=0):
    body = dict(model='dots3', messages=[{'role': 'user', 'content': prompt}], max_tokens=max_tokens, temperature=temp, top_p=0.95,
                stream=True, chat_template_kwargs={'enable_thinking': True})
    if budget > 0: body.update(reasoning_budget_tokens=budget, reasoning_budget_message=BUDGET_MSG)
    req = urllib.request.Request(URL + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    rs, cs = [], []; rec.update(n=0, nr=0, nc=0, t_start=time.time(), finish=None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.decode(errors='replace').strip()
                if not line.startswith('data:'): continue
                d = line[5:].strip()
                if d == '[DONE]': break
                try: j = json.loads(d)
                except Exception: continue
                if j.get('timings'): rec['timings'] = j['timings']
                if j.get('usage'): rec['usage'] = j['usage']
                for ch in j.get('choices', []):
                    dl = ch.get('delta', {}) or {}
                    if dl.get('reasoning_content'): rs.append(dl['reasoning_content']); rec['n'] += 1; rec['nr'] += 1
                    if dl.get('content'): cs.append(dl['content']); rec['n'] += 1; rec['nc'] += 1
                    if ch.get('finish_reason'): rec['finish'] = ch['finish_reason']
                if rec['n'] == 1: rec['t_first'] = time.time()
                if STOP.is_set(): rec['finish'] = rec['finish'] or 'stopped'; break
    except Exception as e: rec['error'] = str(e)[:200]
    rec['t_end'] = time.time(); return ''.join(rs), ''.join(cs)
# ---- data ----
evp = first('arc-agi_evaluation_challenges.json'); tasks = json.load(open(evp)); sols = json.load(open(evp.replace('challenges', 'solutions')))
ids = sorted(tasks); random.Random(2026).shuffle(ids)
fmt = lambda g: '\n'.join(' '.join(map(str, r)) for r in g)
def prompt(t, x):
    s = 'Solve this ARC-AGI puzzle. Each grid is rows of digits 0-9 (colors) separated by spaces.\nFind the single rule that maps every training input to its output, check it against all pairs, then apply it to the test input.\n\n'
    for i, p in enumerate(t['train']): s += f'Example {i+1} input:\n{fmt(p["input"])}\nExample {i+1} output:\n{fmt(p["output"])}\n\n'
    return s + f'Test input:\n{fmt(x["input"])}\n\nGive the final test output grid at the end inside one ``` block, rows of space-separated digits, nothing else in the block.'
def parse(txt):
    for b in reversed(re.findall(r'```[a-z]*\n(.*?)```', txt or '', re.S)):
        rows = [[int(c) for c in re.findall(r'\d', ln)] for ln in b.strip().splitlines() if ln.strip()]
        if rows and all(len(r) == len(rows[0]) for r in rows) and len(rows) <= 30 and len(rows[0]) <= 30: return rows
    return None
# ---- 3. throughput sweep ----
def sweep(tag, np_):
    small = min(ids, key=lambda k: len(prompt(tasks[k], tasks[k]['test'][0])))
    p = prompt(tasks[small], tasks[small]['test'][0]); out = []
    for c in [x for x in (1, 4, 8, 12, 16) if x <= np_]:
        if time.time() > END_ALL - 1800: break
        recs = [{} for _ in range(c)]; th = [threading.Thread(target=chat, args=(p, 512, recs[i])) for i in range(c)]
        t = time.time(); [x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t
        ntok = sum(r.get('n', 0) for r in recs)
        pred = [r.get('timings', {}).get('predicted_per_second') for r in recs]
        row = dict(c=c, wall=round(wall, 1), tokens=ntok, agg_tps=round(ntok / wall, 1), per_stream=pred[:4], err=[r.get('error') for r in recs if r.get('error')][:2])
        log('SWEEP', tag, row); out.append(row)
    sh('nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv')
    return out
modes = {}; running = None
# Dots3 (MLA) is excluded from -sm tensor in b10569 (llm_arch_supports_sm_tensor); compare one big ubatch against small
# ubatches, which lets llama.cpp pipeline micro-batches across the 4 layer-split GPUs during decode.
for mode in (('layer', 512), ('layer', 4)):
    key = f'{mode[0]}_ub{mode[1]}'
    if time.time() > END_ALL - 3600: break
    if running: stop(); running = None
    b = boot(mode)
    if not b: modes[key] = {'boot': 'failed'}; note('modes', modes); continue
    np_, ctx, dt = b; running = (key, b, mode); modes[key] = dict(np=np_, ctx=ctx, load_s=dt, sweep=sweep(key, np_)); note('modes', modes)
ok = {k: v for k, v in modes.items() if v.get('sweep')}
if not ok: log('NO MODE WORKED'); stop(); sys.exit(0)
best = max(ok, key=lambda k: max(r['agg_tps'] for r in ok[k]['sweep'])); note('best_mode', best); log('best mode', best)
# ---- 3b. operating point for the real rerun (D37, pre-registered): 259 inputs, ~10.5 h of generation, 0.8 safety ----
RERUN_INPUTS, GEN_S, SAFE, ROOM = 259, 10.5 * 3600, 0.8, 6000
_pts = sorted((r['c'], r['agg_tps']) for r in ok[best]['sweep'] if r.get('tokens', 0) > 0)
def tps_at(c):
    if not _pts: return 0.0
    if c <= _pts[0][0]: return _pts[0][1] * c / _pts[0][0]
    for (c0, t0), (c1, t1) in zip(_pts, _pts[1:]):
        if c0 <= c <= c1: return t0 + (t1 - t0) * (c - c0) / (c1 - c0)
    return _pts[-1][1]
_nctx = (modes[best].get('ctx') or FIT_CTX[0] or 131072)
OP = None; _cands = []
for B in (30000, 24000, 20000, 16000, 12000):
    npe = max(1, min(16, _nctx // (B + ROOM + 4000)))
    aff = SAFE * tps_at(npe) * GEN_S / RERUN_INPUTS
    _cands.append(dict(budget=B, workers=npe, tps=round(tps_at(npe), 1), afford_tokens=round(aff)))
    if OP is None and aff >= B + ROOM: OP = _cands[-1]
note('operating_point', dict(chosen=OP, candidates=_cands, n_ctx=_nctx)); log('OPERATING POINT', OP, _cands)
if not OP:
    note('verdict', 'KILL-T: no thinking budget >= 12000 affordable for 259 inputs in 10.5 h'); log('KILL-T'); stop(); sys.exit(0)
BUDGET = OP['budget']; MAXTOK = BUDGET + ROOM
# ---- 4. accuracy box ----
if running and running[0] == best: b = running[1]
else:
    stop(); b = boot(('layer', int(best.split('_ub')[1])))
if not b: log('reboot failed'); sys.exit(0)
np_, ctx, _ = b
jobs = [(k, i) for k in ids[:NTASK] for i in range(len(tasks[k]['test']))]
recs = []; lock = threading.Lock(); qi = [0]
def worker():
    while not STOP.is_set():
        with lock:
            if qi[0] >= len(jobs): return
            k, i = jobs[qi[0]]; qi[0] += 1
        rec = dict(task=k, test=i); recs.append(rec)
        r_, c_ = chat(prompt(tasks[k], tasks[k]['test'][i]), MAXTOK, rec, budget=BUDGET)
        g = parse(c_) or parse(r_[-4000:]) if (c_ or r_) else None
        rec.update(answered=g is not None, correct=(g == sols[k][i]), reasoning_chars=len(r_), content_chars=len(c_))
        open(f'{OUT}/p1_chains.jsonl', 'a').write(json.dumps(dict(task=k, test=i, reasoning=r_[-20000:], content=c_[-4000:], rec={x: rec[x] for x in rec if x != 'timings'})) + '\n')
        log('DONE', k, i, rec.get('finish'), rec.get('nr'), rec.get('nc'), 'correct' if rec['correct'] else ('answered' if rec['answered'] else 'no-answer'))
ths = [threading.Thread(target=worker, daemon=True) for _ in range(min(np_, OP['workers']))]
[t.start() for t in ths]
def snap():
    now = time.time()
    fin = [r for r in recs if r.get('t_end')]
    return dict(elapsed=round(now - T0), jobs=len(jobs), started=len(recs), finished=len(fin),
                correct=sum(1 for r in fin if r.get('correct')), answered=sum(1 for r in fin if r.get('answered')),
                truncated=sum(1 for r in fin if r.get('finish') == 'length'),
                budget_hit=sum(1 for r in fin if r.get('nr', 0) >= BUDGET - 50), natural_stop=sorted(r.get('nr', 0) for r in fin if r.get('nr', 0) < BUDGET - 50),
                tokens_total=sum(r.get('n', 0) for r in recs), agg_tps=round(sum(r.get('n', 0) for r in recs) / max(1, (now if any(not r.get('t_end') for r in recs) else max([r['t_end'] for r in recs] or [now])) - min([r['t_start'] for r in recs if 't_start' in r] or [now])), 1),
                fin_tokens=sorted(r.get('n', 0) for r in fin), open_tokens=sorted(r.get('n', 0) for r in recs if not r.get('t_end')))
while time.time() < END_ALL - 240 and any(t.is_alive() for t in ths):
    time.sleep(60); s = snap(); note('accuracy', s); log('SNAP', {k: v for k, v in s.items() if k not in ('fin_tokens', 'open_tokens')})
STOP.set(); time.sleep(20); note('accuracy', snap()); note('records', [{k: v for k, v in r.items() if k != 'timings'} for r in recs])
_s = snap(); _fin = [r for r in recs if r.get('t_end') and r.get('finish') in ('stop', 'length')]
note('gate', dict(finished=len(_fin), correct=sum(1 for r in _fin if r.get('correct')), rate=round(sum(1 for r in _fin if r.get('correct')) / max(1, len(_fin)), 3),
                  box_tps=_s['agg_tps'], afford_tokens_box=round(SAFE * _s['agg_tps'] * GEN_S / RERUN_INPUTS), budget=BUDGET, workers=OP['workers']))
log('GATE', RES['gate'])
stop(); log('P1 done', json.dumps({k: RES[k] for k in ('build_s', 'best_mode') if k in RES}), RES.get('accuracy', {}).get('correct'))
