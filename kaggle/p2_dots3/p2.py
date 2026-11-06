# P2 (draft, not pushed): Dots3-Note-Preview IQ2_XXS ARC-AGI-2 submission on L4x4 via llama.cpp b10569.
# Design: one llama-server holding the model across the 4 L4s; NP concurrent chat requests; per-request forced thinking budget
# (reasoning_budget_tokens) sized from measured aggregate throughput and time left; attempt 1 for every test input first,
# then attempt 2 with leftover capacity; submission.json rewritten after every answer. Parameters marked @P1 come from probe P1.
import os, sys, json, glob, re, time, random, subprocess, threading, hashlib, shutil, urllib.request
T0 = time.time(); IN = os.environ.get('P2_IN', '/kaggle/input'); OUT = os.environ.get('P2_OUT', '/kaggle/working')
RERUN = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'1', 'true'}
FAKE = os.environ.get('P2_FAKE_SRV', '')
CFG = dict(SM=os.environ.get('P2_SM', 'layer'),            # @P1 split mode
           NP=int(os.environ.get('P2_NP', '16')),            # @P1 concurrency
           CTX=int(os.environ.get('P2_CTX', '0')),           # 0 = llama.cpp --fit picks (layer mode only)
           GEN_END_S=float(os.environ.get('P2_GEN_END_S', str(11.35 * 3600 if RERUN else 1500))),
           B_MIN=6000, B_MAX=int(os.environ.get('P2_BMAX', '60000')), ANSWER_ROOM=6000, SAFETY=0.85,
           TPS0=float(os.environ.get('P2_TPS0', '150')),     # @P1 prior aggregate tok/s before measurement
           NTASK=10**6 if RERUN else int(os.environ.get('P2_NTASK', '6')), TEMP=1.0, TOP_P=0.95)
PORT = 8080; URL = f'http://127.0.0.1:{PORT}'
BUDGET_MSG = '\n\nI have used my thinking budget. I will stop here and write the final test output grid now.\n'
def log(*a): print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
def sh(c, timeout=None):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)
    log(f'$ {c[:160]} -> rc={r.returncode}\n' + (r.stdout + r.stderr)[-2500:]); return r
def first(pat):
    g = sorted(glob.glob(f'{IN}/**/{pat}', recursive=True)); return g[0] if g else None
# ---- submission skeleton first (valid even if everything below fails) ----
tp = first('arc-agi_test_challenges.json'); test = json.load(open(tp))
SUB = {k: [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in test[k]['test']] for k in test}
SUBLOCK = threading.Lock()
def write_sub():
    with SUBLOCK:
        tmp = f'{OUT}/submission.json.tmp'; json.dump(SUB, open(tmp, 'w')); os.replace(tmp, f'{OUT}/submission.json')
write_sub(); log('placeholder written', len(SUB), 'tasks', sum(len(v) for v in SUB.values()), 'inputs', 'RERUN', RERUN, CFG)
PROCS = []
if not RERUN and os.environ.get('P2_SAVE', 'smoke') == 'smoke':
    log('save run smoke mode: placeholder only'); sys.exit(0)
def main():
    # ---- task set: hidden test file in the rerun; a public-eval slice with solutions in commit runs ----
    if RERUN: tasks, sols = test, {}
    else:
        evp = first('arc-agi_evaluation_challenges.json'); ev = json.load(open(evp)); evs = json.load(open(evp.replace('challenges', 'solutions')))
        ids = sorted(ev); random.Random(2026).shuffle(ids); tasks = {k: ev[k] for k in ids[:CFG['NTASK']]}; sols = evs
    # ---- llama.cpp binary: prebuilt from P1 output if attached and runnable, else build from the pinned source ----
    def build():
        src = first('llama.cpp-b10569.tar.gz.bin'); assert src
        assert hashlib.sha256(open(src, 'rb').read()).hexdigest() == '10c1055619b431a634f7ae82e2d79ede74fde2f2a110cb926a4b12368577d0e8'
        sh('mkdir -p /tmp/lcpp && tar -xzf ' + src + ' -C /tmp/lcpp'); root = glob.glob('/tmp/lcpp/llama.cpp-*')[0]
        nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        sh(f'cd {root} && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release '
           f'-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_CUDA_COMPILER={nvcc} 2>&1 | tail -5')
        sh(f'cd {root} && cmake --build build -j{os.cpu_count()} --target llama-server 2>&1 | tail -5')
        return f'{root}/build/bin/llama-server'
    if FAKE: SRV = FAKE
    else:
        pre = first('lcpp-bin/llama-server'); SRV = None
        if pre:
            shutil.copy(pre, '/tmp/llama-server'); os.chmod('/tmp/llama-server', 0o755)
            if sh('/tmp/llama-server --version').returncode == 0: SRV = '/tmp/llama-server'
        if not SRV: SRV = build()
    gguf = first('*IQ2_XXS-00001-of-00002.gguf'); assert gguf
    cmd = [SRV, '-m', gguf, '-sm', CFG['SM'], '-fa', 'on', '-np', str(CFG['NP']), '-kvu', '-b', '2048', '-ub', '512', '--host', '127.0.0.1',
           '--port', str(PORT), '--jinja', '--metrics', '-t', '8', '--reasoning-format', 'deepseek']
    cmd += (['-fitc', '131072'] if CFG['CTX'] == 0 else ['-ngl', '999', '-c', str(CFG['CTX'])])
    log('START', ' '.join(cmd)); proc = subprocess.Popen(cmd, stdout=open(f'{OUT}/server.log', 'w'), stderr=subprocess.STDOUT); PROCS.append(proc)
    def get(path, t=5):
        try:
            with urllib.request.urlopen(URL + path, timeout=t) as r: return r.status, r.read().decode()
        except Exception as e: return None, str(e)
    while True:
        if proc.poll() is not None: sh(f'tail -40 {OUT}/server.log'); log('SERVER DIED; placeholder stays'); sys.exit(0)
        if get('/health')[0] == 200: break
        if time.time() - T0 > 3600: log('load timeout; placeholder stays'); sys.exit(0)
        time.sleep(5)
    NCTX = [int(m) for m in re.findall(r'n_ctx\s*=\s*(\d+)', open(f'{OUT}/server.log').read())] or [131072]
    NCTX = NCTX[-1]; NPE = max(2, min(CFG['NP'], NCTX // 30000))   # -kvu: all streams share NCTX tokens
    log('server ready', 'n_ctx', NCTX, 'workers', NPE); END = T0 + CFG['GEN_END_S'] if RERUN else time.time() + CFG['GEN_END_S']
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
    STOP = threading.Event(); TOK = {'n': 0}; TOKLOCK = threading.Lock()
    def chat(p, budget, rec):
        body = dict(model='dots3', messages=[{'role': 'user', 'content': p}], max_tokens=budget + CFG['ANSWER_ROOM'], temperature=CFG['TEMP'],
                    top_p=CFG['TOP_P'], stream=True, chat_template_kwargs={'enable_thinking': True},
                    reasoning_budget_tokens=budget, reasoning_budget_message=BUDGET_MSG)
        req = urllib.request.Request(URL + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        rs, cs = [], []; rec.update(nr=0, nc=0, t_start=time.time())
        try:
            with urllib.request.urlopen(req, timeout=6 * 3600) as r:
                for line in r:
                    line = line.decode(errors='replace').strip()
                    if not line.startswith('data:'): continue
                    d = line[5:].strip()
                    if d == '[DONE]': break
                    try: j = json.loads(d)
                    except Exception: continue
                    for ch in j.get('choices', []):
                        dl = ch.get('delta', {}) or {}
                        if dl.get('reasoning_content'): rs.append(dl['reasoning_content']); rec['nr'] += 1
                        if dl.get('content'): cs.append(dl['content']); rec['nc'] += 1
                        if ch.get('finish_reason'): rec['finish'] = ch['finish_reason']
                    with TOKLOCK: TOK['n'] += 1
                    if STOP.is_set(): rec['finish'] = 'stopped'; break
        except Exception as e: rec['error'] = str(e)[:200]
        rec['t_end'] = time.time(); return ''.join(rs), ''.join(cs)
    # ---- scheduler ----
    jobs = [(k, i, 0) for k in tasks for i in range(len(tasks[k]['test']))]
    random.Random(7).shuffle(jobs)
    second = []; LOCK = threading.Lock(); RECS = []; ANS = {}
    GEN_T0 = time.time()
    def rate():
        el = time.time() - GEN_T0
        return CFG['TPS0'] if el < 600 or TOK['n'] < 2000 else TOK['n'] / el
    def budget_for(pending):
        cap = rate() * max(0.0, END - time.time()) * CFG['SAFETY']
        b = cap / (max(1, pending) + 0.5 * NPE)   # in-flight attempts hold about half a budget each on average
        return int(max(CFG['B_MIN'], min(CFG['B_MAX'], b - CFG['ANSWER_ROOM'])))
    def next_job():
        with LOCK:
            if jobs: return jobs.pop(0)
            if second: return second.pop(0)
            return None
    def worker():
        while not STOP.is_set() and time.time() < END - 120:
            j = next_job()
            if j is None: return
            k, i, a = j
            with LOCK: pend = len(jobs) + len(second) + 1
            b = budget_for(pend)
            left = END - time.time() - 60
            b = int(min(b, max(0, left * rate() / NPE - CFG['ANSWER_ROOM'])))   # must finish before END at the per-stream rate
            P = prompt(tasks[k], tasks[k]['test'][i])
            b = int(min(b, NCTX // NPE - len(P) // 2 - CFG['ANSWER_ROOM']))   # own share of the shared cache
            if b < 1500: continue
            rec = dict(task=k, test=i, att=a, budget=b); RECS.append(rec)
            r_, c_ = chat(P, b, rec)
            g = parse(c_) or parse(r_[-3000:])
            rec['answered'] = g is not None
            if sols: rec['correct'] = g == sols[k][i]
            with LOCK:
                if g is not None:
                    ANS.setdefault((k, i), [])
                    if g not in ANS[(k, i)]: ANS[(k, i)].append(g)
                    if k in SUB:
                        got = ANS[(k, i)]
                        SUB[k][i] = {'attempt_1': got[0], 'attempt_2': got[1] if len(got) > 1 else got[0]}
                if a == 0: second.append((k, i, 1))
            write_sub()
            log('DONE', k, i, a, 'budget', b, 'nr', rec['nr'], 'nc', rec['nc'], rec.get('finish'), 'ans' if g else 'none', rec.get('correct', ''))
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(NPE)]
    for t in ths: t.start(); time.sleep(2)
    while any(t.is_alive() for t in ths) and time.time() < END:
        time.sleep(60)
        fin = [r for r in RECS if r.get('t_end')]
        log('SNAP', dict(rate=round(rate(), 1), started=len(RECS), finished=len(fin), answered=sum(r.get('answered', False) for r in fin),
                         correct=sum(1 for r in fin if r.get('correct')), pending=len(jobs), second=len(second), left_s=round(END - time.time())))
    STOP.set(); time.sleep(15); write_sub()
    json.dump([{k: v for k, v in r.items()} for r in RECS], open(f'{OUT}/p2_records.json', 'w'), default=str)
    try: proc.terminate()
    except Exception: pass
    log('P2 done', len(RECS), 'attempts', sum(1 for r in RECS if r.get('answered')), 'answered')

import traceback
try: main()
except SystemExit: pass
except BaseException: traceback.print_exc()
try: write_sub()
except Exception: traceback.print_exc()
for p_ in PROCS:
    try: p_.kill()
    except Exception: pass
log('exit 0'); sys.stdout.flush(); os._exit(0)
