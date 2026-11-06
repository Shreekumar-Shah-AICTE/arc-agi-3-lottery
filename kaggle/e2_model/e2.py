# E2.1: model-first ARC-AGI-2 submission. Qwen3.8-27B GPTQ-W4A16, 2 vLLM replicas x TP2 on L4x4 (E1c config: pass@2 0.383
# on 30 public-eval tasks, default chat template = reasoning effort xhigh). Commit mode = short smoke; scoring rerun = 240 hidden
# tasks / ~259 test inputs (Kaggle data page + placeholder file), i.e. ~2x the 120 tasks assumed in D14, so throughput decides
# how many inputs get a second attempt.
# Changes vs E2 (never submitted):
#  (1) dispatcher: PER workers bound to each server (E1c routed by job index, up to 14 vs 10 running), staggered ramp, and a
#      KV-pressure admission gate (E1c spent ~20% of server-time at KV >= 97%, with stalls at 5-10 tok/s);
#  (2) attempt-1 priority: inputs whose attempt 0 gave no natural answer go first (E1c: all 3 attempt-1 gains came from them,
#      0/12 from inputs whose attempt 0 finished);
#  (3) answer forcing through /v1/completions on the model's own chat-template prefix (the chat path re-renders an empty think
#      block in front of the injected reasoning);
#  (4) deadline-aware max_tokens (late jobs end truncated and get forced instead of timing out); generation ends at T0 + 11.0 h.
import os, sys, json, glob, re, time, random, subprocess, threading, collections, urllib.request
T0 = time.time()
IN = os.environ.get('E2_IN', '/kaggle/input'); OUT = os.environ.get('E2_OUT', '/kaggle/working')
FAKE = os.environ.get('E2_FAKE', '') == '1'   # local test against fake servers on 8001/8002
RERUN = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'1', 'true'}
# Commit run (not scored): 'bench' = 60 public-eval tasks at the full 48k cap for 2.5 h (throughput, pass@1/2, forced-answer
# accuracy, full chains saved); 'smoke' = 3 tasks at a 6k cap for 15 min. The scoring rerun always uses the hidden test file.
COMMIT_MODE = os.environ.get('E2_COMMIT_MODE', 'bench')
_B = COMMIT_MODE == 'bench'
CFG = dict(
    HARD_END_S=float(os.environ.get('E2_HARD_END_S', str(11.0 * 3600))),   # rerun: generation stops at T0 + this (12 h limit, ~55 min margin)
    COMMIT_BUDGET_S=float(os.environ.get('E2_COMMIT_BUDGET_S', '9000' if _B else '900')),
    MAXTOK=49152 if (RERUN or _B) else int(os.environ.get('E2_COMMIT_MAXTOK', '6144')),
    NTASK=10**6 if RERUN else int(os.environ.get('E2_COMMIT_NTASK', '60' if _B else '3')),
    PER=int(os.environ.get('E2_PER', '12')),   # concurrent requests per server (E1c: 12 on average)
    KV_ADMIT=float(os.environ.get('E2_KV_ADMIT', '0.82')),
    RAMP_GAP=float(os.environ.get('E2_RAMP_GAP', '120' if (RERUN or _B) else '3')),   # spacing of the first PER admissions per server
    MIN_GAP=float(os.environ.get('E2_MIN_GAP', '8' if (RERUN or _B) else '0.5')),
    TPS_STREAM=18.0,                           # conservative per-request decode speed for the deadline clamp (E1c ~22)
    TEMPS=(0.6, 1.0), EFFORT=os.environ.get('E2_EFFORT', ''), SAVE_CHAINS=(not RERUN), MODE='rerun' if RERUN else COMMIT_MODE)
MAXTOK = CFG['MAXTOK']; NTASK = CFG['NTASK']; NATT = 2
MAXLEN = MAXTOK + 16384   # room for large-grid prompts (30x30 tasks can exceed 8k prompt tokens)
TP = int(os.environ.get('E1C_TP', '2'))
XHIGH = ('Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible '
         'alternatives, and prioritize correctness, consistency, and clarity in the final answer.')
LOW = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.'
def log(*a): print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
def sh(c): subprocess.run(c, shell=True)
def install():
    try:
        import vllm; return  # noqa
    except Exception: pass
    whl = glob.glob(f'{IN}/**/vllm-*.whl', recursive=True)
    if whl: wd = os.path.dirname(whl[0])
    else:
        arc = glob.glob(f'{IN}/**/wheels.tar.gz', recursive=True)
        subprocess.run(f'mkdir -p /tmp/wh && tar -xzf {arc[0]} -C /tmp/wh', shell=True, check=True)
        wd = os.path.dirname(glob.glob('/tmp/wh/**/*.whl', recursive=True)[0])
    log('pip install vllm from', wd)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', wd, 'vllm'], check=True)
def env():
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    sh('L=$(ldconfig -p | grep -m1 "libcuda.so.1 " | awk \'{print $NF}\'); [ -n "$L" ] && mkdir -p /tmp/cudalib && ln -sf $L /tmp/cudalib/libcuda.so; echo libcuda $L')
    os.environ['LIBRARY_PATH'] = '/tmp/cudalib:/usr/local/nvidia/lib64:' + os.environ.get('LIBRARY_PATH', '')
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2')
def get(url, t=5):
    try:
        with urllib.request.urlopen(url, timeout=t) as r: return r.status, r.read().decode()
    except Exception as e: return 0, str(e)
def start(mdir, gpus, port, prof):
    a = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', mdir, '--served-model-name', 'm', '--port', str(port),
         '--tensor-parallel-size', str(len(gpus.split(','))), '--max-model-len', str(MAXLEN), '--gpu-memory-utilization', '0.92',
         '--max-num-seqs', str(prof.get('seqs', 16)), '--max-num-batched-tokens', '8192', '--enable-chunked-prefill', '--no-enable-prefix-caching',
         '--trust-remote-code', '--no-enable-log-requests', '--disable-uvicorn-access-log', '--reasoning-parser', 'qwen3']
    if prof.get('lmonly'): a += ['--language-model-only']
    if prof.get('fp8kv'): a += ['--kv-cache-dtype', 'fp8']
    if prof.get('mtp'): a += ['--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': prof['mtp']})]
    if prof.get('async'): a += ['--async-scheduling']
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    f = open(f'{OUT}/server_{port}.log', 'a'); f.write(f'\n==== {prof}\n'); f.flush()
    return subprocess.Popen(a, env=e, stdout=f, stderr=subprocess.STDOUT)
def wait_up(procs, ports, limit=1500):
    t = time.time()
    while time.time() - t < limit:
        if any(p.poll() is not None for p in procs): return False
        if all(get(f'http://127.0.0.1:{p}/health')[0] == 200 for p in ports): return True
        time.sleep(10)
    return False
PROFILES = [dict(mtp=3, fp8kv=1, lmonly=1, async_=0, seqs=16), dict(mtp=0, fp8kv=1, lmonly=1, seqs=16), dict(mtp=0, fp8kv=0, lmonly=0, seqs=12)]
def post(port, path, body, timeout):
    rq = urllib.request.Request(f'http://127.0.0.1:{port}{path}', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(rq, timeout=timeout) as r: return json.load(r)
WANT = ('vllm:kv_cache_usage_perc', 'vllm:num_requests_waiting', 'vllm:num_requests_running', 'vllm:generation_tokens_total',
        'vllm:spec_decode_num_accepted_tokens_total', 'vllm:spec_decode_num_draft_tokens_total')
def main():
    def first(pat):
        g = glob.glob(f'{IN}/**/{pat}', recursive=True); return g[0] if g else None
    try:
        _t = json.load(open(first('arc-agi_test_challenges.json')))
        json.dump({t: [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in _t[t]['test']] for t in _t}, open(f'{OUT}/submission.json', 'w'))
    except Exception as e: log('early placeholder failed', e)
    log('CFG', json.dumps(CFG), 'RERUN', RERUN, 'FAKE', FAKE)
    mdir = None; TOK = None
    if FAKE:
        ports = [8001, 8002]; used = {'fake': 1}
    else:
        install(); env()
        cfgs = [os.path.dirname(p) for p in glob.glob(f'{IN}/**/config.json', recursive=True)]
        mdir = [c for c in cfgs if 'gptq' in c.lower() or 'w4a16' in c.lower()] or cfgs
        mdir = mdir[0]; log('model', mdir); sh(f'ls -la {mdir} | head -30')
        ngpu = int(subprocess.run('nvidia-smi -L | wc -l', shell=True, capture_output=True, text=True).stdout.strip() or 4)
        groups = [','.join(str(g) for g in range(i, i + TP)) for i in range(0, ngpu, TP)]
        ports = [8001 + i for i in range(len(groups))]
        used = None
        for prof in PROFILES:
            log('starting profile', prof, groups)
            procs = [start(mdir, g, p, prof) for g, p in zip(groups, ports)]
            if wait_up(procs, ports): used = prof; break
            log('profile failed'); [p.kill() for p in procs]; time.sleep(15)
            for p in ports: sh(f'tail -25 {OUT}/server_{p}.log')
        if not used: log('ALL PROFILES FAILED'); return
        try:
            from transformers import AutoTokenizer
            TOK = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
        except Exception as e: log('tokenizer load failed, using built-in template', str(e)[:120])
    log('servers up', used); T_GEN = time.time()
    END = T0 + CFG['HARD_END_S'] if RERUN else T_GEN + CFG['COMMIT_BUDGET_S']
    left = lambda: END - time.time()
    tp = first('arc-agi_test_challenges.json'); evp = first('arc-agi_evaluation_challenges.json')
    comp = tp if (RERUN or not evp) else evp
    tasks = json.load(open(comp)); sols = {} if comp == tp else json.load(open(evp.replace('challenges', 'solutions')))
    ids = sorted(tasks); random.Random(2026).shuffle(ids); ids = ids[:NTASK]
    sub_tasks = json.load(open(tp))   # submission must cover the test file's tasks
    sub = {t: [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in sub_tasks[t]['test']] for t in sub_tasks}
    lock = threading.Lock()
    def write_sub():
        tmp = f'{OUT}/submission.json.tmp'; json.dump(sub, open(tmp, 'w')); os.replace(tmp, f'{OUT}/submission.json')
    write_sub(); log('placeholder submission written', len(sub), 'tasks; solving', len(ids), 'tasks from', comp)
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
    def gen_prefix(ptxt):
        # exact generation prefix of the served model (system effort line + user turn + '<think>\n')
        if TOK is not None:
            try:
                kw = {'reasoning_effort': CFG['EFFORT']} if CFG['EFFORT'] else {}
                s = TOK.apply_chat_template([{'role': 'user', 'content': ptxt}], tokenize=False, add_generation_prompt=True, **kw)
                return s if s.endswith('<think>\n') else s + '<think>\n'
            except Exception as e: log('apply_chat_template failed', str(e)[:120])
        sysl = {'': XHIGH, 'xhigh': XHIGH, 'low': LOW, 'medium': ''}[CFG['EFFORT']]
        return (('<|im_start|>system\n' + sysl + '<|im_end|>\n') if sysl else '') + '<|im_start|>user\n' + ptxt + '<|im_end|>\n<|im_start|>assistant\n<think>\n'
    if ids:
        log('gen prefix head/tail:', repr(gen_prefix('PROMPT')[:200]), '...', repr(gen_prefix('PROMPT')[-40:]))
    # ---- job queues: all attempt-0 first; attempt-1 for inputs without a natural answer next; the rest last
    q0 = collections.deque((tid, j, 0) for tid in ids for j in range(len(tasks[tid]['test'])))
    qhi, qlo = collections.deque(), collections.deque(); live0 = set()
    n_inputs = len(q0)
    res = {}; tok = {'c': 0, 'p': 0}; stop = threading.Event()
    st = {p: dict(kv=0.0, wait=0.0, run=0.0, t=0.0, adm=0, last=0.0, kvmax=0.0, hi=0, polls=0) for p in ports}
    def next_job():
        with lock:
            if left() < 300: return None
            if q0: z = q0.popleft(); live0.add(z[:2]); return z
            if qhi: return qhi.popleft()
            if live0: return 'wait'
            if qlo: return qlo.popleft()
            return None
    def admit(port):
        while True:
            if left() < 300: return False
            now = time.time()
            with lock:
                s = st[port]; gap = CFG['RAMP_GAP'] if s['adm'] < CFG['PER'] else CFG['MIN_GAP']
                fresh = now - s['t'] < 20
                if now - s['last'] >= gap and (not fresh or (s['kv'] < CFG['KV_ADMIT'] and s['wait'] < 1)):
                    s['adm'] += 1; s['last'] = now; return True
            time.sleep(1)
    def call(job, port, wait_s):
        tid, j, a = job
        if left() < 300: return None
        ptxt = prompt(tasks[tid], tasks[tid]['test'][j])
        mt = max(2048, min(MAXTOK, MAXLEN - int(len(ptxt) / 1.6) - 512))   # never request past the context limit
        mt = max(1024, min(mt, int((left() - 240) * CFG['TPS_STREAM'])))     # finish before the deadline
        body = {'model': 'm', 'messages': [{'role': 'user', 'content': ptxt}], 'max_tokens': mt,
                'temperature': CFG['TEMPS'][a], 'top_p': 0.95, 'top_k': 20, 'seed': 1000 * a + j}
        if CFG['EFFORT']: body['chat_template_kwargs'] = {'reasoning_effort': CFG['EFFORT']}
        t1 = time.time(); txt = None; reas = ''; ct = 0; fin = 'err'
        try:
            r = post(port, '/v1/chat/completions', body, max(60, left() - 30))
            ch = r['choices'][0]; txt = ch['message'].get('content') or ''; reas = ch['message'].get('reasoning_content') or ch['message'].get('reasoning') or ''
            ct = r['usage']['completion_tokens']; fin = ch.get('finish_reason')
            with lock: tok['c'] += ct; tok['p'] += r['usage']['prompt_tokens']
        except Exception as e: fin = 'exc:' + str(e)[:80]
        g_nat = parse(txt)
        g = g_nat or parse(reas[-4000:])
        forced = False
        if g is None and fin == 'length' and reas and left() > 120:
            # answer forcing: close the thinking block on the served template and ask only for the grid (~10-18k prefill, <=3k decode)
            fp = gen_prefix(ptxt) + reas[-24000:].strip() + '\n</think>\n\nTime is up. Final test output grid:\n```\n'
            fb = {'model': 'm', 'prompt': fp, 'max_tokens': 3072, 'temperature': 0.2, 'top_p': 0.95, 'stop': ['```'], 'seed': 7 + j}
            try:
                r2 = post(port, '/v1/completions', fb, max(60, min(900, left() - 30)))
                t2 = r2['choices'][0].get('text') or ''
                g = parse('```\n' + t2.strip('\n') + '\n```'); forced = g is not None
                with lock: tok['c'] += r2['usage']['completion_tokens']
            except Exception as e: log('force failed', str(e)[:120])
        ok = (g == sols[tid][j]) if sols else None
        with lock:
            if g is not None and tid in sub and j < len(sub[tid]):
                slot = 'attempt_1' if a == 0 else 'attempt_2'
                sub[tid][j][slot] = g
                other = 'attempt_2' if a == 0 else 'attempt_1'
                if sub[tid][j][other] == [[0]]: sub[tid][j][other] = g   # never leave an empty attempt
                write_sub()
            res[(tid, j, a)] = {'grid': g, 'ok': ok, 'ctok': ct, 'fin': fin, 'forced': forced, 'natural': g_nat is not None,
                                'secs': round(time.time() - t1), 'port': port, 'wait': round(wait_s)}
            if a == 0:
                (qlo if (fin == 'stop' and g_nat is not None) else qhi).append((tid, j, 1))
        log(f'{tid} t{j} a{a} p{port} fin={fin} forced={forced} ctok={ct} parsed={g is not None} ok={ok} secs={round(time.time()-t1)} '
            f'wait={round(wait_s)} agg_tps={tok["c"]/max(1, time.time()-T_GEN):.0f} q0={len(q0)} qhi={len(qhi)} qlo={len(qlo)}')
        with open(f'{OUT}/e2_raw.jsonl', 'a') as f:
            f.write(json.dumps({'tid': tid, 'j': j, 'a': a, 'fin': fin, 'forced': forced, 'natural': g_nat is not None, 'ctok': ct, 'ok': ok,
                                'mt': mt, 'port': port, 't_start': round(t1 - T_GEN), 'secs': round(time.time() - t1),
                                'txt': (txt or '')[-3000:], 'reas_tail': reas[-6000:]}) + '\n')
        if CFG['SAVE_CHAINS']:
            import gzip
            with lock, gzip.open(f'{OUT}/chains.jsonl.gz', 'at') as f:
                f.write(json.dumps({'tid': tid, 'j': j, 'a': a, 'fin': fin, 'ctok': ct, 'ok': ok, 'reasoning': reas, 'content': txt or ''}) + '\n')
        return True
    def worker(port):
        while True:
            z = next_job()
            if z is None: return
            if z == 'wait': time.sleep(3); continue
            tw = time.time()
            try:
                if not admit(port): return
                call(z, port, time.time() - tw)
            except Exception as e: log('worker error', repr(e)[:160])
            finally:
                with lock: live0.discard(z[:2]) if z[2] == 0 else None
    def poller():
        last_log = 0.0
        while not stop.is_set():
            line = []
            for p in ports:
                s, m = get(f'http://127.0.0.1:{p}/metrics')
                if s != 200: continue
                d = {}
                for l in m.splitlines():
                    if not l.startswith('vllm:'): continue
                    name = l.split('{', 1)[0].split(' ', 1)[0]
                    if name in WANT:
                        try: d[name] = d.get(name, 0.0) + float(l.rsplit(' ', 1)[1])
                        except Exception: pass
                with lock:
                    x = st[p]; x['t'] = time.time(); x['polls'] += 1
                    x['kv'] = d.get('vllm:kv_cache_usage_perc', x['kv']); x['wait'] = d.get('vllm:num_requests_waiting', 0.0)
                    x['run'] = d.get('vllm:num_requests_running', 0.0); x['kvmax'] = max(x['kvmax'], x['kv']); x['hi'] += x['kv'] >= 0.97
                line.append(f'{p} kv={x["kv"]:.3f} run={x["run"]:.0f} wait={x["wait"]:.0f} gen={d.get("vllm:generation_tokens_total", 0):.0f} '
                            f'acc={d.get("vllm:spec_decode_num_accepted_tokens_total", 0):.0f} drf={d.get("vllm:spec_decode_num_draft_tokens_total", 0):.0f}')
            if time.time() - last_log >= 60:
                last_log = time.time()
                with open(f'{OUT}/metrics.log', 'a') as f: f.write(f'{time.time()-T_GEN:.0f} ' + ' | '.join(line) + '\n')
            stop.wait(4)
    threading.Thread(target=poller, daemon=True).start()
    ths = [threading.Thread(target=worker, args=(p,), daemon=True) for p in ports for _ in range(CFG['PER'])]
    [t.start() for t in ths]
    while any(t.is_alive() for t in ths) and left() > -60: time.sleep(5)
    stop.set(); wall = time.time() - T_GEN
    with lock: write_sub()
    score = 0.0; per = {}
    for tid in (ids if sols else []):
        n = len(tasks[tid]['test']); s = sum(any((res.get((tid, j, a)) or {}).get('ok') for a in range(NATT)) for j in range(n)) / n
        per[tid] = s; score += s
    p1 = sum(sum(bool((res.get((tid, j, 0)) or {}).get('ok')) for j in range(len(tasks[tid]['test']))) / len(tasks[tid]['test']) for tid in ids) if sols else None
    vals = list(res.values()); fins = collections.Counter(str(v['fin'])[:12] for v in vals)
    rep = {'cfg': CFG, 'profile': used, 'tp': TP, 'maxtok': MAXTOK, 'n_tasks': len(ids), 'n_inputs': n_inputs, 'done': len(vals),
           'done_a0': sum(1 for k in res if k[2] == 0), 'done_a1': sum(1 for k in res if k[2] == 1),
           'pass2': score / len(ids) if sols and ids else None, 'pass1': p1 / len(ids) if sols and ids else None,
           'forced_parsed': sum(bool(v.get('forced')) for v in vals), 'forced_ok': sum(bool(v.get('forced') and v.get('ok')) for v in vals),
           'agg_tps': tok['c'] / wall, 'ctok': tok['c'], 'wall': wall, 'parsed': sum(v['grid'] is not None for v in vals), 'finish': dict(fins),
           'servers': {p: {k: st[p][k] for k in ('adm', 'kvmax', 'hi', 'polls')} for p in ports}, 'per_task': per}
    json.dump(rep, open(f'{OUT}/e2_report.json', 'w'), indent=1)
    log('REPORT', json.dumps({k: v for k, v in rep.items() if k != 'per_task'}))
    if not FAKE: sh(f'grep -h "throughput\\|Speculative\\|acceptance" {OUT}/server_*.log | tail -6')
if __name__ == '__main__':
    main()
