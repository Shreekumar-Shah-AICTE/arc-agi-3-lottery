# E1c: fair test of Qwen3.8-27B GPTQ-W4A16 on Kaggle L4x4 for ARC-AGI-2 direct grid answers.
# Pre-registered gate (D11): aggregate decode >= 600 tok/s AND pass@2 >= 25% on 30 seeded public-eval tasks.
import os, sys, json, glob, re, time, random, subprocess, threading, urllib.request, concurrent.futures as cf
T0 = time.time(); OUT = '/kaggle/working'
BUDGET_S = float(os.environ.get('E1C_BUDGET_S', '8400'))   # hard wall for generation phase
MAXTOK = int(os.environ.get('E1C_MAXTOK', '49152'))
NTASK = int(os.environ.get('E1C_NTASK', '30')); NATT = 2
TP = int(os.environ.get('E1C_TP', '2')); CONC = int(os.environ.get('E1C_CONC', '24'))
def log(*a): print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
def sh(c): subprocess.run(c, shell=True)
def install():
    try:
        import vllm; return  # noqa
    except Exception: pass
    whl = glob.glob('/kaggle/input/**/vllm-*.whl', recursive=True)
    if whl: wd = os.path.dirname(whl[0])
    else:
        arc = glob.glob('/kaggle/input/**/wheels.tar.gz', recursive=True)
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
         '--tensor-parallel-size', str(len(gpus.split(','))), '--max-model-len', str(MAXTOK + 8192), '--gpu-memory-utilization', '0.92',
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
def main():
    install(); env()
    cfgs = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)]
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
    log('servers up', used); T_GEN = time.time()
    comp = glob.glob('/kaggle/input/**/arc-agi_evaluation_challenges.json', recursive=True)[0]
    tasks = json.load(open(comp)); sols = json.load(open(comp.replace('challenges', 'solutions')))
    ids = sorted(tasks); random.Random(2026).shuffle(ids); ids = ids[:NTASK]
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
    jobs = [(tid, j, a) for tid in ids for j in range(len(tasks[tid]['test'])) for a in range(NATT)]
    # interleave attempts so attempt-0 of every task runs first
    jobs.sort(key=lambda z: z[2])
    res = {}; lock = threading.Lock(); tok = {'c': 0, 'p': 0}; ev = []
    stop = threading.Event()
    def call(job, k):
        tid, j, a = job; port = ports[k % len(ports)]
        body = {'model': 'm', 'messages': [{'role': 'user', 'content': prompt(tasks[tid], tasks[tid]['test'][j])}], 'max_tokens': MAXTOK,
                'temperature': 0.6 if a == 0 else 1.0, 'top_p': 0.95, 'top_k': 20, 'seed': 1000 * a + j}
        t1 = time.time(); txt = None; reas = ''; ct = 0; fin = 'err'
        try:
            rq = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
            r = json.load(urllib.request.urlopen(rq, timeout=max(60, BUDGET_S - (time.time() - T_GEN))))
            ch = r['choices'][0]; txt = ch['message'].get('content') or ''; reas = ch['message'].get('reasoning_content') or ch['message'].get('reasoning') or ''
            ct = r['usage']['completion_tokens']; fin = ch.get('finish_reason')
            with lock: tok['c'] += ct; tok['p'] += r['usage']['prompt_tokens']
        except Exception as e: fin = 'exc:' + str(e)[:80]
        g = parse(txt) or parse(reas[-4000:])
        ok = g == sols[tid][j]
        with lock:
            res[(tid, j, a)] = {'grid': g, 'ok': ok, 'ctok': ct, 'fin': fin, 'secs': round(time.time() - t1), 'port': port}
            ev.append(time.time() - T_GEN)
        log(f'{tid} t{j} a{a} fin={fin} ctok={ct} parsed={g is not None} ok={ok} secs={round(time.time()-t1)} agg_tps={tok["c"]/(time.time()-T_GEN):.0f}')
        with open(f'{OUT}/e1c_raw.jsonl', 'a') as f: f.write(json.dumps({'tid': tid, 'j': j, 'a': a, 'fin': fin, 'ctok': ct, 'ok': ok, 'txt': (txt or '')[-3000:], 'reas_tail': reas[-1500:]}) + '\n')
    def sampler():
        while not stop.is_set():
            for p in ports:
                s, m = get(f'http://127.0.0.1:{p}/metrics')
                keep = [l for l in m.splitlines() if l.startswith(('vllm:num_requests_running', 'vllm:kv_cache_usage_perc', 'vllm:spec_decode_num_accepted_tokens_total', 'vllm:spec_decode_num_draft_tokens_total', 'vllm:generation_tokens_total'))]
                with open(f'{OUT}/metrics.log', 'a') as f: f.write(f'{time.time()-T_GEN:.0f} {p} ' + ' | '.join(keep) + '\n')
            stop.wait(60)
    threading.Thread(target=sampler, daemon=True).start()
    with cf.ThreadPoolExecutor(CONC) as ex: list(ex.map(lambda kj: call(kj[1], kj[0]), enumerate(jobs)))
    stop.set(); wall = time.time() - T_GEN
    score = 0.0; per = {}
    for tid in ids:
        n = len(tasks[tid]['test']); s = sum(any(res.get((tid, j, a), {}).get('ok') for a in range(NATT)) for j in range(n)) / n
        per[tid] = s; score += s
    vals = list(res.values()); fins = {}
    for v in vals: fins[str(v['fin'])[:12]] = fins.get(str(v['fin'])[:12], 0) + 1
    rep = {'profile': used, 'tp': TP, 'conc': CONC, 'maxtok': MAXTOK, 'n_tasks': len(ids), 'n_jobs': len(jobs), 'done': len(vals),
           'pass2': score / len(ids), 'pass1_a0': sum(sum(res.get((t, j, 0), {}).get('ok', False) for j in range(len(tasks[t]['test']))) / len(tasks[t]['test']) for t in ids) / len(ids),
           'agg_tps': tok['c'] / wall, 'ctok': tok['c'], 'wall': wall, 'parsed': sum(v['grid'] is not None for v in vals), 'finish': fins,
           'gate': {'tps_ge_600': tok['c'] / wall >= 600, 'pass2_ge_25': score / len(ids) >= 0.25}, 'per_task': per}
    json.dump(rep, open(f'{OUT}/e1c_report.json', 'w'), indent=1)
    log('REPORT', json.dumps({k: v for k, v in rep.items() if k != 'per_task'}))
    sh(f'grep -h "throughput\\|Speculative\\|acceptance" {OUT}/server_*.log | tail -6')
if __name__ == '__main__':
    main()
