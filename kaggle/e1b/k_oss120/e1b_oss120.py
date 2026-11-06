# E1b: can an open LLM on Kaggle L4x4 do train-verified program synthesis on ARC-AGI-2 eval?
import os, sys, json, glob, re, time, subprocess, tempfile, concurrent.futures as cf
T0 = time.time()
MODEL_HINT = os.environ.get('E1B_MODEL', 'gpt-oss-120b')      # substring of model dir
K = int(os.environ.get('E1B_K', '4'))                    # samples per task in round 1
MAXTOK = int(os.environ.get('E1B_MAXTOK', '24576'))
BUDGET_S = float(os.environ.get('E1B_BUDGET_S', '9000'))
REPAIR = int(os.environ.get('E1B_REPAIR', '1'))
OUT = '/kaggle/working'
def log(*a): print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
def main():
    try:
        import vllm  # noqa
    except Exception:
        arc = glob.glob('/kaggle/input/**/wheels.tar.gz', recursive=True)
        log('installing vllm from', arc)
        subprocess.run(f'mkdir -p /tmp/wh && tar -xzf {arc[0]} -C /tmp/wh', shell=True, check=True)
        wd = os.path.dirname(glob.glob('/tmp/wh/**/*.whl', recursive=True)[0])
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', wd, 'vllm', 'openai_harmony'], check=True)
    subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2', shell=True)
    cfgs = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)]
    mdir = [c for c in cfgs if MODEL_HINT.lower() in c.lower().replace('_', '-')]
    log('model dirs', mdir or cfgs); mdir = (mdir or cfgs)[0]
    from vllm import LLM, SamplingParams
    import vllm as _v; log('vllm', _v.__version__)
    llm = LLM(mdir, tensor_parallel_size=4, max_model_len=MAXTOK + 8192, gpu_memory_utilization=0.92, enable_prefix_caching=True, max_num_seqs=64, trust_remote_code=True)
    log('model loaded')
    comp = glob.glob('/kaggle/input/**/arc-agi_evaluation_challenges.json', recursive=True)[0]
    tasks = json.load(open(comp)); sols = json.load(open(comp.replace('challenges', 'solutions')))
    fmt = lambda g: '\n'.join(''.join(map(str, r)) for r in g)
    PROMPT = """You are solving an ARC-AGI-2 puzzle. Grids are given as lines of digits 0-9 (one line per row, one digit per cell).
    Infer the single general transformation rule that maps every train input to its output. Reason carefully about objects, colors, symmetry, counting and geometry.
    Then return ONLY one python code block defining `def transform(grid: list[list[int]]) -> list[list[int]]`. numpy is allowed. It must work for all train pairs and the test inputs.
    {pairs}
    {tests}"""
    def prompt(t):
        pairs = '\n'.join(f'Train {i+1} input:\n{fmt(p["input"])}\nTrain {i+1} output:\n{fmt(p["output"])}\n' for i, p in enumerate(t['train']))
        tests = '\n'.join(f'Test {j+1} input:\n{fmt(x["input"])}\n' for j, x in enumerate(t['test']))
        return PROMPT.format(pairs=pairs, tests=tests)
    RUNNER = """import json,sys
    src=open(sys.argv[1]).read(); data=json.load(open(sys.argv[2])); ns={}; out=[]
    try: exec(src, ns)
    except Exception: print(json.dumps([None]*len(data))); sys.exit()
    for g in data:
        try: r=ns['transform']([row[:] for row in g]); out.append([[int(v) for v in row] for row in r])
        except Exception: out.append(None)
    print(json.dumps(out))"""
    def run_prog(code, grids, timeout=15):
        with tempfile.TemporaryDirectory() as d:
            open(f'{d}/p.py', 'w').write(code); json.dump(grids, open(f'{d}/g.json', 'w')); open(f'{d}/r.py', 'w').write(RUNNER)
            try:
                p = subprocess.run([sys.executable, f'{d}/r.py', f'{d}/p.py', f'{d}/g.json'], capture_output=True, text=True, timeout=timeout)
                return json.loads(p.stdout.strip().splitlines()[-1])
            except Exception: return [None] * len(grids)
    def extract(txt):
        m = re.findall(r'```(?:python)?\n(.*?)```', txt or '', re.S); return m[-1] if m else None
    def evaluate(tid, code):
        t = tasks[tid]; n = len(t['train'])
        res = run_prog(code, [p['input'] for p in t['train']] + [x['input'] for x in t['test']])
        ok = [res[i] == t['train'][i]['output'] for i in range(n)]
        return {'code': code, 'train_ok': sum(ok), 'n_train': n, 'test_pred': res[n:], 'train_pred': res[:n]}
    tok = llm.get_tokenizer()
    def chat_text(msgs):
        kw = {'reasoning_effort': 'high'} if 'gpt-oss' in mdir.lower() or 'gpt_oss' in mdir.lower() else {}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    ids = sorted(tasks)
    base_msgs = {tid: [{'role': 'user', 'content': prompt(tasks[tid])}] for tid in ids}
    ids = [tid for tid in ids if len(tok.encode(base_msgs[tid][0]['content'])) < 7500]
    log('tasks usable', len(ids), 'of', len(tasks))
    cands = {tid: [] for tid in ids}; stats = {'rounds': []}
    def gen_round(reqs, n, label):
        t1 = time.time(); sp = SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=MAXTOK)
        outs = []; CH = 40
        for i in range(0, len(reqs), CH):
            if time.time() - T0 > BUDGET_S: log('budget hit, stopping at', i); reqs = reqs[:i]; break
            outs += llm.generate([chat_text(m) for _, m in reqs[i:i + CH]], sp); log(label, 'chunk', i, 'done')
        ntok = sum(len(o.token_ids) for r in outs for o in r.outputs)
        log(label, 'gen done', len(reqs), 'reqs', ntok, 'tokens', round(ntok / (time.time() - t1)), 'tok/s')
        jobs = [(tid, extract(o.text)) for (tid, _), r in zip(reqs, outs) for o in r.outputs]
        with cf.ThreadPoolExecutor(max(4, os.cpu_count())) as ex:
            evs = list(ex.map(lambda j: (j[0], evaluate(*j) if j[1] else None), jobs))
        for tid, ev in evs:
            if ev: cands[tid].append(ev)
        stats['rounds'].append({'label': label, 'reqs': len(reqs), 'tokens': ntok, 'secs': time.time() - t1, 'no_code': sum(1 for _, c in jobs if not c)})
    def report(label):
        solved = ver = 0; per = {}
        for tid in ids:
            good = [c for c in cands[tid] if c['train_ok'] == c['n_train']]
            att = []
            for c in sorted(good, key=lambda c: len(c['code'])):
                if c['test_pred'] not in att: att.append(c['test_pred'])
            sc = 0.0
            for j, sol in enumerate(sols[tid]):
                if any(a and j < len(a) and a[j] == sol for a in att[:2]): sc += 1 / len(sols[tid])
            per[tid] = {'n': len(cands[tid]), 'verified': len(good), 'score': sc, 'best_train': max([c['train_ok'] / c['n_train'] for c in cands[tid]] or [0])}
            solved += sc; ver += bool(good)
        fp = sum(1 for p in per.values() if p['verified'] and p['score'] == 0)
        log(f'{label}: tasks_with_verified={ver} solved={solved:.2f}/{len(tasks)} ({100*solved/len(tasks):.2f}%) verified_but_wrong={fp}')
        json.dump({'per_task': per, 'stats': stats, 'model': mdir, 'K': K, 'MAXTOK': MAXTOK, 'label': label, 'solved': solved, 'n_tasks': len(tasks)}, open(f'{OUT}/e1b_report.json', 'w'))
        json.dump({tid: [{k: c[k] for k in ('code', 'train_ok', 'n_train', 'test_pred')} for c in cands[tid]] for tid in ids}, open(f'{OUT}/e1b_cands.json', 'w'))
    gen_round([(tid, base_msgs[tid]) for tid in ids], K, 'round1'); report('round1')
    if REPAIR and time.time() - T0 < BUDGET_S * 0.55:
        reqs = []
        for tid in ids:
            if any(c['train_ok'] == c['n_train'] for c in cands[tid]) or not cands[tid]: continue
            best = max(cands[tid], key=lambda c: c['train_ok']); t = tasks[tid]
            fb = []
            for i, p in enumerate(t['train']):
                got = best['train_pred'][i]
                if got != p['output']: fb.append(f'Train {i+1}: expected\n{fmt(p["output"])}\nbut your code produced\n' + (fmt(got) if got else 'an exception or invalid output'))
            msg = base_msgs[tid] + [{'role': 'assistant', 'content': '```python\n' + best['code'] + '```'},
                   {'role': 'user', 'content': 'This program is wrong on some train pairs:\n' + '\n'.join(fb[:2])[:6000] + '\nRethink the rule and return a corrected full program in one python code block.'}]
            if len(tok.encode(chat_text(msg))) < 8000: reqs.append((tid, msg))
        gen_round(reqs, 2, 'repair'); report('repair')
    log('DONE total secs', round(time.time() - T0))

if __name__ == '__main__':
    main()
