# Fake vLLM server for E2.1 dispatcher tests: /health, /metrics (KV usage grows with in-flight requests), /v1/chat/completions,
# /v1/completions (answer forcing), /stats (max in-flight seen). Usage: python3 test_fake_server.py PORT DATA_DIR
import json, sys, http.server, re, threading, time, random
port = int(sys.argv[1]); D = sys.argv[2]
ch = json.load(open(f'{D}/arc-agi_evaluation_challenges.json')); sols = json.load(open(f'{D}/arc-agi_evaluation_solutions.json'))
tc = json.load(open(f'{D}/arc-agi_test_challenges.json'))
fmt = lambda g: '\n'.join(' '.join(map(str, r)) for r in g)
idx = {}
for src in (ch, tc):
    for t, v in src.items():
        for j, x in enumerate(v['test']): idx.setdefault(fmt(x['input']), []).append((t, j, src is ch))
lock = threading.Lock(); S = {'inflight': 0, 'max': 0, 'chat': 0, 'comp': 0, 'bad_force': 0}
KV_PER = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
LAT = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
def find(p):
    for k, lst in idx.items():
        if k in p:
            for t, j, ev in lst:
                src = ch if ev else tc
                if fmt(src[t]['train'][0]['input']) in p: return t, j, ev
    return None, None, None
def grid_for(t, j, ev):
    return sols[t][j] if ev else [[(j + len(t)) % 10] * 3] * 2
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, obj, ctype='application/json'):
        o = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(o))); self.end_headers(); self.wfile.write(o)
    def do_GET(self):
        if self.path.startswith('/metrics'):
            with lock: n = S['inflight']
            m = (f'vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} {min(1.0, n * KV_PER)}\n'
                 f'vllm:num_requests_running{{engine="0",model_name="m"}} {n}.0\nvllm:num_requests_waiting{{engine="0",model_name="m"}} 0.0\n'
                 f'vllm:num_requests_waiting_by_reason{{engine="0",reason="x"}} 5.0\n'
                 f'vllm:generation_tokens_total{{engine="0",model_name="m"}} 123.0\n')
            return self.send(200, m.encode(), 'text/plain')
        if self.path.startswith('/stats'):
            with lock: return self.send(200, dict(S))
        return self.send(200, b'ok', 'text/plain')
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with lock: S['inflight'] += 1; S['max'] = max(S['max'], S['inflight'])
        try:
            time.sleep(LAT * random.uniform(0.2, 0.6))
            if self.path == '/v1/completions':
                p = b['prompt']; t, j, ev = find(p)
                with lock: S['comp'] += 1
                good = p.endswith('```\n') and '<|im_start|>assistant\n<think>\n' in p and '\n</think>\n\n' in p and p.count('<think>') == 1
                if not good:
                    with lock: S['bad_force'] += 1
                return self.send(200, {'choices': [{'text': fmt(grid_for(t, j, ev)) + '\n', 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 40, 'prompt_tokens': 900}})
            t, j, ev = find(b['messages'][0]['content']); h = (hash(t) + j + b.get('seed', 0)) % 3
            with lock: S['chat'] += 1
            if h == 0:   # truncated, no grid anywhere -> forcing path
                msg = {'content': '', 'reasoning_content': 'thinking ' * 50}; fin = 'length'
            elif h == 1:
                msg = {'content': 'Answer:\n```\n' + fmt(grid_for(t, j, ev)) + '\n```', 'reasoning_content': 'ok'}; fin = 'stop'
            else:        # natural but wrong answer
                msg = {'content': '```\n1 2\n3 4\n```', 'reasoning_content': 'hmm'}; fin = 'stop'
            return self.send(200, {'choices': [{'message': msg, 'finish_reason': fin}], 'usage': {'completion_tokens': 100, 'prompt_tokens': 50}})
        finally:
            with lock: S['inflight'] -= 1
http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
