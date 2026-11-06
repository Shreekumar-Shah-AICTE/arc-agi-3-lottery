#!/usr/bin/env python3
# Local dry-run stand-in for llama-server (P1c). Streams SSE like llama.cpp: reasoning_content deltas, then content,
# honours reasoning_budget_tokens and max_tokens, answers correctly ~50% of the time using the local solutions file.
import sys, os, json, time, random, glob, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
a = sys.argv; port = int(a[a.index('--port') + 1]); npar = int(a[a.index('-np') + 1]) if '-np' in a else 1
ctx = int(a[a.index('-c') + 1]) if '-c' in a else 393216
RATE = float(os.environ.get('FAKE_TPS', '2000'))     # tokens/s per stream
IN = os.environ.get('P1_IN', '/kaggle/input')
ev = glob.glob(f'{IN}/**/arc-agi_evaluation_challenges.json', recursive=True)[0]
T = json.load(open(ev)); S = json.load(open(ev.replace('challenges', 'solutions')))
fmt = lambda g: '\n'.join(' '.join(map(str, r)) for r in g)
KEY = {fmt(x['input']): (k, i) for k in T for i, x in enumerate(T[k]['test'])}
print(f'llama_context: n_ctx = {ctx}', flush=True); print(f'n_seq_max = {npar}', flush=True)
time.sleep(3)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        p = b['messages'][0]['content']; mx = b.get('max_tokens', 512); bud = b.get('reasoning_budget_tokens', 0)
        ti = p.split('Test input:\n')[-1].split('\n\nGive the final')[0] if 'Test input:' in p else ''
        k = KEY.get(ti); ans = S[k[0]][k[1]] if k and random.random() < 0.5 else [[1, 2], [3, 4]]
        nat = random.randint(200, 2 * bud) if bud else mx
        nr = min(nat, bud) if bud else mx; ctoks = ['```\n'] + [fmt(ans) + '\n'] + ['```']
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        t0 = time.time(); n = 0
        def send(d, fin=None, extra=None):
            j = {'choices': [{'delta': d, 'finish_reason': fin}]}
            if extra: j.update(extra)
            self.wfile.write(('data: ' + json.dumps(j) + '\n\n').encode())
        try:
            for _ in range(nr):
                if n >= mx: break
                send({'reasoning_content': 'x'}); n += 1
                if n % 200 == 0: self.wfile.flush(); time.sleep(max(0, t0 + n / RATE - time.time()))
            if bud and nat > bud and n < mx: send({'reasoning_content': '[budget msg]'}); n += 1
            for c in ctoks:
                if n >= mx: break
                send({'content': c}); n += 1
            dt = max(1e-3, time.time() - t0)
            send({}, 'length' if n >= mx else 'stop', {'timings': {'predicted_n': n, 'predicted_per_second': n / dt}})
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
        except BrokenPipeError: pass
ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
