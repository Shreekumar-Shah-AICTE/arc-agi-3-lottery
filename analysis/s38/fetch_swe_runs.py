"""Fetch the per-run SWE-Bench-Verified evaluation reports of Bjarnason et al. (2026), 'On Randomness in Agentic Evals'
(arXiv 2602.07150), from their public dataset huggingface.co/datasets/ASSERT-KTH/agentic-evals-artifacts (CC BY 4.0).
Keeps only the ids of resolved instances per run, in data/swe_runs/resolved_ids.json (7 Nov 2026, exploratory)."""
import json, re, time, urllib.request, concurrent.futures as cf
from pathlib import Path
API = 'https://huggingface.co/api/datasets/ASSERT-KTH/agentic-evals-artifacts/tree/main?recursive=true'
RAW = 'https://huggingface.co/datasets/ASSERT-KTH/agentic-evals-artifacts/resolve/main'
OUT = Path(__file__).parent / 'data' / 'swe_runs' / 'resolved_ids.json'
def fetch(u, k=0):
    try:
        r = urllib.request.urlopen(u, timeout=60); return r, json.load(r)
    except Exception as e:
        if k < 6 and '404' not in str(e): time.sleep(5 * (k + 1)); return fetch(u, k + 1)
        raise RuntimeError(f'{u}: {e}')
paths, u = [], API
while u:
    r, d = fetch(u); paths += [x['path'] for x in d if x['type'] == 'file']
    m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get('Link', '') or ''); u = m and m.group(1)
reports = sorted(p for p in paths if re.search(r'^[^/]+/[^/]+/[^/]+__[a-z0-9]{6,10}\.json$', p))   # SWE-bench harness reports
def one(p):
    d = fetch(f'{RAW}/{p}')[1]
    return p.rsplit('/', 1)[0], {'file': p, 'total': d['total_instances'], 'resolved_ids': sorted(d['resolved_ids'])}
with cf.ThreadPoolExecutor(4) as ex: res = dict(ex.map(one, reports))
OUT.write_text(json.dumps({'source': RAW, 'licence': 'CC BY 4.0', 'runs': res}, indent=0, sort_keys=True))
print(len(paths), 'files listed,', len(res), 'run reports ->', OUT)
