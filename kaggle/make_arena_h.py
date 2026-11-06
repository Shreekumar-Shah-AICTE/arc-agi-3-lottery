# S19 (F4): A1h = A1 (arc3-arena-p50-fork, exact public artificialagencylab blocks2-p50-int3 runtime) plus a bounded
# parallel checkpoint prefetcher in the notebook. The bundle, its sha256 and every engine setting are unchanged.
# Why: A1 read the 135 GB checkpoint (NFS, 125.91 GiB > 90% of 78.5 GiB RAM, so vLLM skipped its own prefetch) as one
# stream at ~131 MB/s (1,031 s of 1,567 s setup); A2 died on a slower host (ready guard 3,000 s). The prefetcher reads
# the same files in vLLM's natural-sort order with N threads, staying at most WINDOW bytes ahead of the loader position
# parsed from the vLLM log, so the loader finds pages in cache. It can never fail the run (daemon thread, all errors
# swallowed, summary in /kaggle/working/arena-prefetch.json).
# Usage: python3 make_arena_h.py /data/arc/kaggle/arena_p50 /data/arc/kaggle/arena_p50h arc3-arena-p50h-fork
# Build only. Pushing to Kaggle needs an operator go (BOOT.md).
import json, sys
src, dst, kid = sys.argv[1:4]
nb = json.load(open(f'{src}/arena_p50.ipynb'))
PREFETCH = r'''
def _arc_prefetch(threads=8, window=10e9, max_seconds=3300):
    # S19 A1h: bounded parallel page-cache prefetch of the vLLM checkpoint. Never raises.
    import json, os, re, threading, time
    from pathlib import Path
    t0 = time.time(); stats = {'threads': threads, 'window_bytes': window, 'files': 0, 'bytes': 0, 'done_files': 0,
                               'done_bytes': 0, 'errors': 0, 'events': []}
    out = Path('/kaggle/working/arena-prefetch.json')
    def note(ev):
        stats['events'].append([round(time.time() - t0, 1), ev]); print('ARC_PREFETCH', json.dumps(ev), flush=True)
    try:
        base = next((p for p in (Path('/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1'),
                                 Path('/kaggle/input/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1')) if p.is_dir()), None)
        over = next((p for p in (Path('/kaggle/input/datasets/richardcsaky/arc-expert-int3-g16-v1'),
                                 Path('/kaggle/input/arc-expert-int3-g16-v1')) if p.is_dir()), None)
        if base is None:
            note({'skip': 'base model mount not found'}); return
        names = sorted(set(json.loads((base / 'model.safetensors.index.json').read_text())['weight_map'].values()),
                       key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)])
        files = [(over / n) if (over is not None and (over / n).is_file()) else (base / n) for n in names]
        sizes = [f.stat().st_size for f in files]
        cum = [0]
        for s in sizes: cum.append(cum[-1] + s)
        stats.update(files=len(files), bytes=cum[-1], overlay=str(over), base=str(base))
        note({'start': len(files), 'gb': round(cum[-1] / 1e9, 1)})
        logs = [Path('/tmp/arena/serving/vllm-openai-server.log'), Path('/kaggle/working/arena/serving/vllm-openai-server.log')]
        prog = re.compile(rb'(\d+)/(\d+) \[')
        state = {'pos': 0, 'next': 0}
        lock = threading.Lock()
        def loader_pos():
            for lp in logs:
                try:
                    with open(lp, 'rb') as fh:
                        fh.seek(max(0, lp.stat().st_size - 8192)); tail = fh.read()
                    m = [x for x in prog.findall(tail) if int(x[1]) == len(files)]
                    if m: return int(m[-1][0])
                except Exception:
                    pass
            return 0
        def worker():
            buf = bytearray(16 << 20); mv = memoryview(buf)
            while time.time() - t0 < max_seconds:
                with lock:
                    i = state['next']
                    if i >= len(files): return
                    if cum[i] - cum[state['pos']] > window:
                        i = None
                    else:
                        state['next'] += 1
                if i is None:
                    time.sleep(1); continue
                try:
                    with open(files[i], 'rb', buffering=0) as fh:
                        while fh.readinto(mv): pass
                    with lock:
                        stats['done_files'] += 1; stats['done_bytes'] += sizes[i]
                except Exception as exc:
                    stats['errors'] += 1
                    if stats['errors'] < 5: note({'error': repr(exc)[:160], 'file': files[i].name})
        def monitor():
            last = -1
            while time.time() - t0 < max_seconds and stats['done_files'] < len(files):
                state['pos'] = min(loader_pos(), len(files))
                el = time.time() - t0
                if int(el) // 60 != last:
                    last = int(el) // 60
                    note({'t': round(el), 'loader': state['pos'], 'prefetched': stats['done_files'],
                          'gb': round(stats['done_bytes'] / 1e9, 1), 'mb_s': round(stats['done_bytes'] / 1e6 / max(el, 1), 1)})
                    try: out.write_text(json.dumps(stats))
                    except Exception: pass
                time.sleep(2)
        pool = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
        for th in pool: th.start()
        mon = threading.Thread(target=monitor, daemon=True); mon.start()
        for th in pool: th.join()
        el = time.time() - t0
        note({'finished': stats['done_files'], 'gb': round(stats['done_bytes'] / 1e9, 1), 'seconds': round(el),
              'mb_s': round(stats['done_bytes'] / 1e6 / max(el, 1), 1)})
    except Exception as exc:
        note({'fatal': repr(exc)[:200]})
    try: out.write_text(json.dumps(stats))
    except Exception: pass

try:
    import threading as _th
    _th.Thread(target=_arc_prefetch, daemon=True).start()
except Exception as _exc:
    print('prefetch not started:', repr(_exc)[:200])
'''
c1 = ''.join(nb['cells'][1]['source'])
anchor = "command = [sys.executable"
assert c1.count(anchor) == 1
c1 = c1.replace(anchor, PREFETCH.lstrip('\n') + anchor)
nb['cells'][1]['source'] = c1.splitlines(keepends=True)
md = ''.join(nb['cells'][0]['source']).replace('# arc3-arena-p50-fork (S18 build, D51)', f'# {kid} (S19 build, A1h = A1 + prefetch)')
md += ("\nS19 change (A1h): one daemon thread pool prefetches the checkpoint files into page cache in vLLM's load order, "
       "at most 10 GB ahead of the loader (position parsed from the vLLM log). A1 loaded 135 GB as a single NFS stream "
       "(1,031 s); A2 failed its 3,000 s ready guard on a slower host. Engine, bundle and sha256 are unchanged.\n")
nb['cells'][0]['source'] = md.splitlines(keepends=True)
json.dump(nb, open(f'{dst}/arena_p50h.ipynb', 'w'), indent=1)
meta = json.load(open(f'{src}/kernel-metadata.json'))
meta.update(id=f'shreekumarshah/{kid}', title=kid, code_file='arena_p50h.ipynb')
json.dump(meta, open(f'{dst}/kernel-metadata.json', 'w'), indent=1)
print('built', dst)
