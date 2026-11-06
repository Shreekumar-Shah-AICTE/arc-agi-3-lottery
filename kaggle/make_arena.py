# S18 (D51): exact forks of the public artificialagencylab.com (Richard Csaky, Charles Chevalier) arena runtime,
# blocks2 family (30 Sep). Nothing in their runtime is changed; the notebook only points at one bundle, checks its
# sha256 and prints mount diagnostics before the bootstrap. Usage:
#   python3 make_arena.py p50 /data/arc/kaggle/arena_p50 arc3-arena-p50-fork   (primary: 76 streams, 50% expert prune)
#   python3 make_arena.py p37 /data/arc/kaggle/arena_p37 arc3-arena-p37-fork   (alternate: 60 streams, 37.5% prune)
# Build only. Pushing to Kaggle needs an operator go (BOOT.md).
import json, os, sys
arm, d, kid = sys.argv[1:4]
V = {
    'p50': dict(ds='arc-arena-hero-v2-blocks2-p50-int3-runtime', blob='arena-flash-competition-blocks2-p50-int3-v1.tar.gz.blob',
                sha='f76187cea700749ab7572b952a9ae5e0038dfbec83619bf706e720cc9c9db286', streams=76, prune='50%', blocks='76 / 34',
                refill=42),
    'p37': dict(ds='arc-arena-hero-v2-blocks2-p37-int3-runtime', blob='arena-flash-competition-blocks2-p37-int3-v1.tar.gz.blob',
                sha='7078af80e891085280966c43ee45884dc883fabc56b1bd6b72fa1a023191861a', streams=60, prune='37.5%', blocks='60 / 50',
                refill=10),
}[arm]
md = f"""# {kid} (S18 build, D51)
Exact fork of the public **artificialagencylab.com** runtime `richardcsaky/{V['ds']}` (uploaded 30 Sep 2026;
team rank 7, best public LB 38.62 on 8 Oct, 37.94 by 5 Oct). Their code is not modified. This notebook only
selects the bundle, checks its sha256 and prints mount diagnostics.

What it runs (from the bundle's `configs/comparison.json`):
- Qwen3.8-Flash-Next NVFP4 base (RadixArk via keithtyser vLLM runtime), routed experts replaced by an ARC-calibrated
  int3 g16 overlay (`richardcsaky/arc-expert-int3-g16-v1`, manifest sha256 7fca7969...), {V['prune']} of routed experts pruned per layer
- int4 KV cache, MTP off, {V['streams']} concurrent game streams, 98,304-token window, completion cap 16,384, temperature 0.7
- 2-block schedule: 30,500 s play wall as 2 blocks of 15,250 s, roster split {V['blocks']}; a level-0 game retires at 90K
  completion tokens, each cleared level adds 90K; freed slots go to never-run games, then to parked games with the most levels
  ({V['refill']} spare refill slots in block 2)
- Duck-transfer harness with harness_fixes (exact compaction trigger, summary guard, python inventory, cap follow-up,
  stuck notice, sandbox, timeout interrupt, prompt text, summary headroom, inventory imports) and `run_plan` multi-step actions

Save run (not a rerun): setup plus a 300 s public preview on 110 cyclic clones. Envelope 5,400 s instead of their
4,200 s: the preview needs setup <= envelope - 600 s cleanup - 300 s, and their setup guard allows up to 3,600 s, so
4,200 s could refuse the preview on a slow mount. The rerun ignores this value (bootstrap_arena.py sets the envelope to
submission_seconds when KAGGLE_IS_COMPETITION_RERUN is set), so the scored run is unchanged. Expected cost ~0.5 GPU-h
(two public runb save runs measured setup at 1,164 and 1,245 s on this image and GPU).
Competition rerun: 32,400 s envelope from notebook start, 600 s cleanup reserve, gateway roster.
Pinned image is the one their public notebook used (Python 3.12).
"""
cell1 = f"""from pathlib import Path
import hashlib, json, os, subprocess, sys, tarfile, time
assert sys.version_info[:2] == (3, 12), 'Select the Python 3.12 notebook image'
NOTEBOOK_START_EPOCH = time.time()
try:  # S18 diagnostics only (no behaviour change; can never fail the run): GPU, mounts, rerun flag
    print(json.dumps({{'python': sys.version.split()[0], 'rerun': os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '')}}))
    for root in (Path('/kaggle/input'), Path('/kaggle/input/datasets'), Path('/kaggle/input/models')):
        if root.is_dir():
            print(root, sorted(p.name for p in root.iterdir())[:40])
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], check=False, timeout=60)
except Exception as exc:
    print('diagnostics skipped:', repr(exc)[:200])
roots = [Path('/kaggle/input/datasets/richardcsaky/{V['ds']}'), Path('/kaggle/input/{V['ds']}')]
mount = next((p for p in roots if p.is_dir()), None)
assert mount is not None, 'Attach the prepared richardcsaky/{V['ds']} dataset'
blob = mount / '{V['blob']}'
with blob.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == '{V['sha']}', 'Runtime archive differs from reviewed build'
bundle = Path('/tmp/arena-runtime')
bundle.mkdir(exist_ok=False)
with tarfile.open(blob, 'r:gz') as archive:
    archive.extractall(bundle, filter='data')
command = [sys.executable, str(bundle / 'bootstrap_arena.py'), '--bundle', str(bundle), '--working', '/tmp/arena',
           '--output', '/kaggle/working/arena', '--max-seconds', '5400', '--notebook-start-epoch', str(NOTEBOOK_START_EPOCH)]
result = subprocess.run(command, check=False)
print(json.dumps({{'bootstrap_exit': result.returncode, 'elapsed': time.time() - NOTEBOOK_START_EPOCH}}))
"""
cell2 = """import json, os
from pathlib import Path
EXPECTED_PREVIEW_TRIALS = 1
rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1', 'true')
report_path = Path('/kaggle/working/arena/trials/study.json')
if not report_path.is_file():
    raise RuntimeError(f'Arena stopped before final report: bootstrap_exit={result.returncode}')
record = json.loads(report_path.read_text())
acceptable = {'complete', 'complete_with_preemptions', 'error'} if rerun else {'complete'}
trials = record.get('trials', [])
expected = 1 if rerun else EXPECTED_PREVIEW_TRIALS
# On the hidden rerun the gateway scorecard is already closed: only a missing or unacceptable
# gateway trial fails the notebook, never the bootstrap exit code or the preview-only checks.
if len(trials) != expected or any(t.get('status') not in acceptable for t in trials) or (result.returncode and not rerun):
    raise RuntimeError('Arena validation failed; inspect saved runtime and trial reports')
for row in record.get('comparison', []):
    print(json.dumps(row))
if not rerun:
    import pandas as pd
    pd.DataFrame([['1_0', '1', True, 1]],
                 columns=['row_id', 'game_id', 'end_of_game', 'score']).to_parquet(
                     '/kaggle/working/submission.parquet', index=False)
    print('Public validation completed. The placeholder selects the notebook for a hidden rerun.')
else:
    print('Hidden gateway scorecard closed; the competition gateway produces the submission artifact.')
"""
def cell(t, s):
    c = {'cell_type': t, 'metadata': {}, 'source': s.splitlines(True)}
    if t == 'code': c.update(execution_count=None, outputs=[])
    return c
nb = {'cells': [cell('markdown', md), cell('code', cell1), cell('code', cell2)],
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.12'}},
      'nbformat': 4, 'nbformat_minor': 4}
os.makedirs(d, exist_ok=True)
code = f'{os.path.basename(d)}.ipynb'
json.dump(nb, open(os.path.join(d, code), 'w'), indent=1)
meta = {'id': f'shreekumarshah/{kid}', 'title': kid, 'code_file': code, 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False, 'keywords': ['gpu'],
        'dataset_sources': [f"richardcsaky/{V['ds']}", 'richardcsaky/arc-expert-int3-g16-v1',
                            'keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1', 'keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1'],
        'kernel_sources': [], 'competition_sources': ['arc-prize-2026-arc-agi-3'],
        'model_sources': ['keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1'],
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
        'machine_shape': 'NvidiaRtxPro6000'}
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
print('built', d, code)
