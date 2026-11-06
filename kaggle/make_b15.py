# B15 = B12-D25 with one change: the LHS vLLM runtime (vLLM main e975732 image runtime, primitive-ai mixed
# NVFP4-FP8 checkpoint with its own MTP head) replaces SGLang, at 14 streams. Harness, prompts, sampling and
# every other setting stay B12. Same save-run probe as B12-D25 (25 public games, 40 min each). Gate D41.
# usage: python3 make_b15.py /data/arc/kaggle/b15_vllm arc3-b15-vllm-d25 [base]
# base = b12d25 (default: the dev probe) or b12_x6d (B12's submission notebook, which differs from b12d25 only in cells 0 and 17)
import hashlib, json, os, sys
d, kid = sys.argv[1:3]
base = sys.argv[3] if len(sys.argv) > 3 else 'b12d25'
assert base in ('b12d25', 'b12_x6d'), base
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)

# Old serving constants are read only by cells 7 and 13, which are both replaced.
for k, c in enumerate(n['cells']):
    if k not in (5, 7, 13) and c['cell_type'] == 'code':
        for name in ('WHEELHOUSE_DIR', 'MODEL_DIR', 'DRAFT_MODEL_DIR', 'precache_model_thread'):
            assert name not in src(k), (k, name)

s = src(5)
s = rep(s, "'ARC3_MAX_ACTIVE_STREAMS': 10,", "'ARC3_MAX_ACTIVE_STREAMS': 14,  # B15: = vLLM --max-num-seqs 14 (LHS pool)")
# D43: the image-runtime boot (unpack ~19.5 GB + load + graph capture) is unmeasured on our account; the loop exits at READY.
s = rep(s, 'SERVER_STARTUP_TIMEOUT = 12 * 60\n', 'SERVER_STARTUP_TIMEOUT = 20 * 60  # B15 (D43): image-runtime boot unmeasured\n')
put(5, s)
put(7, "# B15: no Franzen wheel/model precache. The LHS boot prefetches its own GPU shards.\nprint('B15: precache skipped')\n")
put(12, "## 5. Start serving\n\nB15: start the LHS vLLM runtime (lordhansolo setup command, patched constants) instead of SGLang.")

BOOT = open('/data/arc/kaggle/b15_boot_cell.py').read()
cmd = json.load(open('/data/runs/intel/lhs_ds/setup_commands.json'))[0]
h = "<<'PYSETUP'\n"; body = cmd[cmd.index(h) + len(h):cmd.rindex('\nPYSETUP')]
BOOT = rep(BOOT, 'B15_LHS_SHA', hashlib.sha256(body.encode()).hexdigest()[:16])
put(13, BOOT)

n['cells'][0]['source'] = ("# B15: B12 harness on the LHS vLLM runtime (dev probe)\n\nPrivate experiment. Harness, prompts and settings = our B12 "
    "notebook. One change: serving = lordhansolo's vLLM main e975732 image runtime + primitive-ai mixed NVFP4-FP8 checkpoint "
    "(MTP head), 14 streams, instead of SGLang + Intel W4A16 at 10 streams. Save run = the B12-D25 probe (25 public games, 40 min each).\n\n"
    ).splitlines(True) + n['cells'][0]['source'][2:]
if base == 'b12_x6d':
    n['cells'][0]['source'] = ("# B15-submit: B12 submission notebook on the LHS vLLM runtime\n\nSubmission candidate, built only (no push "
        "or submit without the operator's go, D45). = our B12 submission notebook (arc3-b12-x6dprime) with B15's serving change: lordhansolo's "
        "vLLM main e975732 image runtime + primitive-ai mixed NVFP4-FP8 checkpoint (MTP head), 14 streams, instead of SGLang + Intel W4A16 "
        "at 10 streams. Save run = B12's (pipeline check only).\n\n").splitlines(True) + json.load(open('/data/arc/kaggle/b12_x6d/b12_x6d.ipynb'))['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json'))
lhs = json.load(open('/data/arc/kaggle/b14_lhs/kernel-metadata.json'))
# D43: keep B12-D25's GPU image. D41 pinned LHS's metadata image (gcr.io/kaggle-images/python@dafd4ce), which is the
# CPU image (a CPU-only public notebook carries the same digest); B14 died on it at 7.7 s with 'nvidia-smi failed'.
assert lhs['docker_image'] != m['docker_image'] and 'kaggle-private-byod' in m['docker_image']
m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn,
         dataset_sources=['dfranzen/taaf-kaggle-source-bundle-copy', 'lordhansolo/taaf-kaggle-source',
                          'lordhansolo/vllm-main-e975732-arc3'],
         model_sources=['lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1'])
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
