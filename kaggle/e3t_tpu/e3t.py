# E3T: E3 low-effort census + xhigh 60-task bench on Kaggle TPU v5e-8, Qwen3.8-27B bf16 via vllm-tpu 0.28 (recipe: github.com/ARahim3/kaggle-tpu-lab, MIT).
# Diagnostic only, never a submission. No tunnel: server bound to the kernel, client runs inside the same session.
"""
Serve Qwen3.8-27B (bf16) on a Kaggle TPU v5e-8 with vLLM.

This script is pushed to Kaggle as a script kernel by ../launch.py, which fills
in the CFG line below. It also runs standalone with defaults (e.g. pasted into
a Kaggle notebook/script in the UI) — then it just prints instead of using ntfy.

Steps (each one is announced in the log):
  1/6  runtime  — venv with vllm-tpu (pinned, CPU torch) built by uv in ~30 s,
                  resolution pinned to the env dataset's build date; + the MTP fix
  2/6  cache    — restore the pre-built XLA compile cache from the env dataset
  3/6  weights  — find the mounted weights dataset (or download from HF to /tmp)
  4/6  server   — start vLLM (TP=8, text-only, MTP speculative decoding)
  5/6  tunnel   — open a public cloudflared URL (printed before the server is
                  live so you can prepare your client)
  6/6  ready    — READY banner + self-test, then keep serving until
                  keepalive_min elapses

With both datasets attached the endpoint is live in ~22 minutes (~12 with
text_only, ~6 with fast_start). Without the env dataset the compile is cold (+15 min).
"""
import base64
import collections
import struct
import zlib
import glob
import gzip
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

CFG = {"text_only": True, "max_model_len": 69632, "max_num_seqs": 16, "keepalive_min": 1, "api_key": "sk-e3t-local"}

DEFAULTS = {
    "vllm_tpu_version": "0.28.0",
    "weights_dataset": "rahim3/qwen3-8-27b-bf16",     # HF mirror of Qwen/Qwen3.8-27B
    "env_dataset": "rahim3/qwen38-tpu-env-v5e8",       # XLA cache + cloudflared + manifest
    "hf_model_id": "Qwen/Qwen3.8-27B",                # fallback download source
    "max_model_len": 262144,       # native context; drop to 131072 + max_num_seqs 16 for throughput
    "max_num_seqs": 4,
    "mtp_tokens": 3,               # MTP spec decoding (+34% in our A/B test). Stock vllm-tpu
                                   # 0.28.0 corrupts outputs with it (missing GDN state
                                   # rollback); we apply patches/mtp-rollback-v0280.diff
                                   # (a port of upstream PR #3178) before serving —
                                   # verified lossless, 12/12 greedy exact-match.
    "async_scheduling": None,      # None = vLLM's default (on). false passes --no-async-scheduling: needed by
                                   # clients that use JSON mode / structured outputs while MTP is on (vllm-tpu
                                   # 0.28.0 hands the scheduler device arrays on that path -> "AttributeError:
                                   # __delitem__" and the server exits); costs some throughput
    "reasoning_effort_default": "xhigh",   # server-side default: xhigh | medium | low
    "tool_call_parser": "qwen3_coder",  # matches Qwen3.8's XML tool format; "" disables
    "text_only": False,            # True: skip the vision tower + its TPU graphs (saves ~8 min,
                                   # image inputs then error out)
    "min_token_bucket": 64,        # smallest padded batch (tokens); 16 = more graphs to compile
    "precompile_workers": 4,       # parallel XLA compile threads (1 = sequential)
    "fast_start": False,           # True: skip precompile -> READY in ~4 min (needs the env
                                   # dataset's cache); the script then warms the common
                                   # request shapes itself; rare shapes stall once (~1 min)
    "keepalive_min": 480,          # auto-shutdown guard (Kaggle TPU caps at 9h anyway)
    "api_key": "",                 # generated if empty
    "ntfy_topic": "",              # optional: publish progress to ntfy.sh/<topic>
    "served_model_name": "qwen3.8-27b",
    "verbose": False,              # show every vLLM log line (always saved to vllm.log)
    "build_bundle": False,         # maintainer mode: build the env dataset instead of serving
}
CFG = {**DEFAULTS, **(CFG or {})}
# Notebook flow: drop overrides in a serve_config.json next to this script.
_cfg_file = Path("serve_config.json")
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "sk-" + secrets.token_hex(16)

PORT = 8000
VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"
XLA_CACHE = "/tmp/xla_cache"
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
RAW_LOG = WORK / "vllm.log"          # every line vLLM/pip print, for debugging
CLOUDFLARED = Path("/tmp/cloudflared")
T0 = time.time()
PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"

os.environ["HF_HOME"] = "/tmp/hf"                 # /kaggle/working is only ~21 GB
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["VLLM_XLA_CACHE_PATH"] = XLA_CACHE
os.environ["MIN_TOKEN_BUCKET"] = str(CFG["min_token_bucket"])
os.environ["NUM_PRECOMPILE_WORKERS"] = str(CFG["precompile_workers"])
if CFG["fast_start"]:
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
# The venv ships its own libtpu; don't let the image's TPU_LIBRARY_PATH override it.
os.environ.pop("TPU_LIBRARY_PATH", None)

_raw = open(RAW_LOG, "a", buffering=1)


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    print(line, flush=True)
    _raw.write(line + "\n")


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/6  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    """Progress event: always logged; also pushed to ntfy if a topic is set."""
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab {phase}",
                "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"(ntfy publish failed: {e})")


def sh(cmd, tag, show=None, env=None):
    """Run a command; stream its output to vllm.log (and to the console when
    show/verbose). Returns the exit code."""
    show = CFG["verbose"] if show is None else show
    tail = collections.deque(maxlen=40)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        _raw.write(f"[{tag}] {line}\n")
        if show:
            print(f"[{tag}] {line[:400]}", flush=True)
    rc = p.wait()
    if rc != 0 and not show:
        log(f"[{tag}] exited with code {rc}; last lines:")
        for ln in list(tail)[-15:]:
            print("    " + ln[:300], flush=True)
    return rc


def find_input(*patterns):
    """Datasets mount at /kaggle/input/<slug> (UI) or /kaggle/input/datasets/<owner>/<slug> (API push)."""
    for pat in patterns:
        hits = glob.glob(f"/kaggle/input/{pat}") + glob.glob(f"/kaggle/input/datasets/*/{pat}")
        if hits:
            return hits[0]
    return None


def fetch_cloudflared():
    if CLOUDFLARED.exists():
        return
    try:
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    except Exception as e:
        log(f"(cloudflared download failed: {e})")


# gzip+base64 of patches/mtp-rollback-v0280.diff; regenerated by tools/embed_patch.py
MTP_PATCH_B64 = "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"  # __EMBEDDED_PATCH__


def apply_mtp_patch():
    """Port of upstream PR #3178 (GDN state rollback on rejected draft tokens).
    Without it ANY speculative decoding corrupts outputs on this model."""
    if not MTP_PATCH_B64:
        return True
    Path("/tmp/mtpfix.diff").write_text(
        gzip.decompress(base64.b64decode(MTP_PATCH_B64)).decode())
    origin = subprocess.check_output(
        [PY, "-c", "import importlib.util as u; print(u.find_spec('tpu_inference').origin)"],
        text=True).strip()
    pkg_root = os.path.dirname(os.path.dirname(origin))
    p = subprocess.run(["patch", "-p1", "-d", pkg_root, "-i", "/tmp/mtpfix.diff",
                        "--no-backup-if-mismatch", "-N"], capture_output=True, text=True)
    _raw.write(p.stdout + p.stderr)
    if p.returncode == 0 or "previously applied" in p.stdout:
        return True
    log(p.stdout[-1500:], p.stderr[-500:])
    return False


def runtime_ok():
    r = subprocess.run([PY, "-c", "import importlib.util as u, jax, torch\n"
                        "assert u.find_spec('vllm') and u.find_spec('tpu_inference')\n"
                        "print(jax.__version__, torch.__version__)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        log(f"   runtime check OK (jax {r.stdout.split()[0]}, torch {r.stdout.split()[1]})")
        return True
    log("   runtime check FAILED:", (r.stderr or r.stdout)[-800:])
    return False


def install_runtime(built=None):
    """Fresh venv with vllm-tpu pinned. CPU torch (what vllm-tpu's own Docker
    image uses) — the default PyPI torch drags in ~3 GB of CUDA libraries that
    a TPU never uses. `built` (a date from the env dataset's manifest) pins the
    dependency resolution to that day so the compile cache keeps matching."""
    ver = CFG["vllm_tpu_version"]
    shutil.rmtree(VENV, ignore_errors=True)
    pin = ["--exclude-newer", f"{built}T23:59:59Z"] if built else []
    log("   building venv with uv" + (f" (packages as of {built})" if built else "") + "...")
    if (sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip") == 0
            and sh([sys.executable, "-m", "uv", "venv", VENV, "--python", sys.executable, "-q"], "uv") == 0
            and sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
                    "--torch-backend=cpu", *pin, f"vllm-tpu=={ver}"], "uv") == 0):
        return "uv"
    log("   uv failed; falling back to pip (slower)")
    shutil.rmtree(VENV, ignore_errors=True)
    if sh([sys.executable, "-m", "venv", "--without-pip", VENV], "venv") != 0:
        return None
    rc = sh([sys.executable, "-m", "pip", "--python", PY, "install", "-q",
             "--extra-index-url", "https://download.pytorch.org/whl/cpu",
             f"vllm-tpu=={ver}"], "pip")
    return "pip" if rc == 0 else None


def tpu_check():
    """Kaggle sometimes starts a "TPU" session with no TPU attached (a CPU-only container; most often on new or
    not-yet-verified accounts). jax then sees one device and vLLM dies minutes later with "Insufficient devices for
    2D mesh: found 1, expected 8" or "No jellyfish device found". Look before installing anything (~20 s)."""
    code = ("import jax\n"
            "try:\n"
            "    d = jax.devices()\n"
            "    print('TPU_CHECK', len(d), d[0].platform, getattr(d[0], 'device_kind', ''))\n"
            "except Exception as e:\n"
            "    print('TPU_CHECK 0 none', str(e).replace(chr(10), ' ')[:200])\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"TPU_CHECK (\d+) (\S+)(.*)", out)
    except Exception as e:  # noqa: BLE001
        log(f"   (TPU check skipped: {e})")
        return
    if m is None:
        log("   (TPU check skipped: the image's jax did not answer; details in vllm.log)")
        _raw.write("[tpu-check] " + out.replace("\n", "\n[tpu-check] ") + "\n")
        return
    n, platform, rest = int(m.group(1)), m.group(2), m.group(3).strip()
    if n == 8 and platform == "tpu":
        log(f"   TPU check OK: 8 chips ({rest})")
        return
    if n == 0 and not re.search(r"jellyfish|TPU initialization failed|initialize backend 'tpu'|No TPU|vfio", rest, re.I):
        log(f"   (TPU check inconclusive, continuing: {rest[:160]})")   # only a clear no-TPU signal stops the run
        return
    msg = (f"this session has no working TPU: jax sees {n} {platform} device(s) {rest}. Kaggle sometimes starts a "
           "TPU session without one (most often on new or not-yet-verified accounts); nothing in this notebook can "
           "fix that. Stop the session and start it again. If it repeats, run `import jax; print(jax.device_count())` "
           "in a fresh cell first: it must print 8 before this script is worth running.")
    log("   " + msg)
    publish("failed", step="no-tpu", tail=msg)
    sys.exit(1)


def server_death_report(max_lines=60):
    """What killed vLLM, from vllm.log: the root-cause exception line, the first error block and a hint for the
    causes we have seen. The console tail alone scrolls the cause away. Returns (cause, block, hint)."""
    try:
        lines = [l for l in RAW_LOG.read_text(errors="replace").splitlines() if l.startswith("[vllm]")]
    except Exception:  # noqa: BLE001
        return "", "", ""
    strip = re.compile(r"^\[vllm\] (?:\((?:EngineCore|APIServer|Worker)[^)]*\) )?(?:ERROR|CRITICAL) [\d-]+ [\d:]+ \[[^\]]+\] ?")
    start_pat = re.compile(r"EngineCore (?:failed|encountered|hit)|Traceback \(most recent call last\)|RESOURCE_EXHAUSTED|INVALID_ARGUMENT|NOT_FOUND")
    start = next((i for i, l in enumerate(lines) if start_pat.search(l)), None)
    block = [strip.sub("", l) for l in lines[start:start + max_lines]] if start is not None else []
    cause = ""
    for l in block:                                   # the first traceback's own exception line is the root cause
        t = l.strip()
        if re.match(r"^[\w.]*(?:Error|Exception)\b.*:", t) or re.match(r"^(?:INVALID_ARGUMENT|NOT_FOUND|RESOURCE_EXHAUSTED)", t):
            cause = t
            break
    joined = "\n".join(block)
    hint = ""
    if re.search(r"found 1, expected 8|jellyfish|unexpected worker hostname|TPU initialization failed", joined):
        hint = ("this session has no working TPU (Kaggle sometimes starts one without, most often on new or "
                "not-yet-verified accounts): stop the session and start it again")
    elif "__delitem__" in joined:
        hint = ("a client sent a JSON-mode / structured-output request while MTP and async scheduling are on, which "
                "vllm-tpu 0.28.0 cannot handle: set \"async_scheduling\": false (or \"mtp_tokens\": 0) and run again")
    elif "RESOURCE_EXHAUSTED" in joined or "out of memory" in joined.lower():
        hint = "the TPU ran out of HBM: lower max_model_len or max_num_seqs"
    return cause, joined, hint


def server_died(server, phase, **extra):
    """Log why the vLLM server exited (root cause first), publish it, and stop the kernel."""
    cause, block, hint = server_death_report()
    tail = "\n".join(list(server.tail)[-40:]) if getattr(server, "tail", None) else ""
    log(f"server exited rc={server.returncode}" + (f" — root cause: {cause}" if cause else ""))
    if hint:
        log(f"   -> {hint}")
    if block:
        log("--- first error block from vllm.log ---\n" + block)
    if tail and not block:
        log("--- last output ---\n" + tail)
    log(f"full log: {RAW_LOG}")
    head = (f"root cause: {cause}\n" if cause else "") + (f"{hint}\n" if hint else "")
    publish(phase, rc=server.returncode, cause=cause, hint=hint,
            tail=(head + "\n" + (block or tail)[-2200:]).strip(), **extra)
    sys.exit(1)


# ---------------- 1. runtime ----------------
banner(1, "Python runtime", f"vllm-tpu {CFG['vllm_tpu_version']}")
tpu_check()
pass  # E3T: no cloudflared
bundle_root = find_input(CFG["env_dataset"].split("/")[-1], "qwen38-tpu-env*")
bundle, manifest = None, {}
if bundle_root:
    # Kaggle may keep the files at the top level or under the kernel's output folder
    hits = glob.glob(f"{bundle_root}/manifest.json") + glob.glob(f"{bundle_root}/*/manifest.json")
    if hits:
        bundle = os.path.dirname(hits[0])
        manifest = json.loads(Path(hits[0]).read_text())
    else:
        bundle = bundle_root
if bundle and Path(bundle, "cloudflared").exists() and not CLOUDFLARED.exists():
    shutil.copy(Path(bundle, "cloudflared"), CLOUDFLARED)
    CLOUDFLARED.chmod(0o755)
if manifest and (manifest.get("python") != PY_VER
                 or manifest.get("vllm_tpu_version") != CFG["vllm_tpu_version"]):
    log(f"   env dataset was built for python {manifest.get('python')} / vllm-tpu "
        f"{manifest.get('vllm_tpu_version')}; this session has python {PY_VER} and wants "
        f"vllm-tpu {CFG['vllm_tpu_version']} -> its compile cache will not match")
    manifest = {}
if not bundle:
    log(f"   env dataset not attached (expected {CFG['env_dataset']}) -> cold compile later")

t = time.time()
publish("install", vllm_tpu=CFG["vllm_tpu_version"])
log("   installer output goes to", RAW_LOG)
runtime = install_runtime(manifest.get("built"))
if runtime is None or not runtime_ok():
    publish("failed", step="install")
    sys.exit(1)
publish("installed", secs=int(time.time() - t), via=runtime)
if apply_mtp_patch():
    publish("mtp-patch-applied")
elif CFG["mtp_tokens"] > 0:
    publish("mtp-patch-failed", note="disabling MTP: unsafe without the rollback patch")
    CFG["mtp_tokens"] = 0
log(f"   runtime ready in {int(time.time() - t)} s")

# ---------------- 2. XLA compile cache ----------------
banner(2, "XLA compile cache")
t = time.time()
cache_tar = (Path(bundle, "xla_cache.tar") if bundle and Path(bundle, "xla_cache.tar").exists()
             else find_input("*/xla_cache*.tar.gz", "xla_cache*.tar.gz"))
cache_dir = find_input("*/*/xla_cache", "*/xla_cache", "xla_cache")
if cache_tar:
    flags = "-xf" if str(cache_tar).endswith(".tar") else "-xzf"
    sh(["tar", flags, str(cache_tar), "-C", "/tmp"], "tar")
elif cache_dir:
    sh(["cp", "-r", cache_dir, "/tmp/"], "cp")
    sh(["chmod", "-R", "u+w", XLA_CACHE], "chmod")
n_entries = len(glob.glob(XLA_CACHE + "/*"))
cache_configs = manifest.get("configs", [])
this_config = [CFG["max_model_len"], CFG["max_num_seqs"], CFG["mtp_tokens"], CFG["text_only"]]
if n_entries:
    covered = (not cache_configs) or (this_config in cache_configs)
    publish("cache-restored", entries=n_entries, secs=int(time.time() - t),
            covers_this_config=covered)
    if not covered:
        log(f"   note: the cache was built for [ctx, seqs, mtp, text_only] in {cache_configs}; "
            f"this run uses {this_config} -> its graphs compile cold (add ~10-15 min)")
    else:
        log("   compiled TPU graphs for this exact config are cached -> fast start")
else:
    publish("cache-missing", note="cold compile: expect ~10 extra minutes")

# ---------------- 3. weights ----------------
banner(3, "Model weights", "55 GB bf16 safetensors")
weights_slug = CFG["weights_dataset"].split("/")[-1]
model_path = find_input(weights_slug)
if model_path and os.path.exists(os.path.join(model_path, "config.json")):
    publish("weights-mounted", path=model_path)
else:
    publish("weights-download", model=CFG["hf_model_id"],
            note="attach the weights dataset to skip this (~5 min parallel download)")
    t = time.time()
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(CFG["hf_model_id"], allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "vocab*", "merges*"])
    publish("weights-downloaded", secs=int(time.time() - t))


# ---------------- 4. vLLM server ----------------
NOISE = ("vllm._C", "metadata.google.internal", "Triton is installed", "Transparent hugepages",
         "Pin memory is not supported", "Expect torch.Tensor", "Inductor compilation",
         "cloud_tpu_init.py", "SyntaxWarning", "Compilation of worker", "AOT lower skipped",
         "torch_dtype", "UserWarning", "warnings.warn", "resource_tracker", "Precompile worker0 sample",
         "Precompile worker0 gather", "Precompile worker0 compute_and_gather")


def server_args(cfg):
    args = [PY, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", "8",
            "--max-model-len", str(cfg["max_model_len"]),
            "--max-num-seqs", str(cfg["max_num_seqs"]),
            "--port", str(PORT),
            "--api-key", cfg["api_key"],
            "--served-model-name", cfg["served_model_name"],
            "--reasoning-parser", "qwen3"]
    if cfg.get("async_scheduling") is not None:
        args.append("--async-scheduling" if cfg["async_scheduling"] else "--no-async-scheduling")
    if cfg["text_only"]:
        # Qwen3.8 is a vision-language checkpoint; we only serve text. This skips
        # the vision tower and roughly halves the number of TPU graphs to compile.
        args += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if cfg["mtp_tokens"] > 0:
        args += ["--speculative-config",
                 json.dumps({"method": "mtp", "num_speculative_tokens": cfg["mtp_tokens"]})]
    if cfg["tool_call_parser"]:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_call_parser"]]
    if cfg["reasoning_effort_default"] != "xhigh":
        # The chat template defaults reasoning_effort to 'xhigh'; ship a copy with a
        # different default so the server-side default changes without client changes.
        tc = json.loads(Path(model_path, "tokenizer_config.json").read_text())
        template = tc["chat_template"].replace(
            "reasoning_effort|default('xhigh')",
            f"reasoning_effort|default('{cfg['reasoning_effort_default']}')")
        Path("/tmp/chat_template.jinja").write_text(template)
        args += ["--chat-template", "/tmp/chat_template.jinja"]
    return args


def n_token_graphs():
    n, b = 1, CFG["min_token_bucket"]
    while b < 2048:  # vLLM's default max_num_batched_tokens on TPU
        b *= 2
        n += 1
    return n


def make_translator():
    """Turns vLLM's firehose into a handful of human lines. Everything raw still
    lands in vllm.log."""
    st = {"graph": 0, "loads": 0, "said": set()}
    n_graphs = n_token_graphs()

    def once(key, msg):
        if key not in st["said"]:
            st["said"].add(key)
            log(msg)

    def tr(line):
        if CFG["verbose"]:
            print(f"[vllm] {line[:500]}", flush=True)
            return
        if any(k in line for k in NOISE):
            return
        m = re.search(r"Loading weights took ([\d.]+) seconds", line)
        if m:
            st["loads"] += 1
            if st["loads"] == 1:
                log(f"   weights read from the dataset in {float(m.group(1)):.0f} s")
            return
        m = re.search(r"load model weights from storage to TPU: ([\d.]+)", line)
        if m:
            if st["loads"] <= 1:
                log(f"   weights sharded across the 8 TPU chips ({float(m.group(1)):.0f} s)")
            else:
                log("   MTP draft head loaded")
            return
        m = re.search(r"KV cache size: ([\d,]+) tokens", line)
        if m:
            log(f"   KV cache fits {m.group(1)} tokens")
            return
        if "Precompile all the subgraphs" in line:
            log(f"   compiling TPU graphs — {n_graphs} text graphs"
                + ("" if CFG["text_only"] else ", the same again for image inputs,")
                + " + helpers (~20 s each if cached, ~1 min if not)")
            return
        m = re.search(r"Precompile worker\d+ backbone --> \{'num_tokens': (\d+)", line)
        if m:
            st["graph"] += 1
            log(f"     graph {st['graph']}/{n_graphs}: batches of {m.group(1)} tokens")
            return
        if "embed_multimodal" in line or "input_embeddings_merger" in line:
            once("vision-enc", "     warming the image encoder (~5 min; \"text_only\": true skips it)")
            return
        if "backbone with embeds" in line:
            once("vision", "     compiling image-input graphs (~3 min)")
            return
        m = re.search(r"Warm-up call pass finished in ([\d.]+) \[secs\] over (\d+) tasks", line)
        if m:
            if float(m.group(1)) > 5:
                log(f"     warm-up run of {m.group(2)} graphs done ({float(m.group(1)):.0f} s)")
            return
        if "Precompile" in line and "drafter" in line:
            once("mtp", "     compiling speculative-decoding (MTP) graphs")
            return
        if "Precompile" in line or "Compilation of" in line:
            once("helpers", "     compiling sampler / helper graphs")
            return
        if "Application startup complete" in line:
            return
        if " ERROR " in line or "Traceback" in line or "Error:" in line or "rror(" in line:
            print(time.strftime("[%H:%M:%S] ") + f"   [vllm] {line[:400]}", flush=True)
    return tr


def launch_server(cfg):
    publish("server-launch", max_model_len=cfg["max_model_len"],
            max_num_seqs=cfg["max_num_seqs"], mtp=cfg["mtp_tokens"],
            text_only=cfg["text_only"], min_token_bucket=cfg["min_token_bucket"])
    tail = collections.deque(maxlen=200)
    p = subprocess.Popen(server_args(cfg), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
    tr = make_translator()

    def pump():
        for line in p.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                _raw.write(f"[vllm] {line}\n")
                tr(line)
    threading.Thread(target=pump, daemon=True).start()
    p.tail = tail
    return p


def healthy(cfg):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                     headers={"Authorization": f"Bearer {cfg['api_key']}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(server, cfg, expect_min):
    t = time.time()
    while time.time() - t < 5400:
        if server.poll() is not None:
            server_died(server, "failed", step="server")
        if healthy(cfg):
            return int(time.time() - t)
        el = int(time.time() - t)
        if el and el % 120 < 6:
            publish("compiling", elapsed_s=el)
            log(f"   ... {el // 60} min into startup (typically ~{expect_min} min)")
        time.sleep(5)
    publish("failed", step="health-timeout", tail="\n".join(list(server.tail)[-60:])[-2500:])
    sys.exit(1)


def stop_server(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=30)
    time.sleep(10)  # let the TPU runtime free the chips


def completion(cfg, prompt, max_tokens, stream=False, timeout=900):
    body = {"model": cfg["served_model_name"], "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0}
    if stream:
        body.update(stream=True, ignore_eos=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    t0 = time.time(); ttft = None; gen = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                gen += 1
            if obj.get("usage"):
                gen = obj["usage"].get("completion_tokens", gen)
    return ttft, time.time() - t0, gen


def test_png(w=256, h=256, rgb=(200, 30, 30)):
    """A solid-colour PNG without PIL, for the image self-test."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def chat(cfg, messages, max_tokens=32, timeout=600):
    body = {"model": cfg["served_model_name"], "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def self_test(cfg):
    """Warms the remaining lazy paths, checks an image request when images are
    enabled, and reports single-stream decode speed."""
    tps = None
    try:
        completion(cfg, "Hello", 8)
        txt = completion(cfg, "The capital of France is", 8)["choices"][0]["text"]
        ttft, total, gen = completion(cfg, "Write a short story about a lighthouse.", 192, stream=True)
        tps = (gen - 1) / (total - ttft) if gen > 1 else 0.0
        publish("benchmark", decode_tok_s=round(tps, 1), sanity=txt.strip()[:60])
        log(f"   self-test: {tps:.1f} tok/s single-stream decode; "
            f"'The capital of France is' -> {txt.strip()[:40]!r}")
    except Exception as e:
        publish("benchmark-error", err=str(e)[:200])
    if not cfg["text_only"]:
        try:
            img = "data:image/png;base64," + base64.b64encode(test_png()).decode()
            ans = chat(cfg, [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": "What colour is this image? One word."}]}])
            publish("image-test", answer=ans.strip()[:40])
            log(f"   image request works (a red square -> {ans.strip()[:30]!r})")
        except Exception as e:
            publish("image-test-failed", err=str(e)[:200])
            log(f"   IMAGE REQUEST FAILED: {str(e)[:200]}")
    return tps


def exercise(cfg, quiet=False):
    """Hit the shapes a real client hits: short and long prompts, streaming, a
    small concurrent batch. In build mode this puts their graphs in the cache;
    in fast_start mode it loads them so users don't hit the one-time stalls."""
    steps = [("short prompt", lambda: completion(cfg, "Hello", 8)),
             ("4k-token prompt", lambda: completion(
                 cfg, "The quick brown fox jumps over the lazy dog. " * 400, 16, timeout=1200)),
             ("streaming", lambda: completion(
                 cfg, "Write a short story about a lighthouse.", 64, stream=True))]
    n = min(cfg["max_num_seqs"], 8)
    errs = []

    def one():
        try:
            completion(cfg, "Count from one to twenty in words.", 48)
        except Exception as e:
            errs.append(str(e)[:200])

    def batch():
        ths = [threading.Thread(target=one) for _ in range(n)]
        [t.start() for t in ths]
        [t.join() for t in ths]
    steps.append((f"{n} parallel requests", batch))
    for name, fn in steps:
        t = time.time()
        try:
            fn()
        except Exception as e:
            errs.append(f"{name}: {str(e)[:200]}")
        if not quiet:
            log(f"   warmed: {name} ({time.time() - t:.0f} s)")
    return errs


# ---------------- maintainer mode: build the env dataset ----------------
BUILD_CONFIGS = [
    {"max_model_len": 262144, "max_num_seqs": 4, "mtp_tokens": 3, "text_only": False},
    {"max_model_len": 131072, "max_num_seqs": 16, "mtp_tokens": 3, "text_only": False},
    {"max_model_len": 262144, "max_num_seqs": 4, "mtp_tokens": 3, "text_only": True},
]
if CFG["build_bundle"]:
    banner(4, "BUILD MODE", "serving each config once to populate the XLA cache")
    log("   TPU-related env:", {k: v for k, v in os.environ.items() if "TPU" in k or "PJRT" in k})
    results = {}
    for c in BUILD_CONFIGS:
        cfg = {**CFG, **c}
        key = (f"{c['max_model_len']}/{c['max_num_seqs']}/mtp{c['mtp_tokens']}"
               + ("/text" if c["text_only"] else "/mm"))
        server = launch_server(cfg)
        secs = wait_healthy(server, cfg, 30)
        errs = exercise(cfg, quiet=True)
        tps = self_test(cfg)
        stop_server(server)
        results[key] = {"startup_secs": secs, "decode_tok_s": tps, "errors": errs}
        publish("build-config-done", config=key, **results[key])
    # probe: how fast is a start with SKIP_JAX_PRECOMPILE=1 now that the cache is warm?
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    cfg = {**CFG, **BUILD_CONFIGS[0]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 5)
    lat = []
    for i in range(3):
        t = time.time()
        try:
            completion(cfg, ["Hello", "Say hi.", "Name a color."][i], 8, timeout=1800)
            lat.append(round(time.time() - t, 1))
        except Exception as e:
            lat.append(str(e)[:100])
    tps = self_test(cfg)
    stop_server(server)
    os.environ.pop("SKIP_JAX_PRECOMPILE")
    publish("probe-skip-precompile", startup_secs=secs, first_request_secs=lat, decode_tok_s=tps)
    # also a warm re-start of the default config (what users will see)
    cfg = {**CFG, **BUILD_CONFIGS[0]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 15)
    tps = self_test(cfg)
    stop_server(server)
    publish("probe-warm-restart", startup_secs=secs, decode_tok_s=tps)
    cfg = {**CFG, **BUILD_CONFIGS[2]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 8)
    tps = self_test(cfg)
    stop_server(server)
    publish("probe-warm-restart-text-only", startup_secs=secs, decode_tok_s=tps)

    out = WORK / "bundle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    banner(5, "packing the bundle", str(out))
    # (no venv tarball: uv rebuilds the identical env in ~30 s, and Kaggle would
    #  unpack a tar into 100k files anyway — some with '[' in the name, which it rejects)
    sh(["tar", "-cf", str(out / "xla_cache.tar"), "-C", "/tmp", "xla_cache"], "tar")
    fetch_cloudflared()
    if CLOUDFLARED.exists():
        shutil.copy(CLOUDFLARED, out / "cloudflared")
    pkgs = subprocess.run([sys.executable, "-m", "uv", "pip", "list", "--python", PY,
                           "--format=json"], capture_output=True, text=True)
    try:
        pkgs = {d["name"]: d["version"] for d in json.loads(pkgs.stdout)}
    except Exception:
        pkgs = {}
    manifest = {
        "built": time.strftime("%Y-%m-%d"),
        "python": PY_VER,
        "vllm_tpu_version": CFG["vllm_tpu_version"],
        "mtp_patch": "applied at runtime",
        "min_token_bucket": CFG["min_token_bucket"],
        "configs": [[c["max_model_len"], c["max_num_seqs"], c["mtp_tokens"], c["text_only"]]
                    for c in BUILD_CONFIGS],
        "results": results,
        "accelerator": "TPU v5e-8 (Kaggle)",
        "packages": pkgs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sizes = {p.name: round(p.stat().st_size / 1e9, 2) for p in out.iterdir()}
    publish("bundle-built", sizes_gb=sizes, results=results)
    sys.exit(0)

# ---------------- 4. launch ----------------
expect_min = (10 if n_entries else 20) + (0 if CFG["text_only"] else (10 if n_entries else 15))
if CFG["fast_start"]:
    expect_min = 5 if n_entries else 7
    if not n_entries:
        log("   fast_start without a compile cache: every new request shape will compile "
            "cold (~1 min each) — attach the env dataset for this mode to make sense")
banner(4, "Starting vLLM", f"TP=8, ctx {CFG['max_model_len']}, {CFG['max_num_seqs']} seqs, "
       f"MTP k={CFG['mtp_tokens']}, {'text-only' if CFG['text_only'] else 'multimodal'}")
log(f"   expect ~{expect_min} min; progress lines below, full vLLM log in {RAW_LOG}")
server = launch_server(CFG)

url = None  # E3T: no tunnel, server stays on 127.0.0.1 inside the kernel

# ---------------- 6. wait, announce, self-test, keep alive ----------------
startup = wait_healthy(server, CFG, expect_min)
publish("serving", startup_secs=startup)
log("")
log("#" * 70)
log(f"#  READY — the server is live ({elapsed()} after start)")
log(f"#  ENDPOINT : {url + '/v1' if url else 'http://127.0.0.1:8000/v1 (tunnel failed)'}")
log(f"#  API KEY  : {CFG['api_key']}")
log(f"#  MODEL    : {CFG['served_model_name']}   (context {CFG['max_model_len']}, "
    f"{CFG['max_num_seqs']} parallel requests)")
log("#" * 70)
log("#  Try it:")
log(f"#    curl {url + '/v1' if url else 'http://127.0.0.1:8000/v1'}/chat/completions \\")
log(f"#      -H 'Authorization: Bearer {CFG['api_key']}' -H 'Content-Type: application/json' \\")
log("#      -d '{\"model\": \"" + CFG["served_model_name"] + "\", \"messages\": [{\"role\": \"user\", "
    "\"content\": \"Hello!\"}], \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}}'")
log(f"#  Serving for up to {CFG['keepalive_min']} min, then this cell exits on its own.")
log("#" * 70)
publish("ready", endpoint=(f"{url}/v1" if url else None), api_key=CFG["api_key"],
        model=CFG["served_model_name"], max_model_len=CFG["max_model_len"],
        keepalive_min=CFG["keepalive_min"], startup_secs=startup)

if CFG["fast_start"]:
    banner(6, "Warm-up", "loading the common request shapes; the endpoint is usable meanwhile")
    log("   (fast_start: a request with a new shape waits ~1 min the first time)")
    exercise(CFG)
else:
    banner(6, "Self-test", "one short generation; the endpoint is usable meanwhile")
self_test(CFG)


# ---------------- E3T: run the ARC-AGI-2 census client against the local server ----------------
CLIENT_B64 = "IyBFMyBsb3ctZWZmb3J0IGNlbnN1cyA9IEUyLjEgd2l0aCB0aGVzZSBjb21taXQgZGVmYXVsdHM6IGFsbCAxMjAgcHVibGljLWV2YWwgdGFza3MsIDIgc2FtcGxlcyBwZXIgaW5wdXQgYXQgcmVhc29uaW5nX2VmZm9ydD1sb3csCiMgMTZrIGNhcCwgMTYgcGVyIHNlcnZlciwgZnVsbCBjaGFpbnMgc2F2ZWQuIE5vdCBmb3Igc3VibWlzc2lvbiAoZGlhZ25vc3RpYyBvbmx5KS4KaW1wb3J0IG9zCmZvciBfaywgX3YgaW4gZGljdChFMl9FRkZPUlQ9J2xvdycsIEUyX0NPTU1JVF9OVEFTSz0nMTIwJywgRTJfQ09NTUlUX01BWFRPSz0nMTYzODQnLCBFMl9DT01NSVRfTU9ERT0nY2Vuc3VzJywgRTJfQ09NTUlUX0JVREdFVF9TPSc5MDAwJywgRTJfUEVSPScxNicsIEUyX1JBTVBfR0FQPSc0MCcsIEUyX01JTl9HQVA9JzMnKS5pdGVtcygpOiBvcy5lbnZpcm9uLnNldGRlZmF1bHQoX2ssIF92KQojIEUyLjE6IG1vZGVsLWZpcnN0IEFSQy1BR0ktMiBzdWJtaXNzaW9uLiBRd2VuMy44LTI3QiBHUFRRLVc0QTE2LCAyIHZMTE0gcmVwbGljYXMgeCBUUDIgb24gTDR4NCAoRTFjIGNvbmZpZzogcGFzc0AyIDAuMzgzCiMgb24gMzAgcHVibGljLWV2YWwgdGFza3MsIGRlZmF1bHQgY2hhdCB0ZW1wbGF0ZSA9IHJlYXNvbmluZyBlZmZvcnQgeGhpZ2gpLiBDb21taXQgbW9kZSA9IHNob3J0IHNtb2tlOyBzY29yaW5nIHJlcnVuID0gMjQwIGhpZGRlbgojIHRhc2tzIC8gfjI1OSB0ZXN0IGlucHV0cyAoS2FnZ2xlIGRhdGEgcGFnZSArIHBsYWNlaG9sZGVyIGZpbGUpLCBpLmUuIH4yeCB0aGUgMTIwIHRhc2tzIGFzc3VtZWQgaW4gRDE0LCBzbyB0aHJvdWdocHV0IGRlY2lkZXMKIyBob3cgbWFueSBpbnB1dHMgZ2V0IGEgc2Vjb25kIGF0dGVtcHQuCiMgQ2hhbmdlcyB2cyBFMiAobmV2ZXIgc3VibWl0dGVkKToKIyAgKDEpIGRpc3BhdGNoZXI6IFBFUiB3b3JrZXJzIGJvdW5kIHRvIGVhY2ggc2VydmVyIChFMWMgcm91dGVkIGJ5IGpvYiBpbmRleCwgdXAgdG8gMTQgdnMgMTAgcnVubmluZyksIHN0YWdnZXJlZCByYW1wLCBhbmQgYQojICAgICAgS1YtcHJlc3N1cmUgYWRtaXNzaW9uIGdhdGUgKEUxYyBzcGVudCB+MjAlIG9mIHNlcnZlci10aW1lIGF0IEtWID49IDk3JSwgd2l0aCBzdGFsbHMgYXQgNS0xMCB0b2svcyk7CiMgICgyKSBhdHRlbXB0LTEgcHJpb3JpdHk6IGlucHV0cyB3aG9zZSBhdHRlbXB0IDAgZ2F2ZSBubyBuYXR1cmFsIGFuc3dlciBnbyBmaXJzdCAoRTFjOiBhbGwgMyBhdHRlbXB0LTEgZ2FpbnMgY2FtZSBmcm9tIHRoZW0sCiMgICAgICAwLzEyIGZyb20gaW5wdXRzIHdob3NlIGF0dGVtcHQgMCBmaW5pc2hlZCk7CiMgICgzKSBhbnN3ZXIgZm9yY2luZyB0aHJvdWdoIC92MS9jb21wbGV0aW9ucyBvbiB0aGUgbW9kZWwncyBvd24gY2hhdC10ZW1wbGF0ZSBwcmVmaXggKHRoZSBjaGF0IHBhdGggcmUtcmVuZGVycyBhbiBlbXB0eSB0aGluawojICAgICAgYmxvY2sgaW4gZnJvbnQgb2YgdGhlIGluamVjdGVkIHJlYXNvbmluZyk7CiMgICg0KSBkZWFkbGluZS1hd2FyZSBtYXhfdG9rZW5zIChsYXRlIGpvYnMgZW5kIHRydW5jYXRlZCBhbmQgZ2V0IGZvcmNlZCBpbnN0ZWFkIG9mIHRpbWluZyBvdXQpOyBnZW5lcmF0aW9uIGVuZHMgYXQgVDAgKyAxMS4wIGguCmltcG9ydCBvcywgc3lzLCBqc29uLCBnbG9iLCByZSwgdGltZSwgcmFuZG9tLCBzdWJwcm9jZXNzLCB0aHJlYWRpbmcsIGNvbGxlY3Rpb25zLCB1cmxsaWIucmVxdWVzdApUMCA9IHRpbWUudGltZSgpCklOID0gb3MuZW52aXJvbi5nZXQoJ0UyX0lOJywgJy9rYWdnbGUvaW5wdXQnKTsgT1VUID0gb3MuZW52aXJvbi5nZXQoJ0UyX09VVCcsICcva2FnZ2xlL3dvcmtpbmcnKQpGQUtFID0gb3MuZW52aXJvbi5nZXQoJ0UyX0ZBS0UnLCAnJykgPT0gJzEnICAgIyBsb2NhbCB0ZXN0IGFnYWluc3QgZmFrZSBzZXJ2ZXJzIG9uIDgwMDEvODAwMgpSRVJVTiA9IG9zLmVudmlyb24uZ2V0KCdLQUdHTEVfSVNfQ09NUEVUSVRJT05fUkVSVU4nLCAnJykuc3RyaXAoKS5sb3dlcigpIGluIHsnMScsICd0cnVlJ30KIyBDb21taXQgcnVuIChub3Qgc2NvcmVkKTogJ2JlbmNoJyA9IDYwIHB1YmxpYy1ldmFsIHRhc2tzIGF0IHRoZSBmdWxsIDQ4ayBjYXAgZm9yIDIuNSBoICh0aHJvdWdocHV0LCBwYXNzQDEvMiwgZm9yY2VkLWFuc3dlcgojIGFjY3VyYWN5LCBmdWxsIGNoYWlucyBzYXZlZCk7ICdzbW9rZScgPSAzIHRhc2tzIGF0IGEgNmsgY2FwIGZvciAxNSBtaW4uIFRoZSBzY29yaW5nIHJlcnVuIGFsd2F5cyB1c2VzIHRoZSBoaWRkZW4gdGVzdCBmaWxlLgpDT01NSVRfTU9ERSA9IG9zLmVudmlyb24uZ2V0KCdFMl9DT01NSVRfTU9ERScsICdiZW5jaCcpCl9CID0gQ09NTUlUX01PREUgPT0gJ2JlbmNoJwpDRkcgPSBkaWN0KAogICAgSEFSRF9FTkRfUz1mbG9hdChvcy5lbnZpcm9uLmdldCgnRTJfSEFSRF9FTkRfUycsIHN0cigxMS4wICogMzYwMCkpKSwgICAjIHJlcnVuOiBnZW5lcmF0aW9uIHN0b3BzIGF0IFQwICsgdGhpcyAoMTIgaCBsaW1pdCwgfjU1IG1pbiBtYXJnaW4pCiAgICBDT01NSVRfQlVER0VUX1M9ZmxvYXQob3MuZW52aXJvbi5nZXQoJ0UyX0NPTU1JVF9CVURHRVRfUycsICc5MDAwJyBpZiBfQiBlbHNlICc5MDAnKSksCiAgICBNQVhUT0s9NDkxNTIgaWYgKFJFUlVOIG9yIF9CKSBlbHNlIGludChvcy5lbnZpcm9uLmdldCgnRTJfQ09NTUlUX01BWFRPSycsICc2MTQ0JykpLAogICAgTlRBU0s9MTAqKjYgaWYgUkVSVU4gZWxzZSBpbnQob3MuZW52aXJvbi5nZXQoJ0UyX0NPTU1JVF9OVEFTSycsICc2MCcgaWYgX0IgZWxzZSAnMycpKSwKICAgIFBFUj1pbnQob3MuZW52aXJvbi5nZXQoJ0UyX1BFUicsICcxMicpKSwgICAjIGNvbmN1cnJlbnQgcmVxdWVzdHMgcGVyIHNlcnZlciAoRTFjOiAxMiBvbiBhdmVyYWdlKQogICAgS1ZfQURNSVQ9ZmxvYXQob3MuZW52aXJvbi5nZXQoJ0UyX0tWX0FETUlUJywgJzAuODInKSksCiAgICBSQU1QX0dBUD1mbG9hdChvcy5lbnZpcm9uLmdldCgnRTJfUkFNUF9HQVAnLCAnMTIwJyBpZiAoUkVSVU4gb3IgX0IpIGVsc2UgJzMnKSksICAgIyBzcGFjaW5nIG9mIHRoZSBmaXJzdCBQRVIgYWRtaXNzaW9ucyBwZXIgc2VydmVyCiAgICBNSU5fR0FQPWZsb2F0KG9zLmVudmlyb24uZ2V0KCdFMl9NSU5fR0FQJywgJzgnIGlmIChSRVJVTiBvciBfQikgZWxzZSAnMC41JykpLAogICAgVFBTX1NUUkVBTT0xOC4wLCAgICAgICAgICAgICAgICAgICAgICAgICAgICMgY29uc2VydmF0aXZlIHBlci1yZXF1ZXN0IGRlY29kZSBzcGVlZCBmb3IgdGhlIGRlYWRsaW5lIGNsYW1wIChFMWMgfjIyKQogICAgVEVNUFM9KDAuNiwgMS4wKSwgRUZGT1JUPW9zLmVudmlyb24uZ2V0KCdFMl9FRkZPUlQnLCAnJyksIFNBVkVfQ0hBSU5TPShub3QgUkVSVU4pLCBNT0RFPSdyZXJ1bicgaWYgUkVSVU4gZWxzZSBDT01NSVRfTU9ERSkKTUFYVE9LID0gQ0ZHWydNQVhUT0snXTsgTlRBU0sgPSBDRkdbJ05UQVNLJ107IE5BVFQgPSAyCk1BWExFTiA9IE1BWFRPSyArIDE2Mzg0ICAgIyByb29tIGZvciBsYXJnZS1ncmlkIHByb21wdHMgKDMweDMwIHRhc2tzIGNhbiBleGNlZWQgOGsgcHJvbXB0IHRva2VucykKVFAgPSBpbnQob3MuZW52aXJvbi5nZXQoJ0UxQ19UUCcsICcyJykpClhISUdIID0gKCdSZWFzb25pbmcgZWZmb3J0IGlzIHNldCB0byB4aGlnaC4gUGxlYXNlIHRoaW5rIGNhcmVmdWxseSB0aHJvdWdoIHRoZSB0YXNrLCB2YWxpZGF0ZSBrZXkgYXNzdW1wdGlvbnMsIGNvbnNpZGVyIHBsYXVzaWJsZSAnCiAgICAgICAgICdhbHRlcm5hdGl2ZXMsIGFuZCBwcmlvcml0aXplIGNvcnJlY3RuZXNzLCBjb25zaXN0ZW5jeSwgYW5kIGNsYXJpdHkgaW4gdGhlIGZpbmFsIGFuc3dlci4nKQpMT1cgPSAnUmVhc29uaW5nIGVmZm9ydCBpcyBzZXQgdG8gbG93LiBLZWVwIHlvdXIgdGhpbmtpbmcgYnJpZWYgYW5kIGZvY3VzZWQsIG1vdmluZyBkaXJlY3RseSB0byB0aGUgY29uY2x1c2lvbiB3aXRob3V0IHVubmVjZXNzYXJ5IGVsYWJvcmF0aW9uLicKZGVmIGxvZygqYSk6IHByaW50KGYnW3t0aW1lLnRpbWUoKS1UMDo3LjBmfXNdJywgKmEsIGZsdXNoPVRydWUpCmRlZiBzaChjKTogc3VicHJvY2Vzcy5ydW4oYywgc2hlbGw9VHJ1ZSkKZGVmIGluc3RhbGwoKToKICAgIHRyeToKICAgICAgICBpbXBvcnQgdmxsbTsgcmV0dXJuICAjIG5vcWEKICAgIGV4Y2VwdCBFeGNlcHRpb246IHBhc3MKICAgIHdobCA9IGdsb2IuZ2xvYihmJ3tJTn0vKiovdmxsbS0qLndobCcsIHJlY3Vyc2l2ZT1UcnVlKQogICAgaWYgd2hsOiB3ZCA9IG9zLnBhdGguZGlybmFtZSh3aGxbMF0pCiAgICBlbHNlOgogICAgICAgIGFyYyA9IGdsb2IuZ2xvYihmJ3tJTn0vKiovd2hlZWxzLnRhci5neicsIHJlY3Vyc2l2ZT1UcnVlKQogICAgICAgIHN1YnByb2Nlc3MucnVuKGYnbWtkaXIgLXAgL3RtcC93aCAmJiB0YXIgLXh6ZiB7YXJjWzBdfSAtQyAvdG1wL3doJywgc2hlbGw9VHJ1ZSwgY2hlY2s9VHJ1ZSkKICAgICAgICB3ZCA9IG9zLnBhdGguZGlybmFtZShnbG9iLmdsb2IoJy90bXAvd2gvKiovKi53aGwnLCByZWN1cnNpdmU9VHJ1ZSlbMF0pCiAgICBsb2coJ3BpcCBpbnN0YWxsIHZsbG0gZnJvbScsIHdkKQogICAgc3VicHJvY2Vzcy5ydW4oW3N5cy5leGVjdXRhYmxlLCAnLW0nLCAncGlwJywgJ2luc3RhbGwnLCAnLXEnLCAnLS1uby1pbmRleCcsICctLWZpbmQtbGlua3MnLCB3ZCwgJ3ZsbG0nXSwgY2hlY2s9VHJ1ZSkKZGVmIGVudigpOgogICAgb3MuZW52aXJvblsnVkxMTV9VU0VfRkxBU0hJTkZFUl9TQU1QTEVSJ10gPSAnMCcKICAgIHNoKCdMPSQobGRjb25maWcgLXAgfCBncmVwIC1tMSAibGliY3VkYS5zby4xICIgfCBhd2sgXCd7cHJpbnQgJE5GfVwnKTsgWyAtbiAiJEwiIF0gJiYgbWtkaXIgLXAgL3RtcC9jdWRhbGliICYmIGxuIC1zZiAkTCAvdG1wL2N1ZGFsaWIvbGliY3VkYS5zbzsgZWNobyBsaWJjdWRhICRMJykKICAgIG9zLmVudmlyb25bJ0xJQlJBUllfUEFUSCddID0gJy90bXAvY3VkYWxpYjovdXNyL2xvY2FsL252aWRpYS9saWI2NDonICsgb3MuZW52aXJvbi5nZXQoJ0xJQlJBUllfUEFUSCcsICcnKQogICAgc2goJ252aWRpYS1zbWkgLS1xdWVyeS1ncHU9bmFtZSxtZW1vcnkudG90YWwgLS1mb3JtYXQ9Y3N2OyBucHJvYzsgZnJlZSAtZyB8IGhlYWQgLTInKQpkZWYgZ2V0KHVybCwgdD01KToKICAgIHRyeToKICAgICAgICB3aXRoIHVybGxpYi5yZXF1ZXN0LnVybG9wZW4odXJsLCB0aW1lb3V0PXQpIGFzIHI6IHJldHVybiByLnN0YXR1cywgci5yZWFkKCkuZGVjb2RlKCkKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZTogcmV0dXJuIDAsIHN0cihlKQpkZWYgc3RhcnQobWRpciwgZ3B1cywgcG9ydCwgcHJvZik6CiAgICBhID0gW3N5cy5leGVjdXRhYmxlLCAnLW0nLCAndmxsbS5lbnRyeXBvaW50cy5vcGVuYWkuYXBpX3NlcnZlcicsICctLW1vZGVsJywgbWRpciwgJy0tc2VydmVkLW1vZGVsLW5hbWUnLCAnbScsICctLXBvcnQnLCBzdHIocG9ydCksCiAgICAgICAgICctLXRlbnNvci1wYXJhbGxlbC1zaXplJywgc3RyKGxlbihncHVzLnNwbGl0KCcsJykpKSwgJy0tbWF4LW1vZGVsLWxlbicsIHN0cihNQVhMRU4pLCAnLS1ncHUtbWVtb3J5LXV0aWxpemF0aW9uJywgJzAuOTInLAogICAgICAgICAnLS1tYXgtbnVtLXNlcXMnLCBzdHIocHJvZi5nZXQoJ3NlcXMnLCAxNikpLCAnLS1tYXgtbnVtLWJhdGNoZWQtdG9rZW5zJywgJzgxOTInLCAnLS1lbmFibGUtY2h1bmtlZC1wcmVmaWxsJywgJy0tbm8tZW5hYmxlLXByZWZpeC1jYWNoaW5nJywKICAgICAgICAgJy0tdHJ1c3QtcmVtb3RlLWNvZGUnLCAnLS1uby1lbmFibGUtbG9nLXJlcXVlc3RzJywgJy0tZGlzYWJsZS11dmljb3JuLWFjY2Vzcy1sb2cnLCAnLS1yZWFzb25pbmctcGFyc2VyJywgJ3F3ZW4zJ10KICAgIGlmIHByb2YuZ2V0KCdsbW9ubHknKTogYSArPSBbJy0tbGFuZ3VhZ2UtbW9kZWwtb25seSddCiAgICBpZiBwcm9mLmdldCgnZnA4a3YnKTogYSArPSBbJy0ta3YtY2FjaGUtZHR5cGUnLCAnZnA4J10KICAgIGlmIHByb2YuZ2V0KCdtdHAnKTogYSArPSBbJy0tc3BlY3VsYXRpdmUtY29uZmlnJywganNvbi5kdW1wcyh7J21ldGhvZCc6ICdtdHAnLCAnbnVtX3NwZWN1bGF0aXZlX3Rva2Vucyc6IHByb2ZbJ210cCddfSldCiAgICBpZiBwcm9mLmdldCgnYXN5bmMnKTogYSArPSBbJy0tYXN5bmMtc2NoZWR1bGluZyddCiAgICBlID0gZGljdChvcy5lbnZpcm9uLCBDVURBX1ZJU0lCTEVfREVWSUNFUz1ncHVzKQogICAgZiA9IG9wZW4oZid7T1VUfS9zZXJ2ZXJfe3BvcnR9LmxvZycsICdhJyk7IGYud3JpdGUoZidcbj09PT0ge3Byb2Z9XG4nKTsgZi5mbHVzaCgpCiAgICByZXR1cm4gc3VicHJvY2Vzcy5Qb3BlbihhLCBlbnY9ZSwgc3Rkb3V0PWYsIHN0ZGVycj1zdWJwcm9jZXNzLlNURE9VVCkKZGVmIHdhaXRfdXAocHJvY3MsIHBvcnRzLCBsaW1pdD0xNTAwKToKICAgIHQgPSB0aW1lLnRpbWUoKQogICAgd2hpbGUgdGltZS50aW1lKCkgLSB0IDwgbGltaXQ6CiAgICAgICAgaWYgYW55KHAucG9sbCgpIGlzIG5vdCBOb25lIGZvciBwIGluIHByb2NzKTogcmV0dXJuIEZhbHNlCiAgICAgICAgaWYgYWxsKGdldChmJ2h0dHA6Ly8xMjcuMC4wLjE6e3B9L2hlYWx0aCcpWzBdID09IDIwMCBmb3IgcCBpbiBwb3J0cyk6IHJldHVybiBUcnVlCiAgICAgICAgdGltZS5zbGVlcCgxMCkKICAgIHJldHVybiBGYWxzZQpQUk9GSUxFUyA9IFtkaWN0KG10cD0zLCBmcDhrdj0xLCBsbW9ubHk9MSwgYXN5bmNfPTAsIHNlcXM9MTYpLCBkaWN0KG10cD0wLCBmcDhrdj0xLCBsbW9ubHk9MSwgc2Vxcz0xNiksIGRpY3QobXRwPTAsIGZwOGt2PTAsIGxtb25seT0wLCBzZXFzPTEyKV0KZGVmIHBvc3QocG9ydCwgcGF0aCwgYm9keSwgdGltZW91dCk6CiAgICBycSA9IHVybGxpYi5yZXF1ZXN0LlJlcXVlc3QoZidodHRwOi8vMTI3LjAuMC4xOntwb3J0fXtwYXRofScsIGpzb24uZHVtcHMoYm9keSkuZW5jb2RlKCksIGRpY3QoeydDb250ZW50LVR5cGUnOiAnYXBwbGljYXRpb24vanNvbid9LCAqKih7J0F1dGhvcml6YXRpb24nOiAnQmVhcmVyICcgKyBvcy5lbnZpcm9uWydFMl9BUElfS0VZJ119IGlmIG9zLmVudmlyb24uZ2V0KCdFMl9BUElfS0VZJykgZWxzZSB7fSkpKQogICAgd2l0aCB1cmxsaWIucmVxdWVzdC51cmxvcGVuKHJxLCB0aW1lb3V0PXRpbWVvdXQpIGFzIHI6IHJldHVybiBqc29uLmxvYWQocikKV0FOVCA9ICgndmxsbTprdl9jYWNoZV91c2FnZV9wZXJjJywgJ3ZsbG06bnVtX3JlcXVlc3RzX3dhaXRpbmcnLCAndmxsbTpudW1fcmVxdWVzdHNfcnVubmluZycsICd2bGxtOmdlbmVyYXRpb25fdG9rZW5zX3RvdGFsJywKICAgICAgICAndmxsbTpzcGVjX2RlY29kZV9udW1fYWNjZXB0ZWRfdG9rZW5zX3RvdGFsJywgJ3ZsbG06c3BlY19kZWNvZGVfbnVtX2RyYWZ0X3Rva2Vuc190b3RhbCcpCmRlZiBtYWluKCk6CiAgICBkZWYgZmlyc3QocGF0KToKICAgICAgICBnID0gZ2xvYi5nbG9iKGYne0lOfS8qKi97cGF0fScsIHJlY3Vyc2l2ZT1UcnVlKTsgcmV0dXJuIGdbMF0gaWYgZyBlbHNlIE5vbmUKICAgIHRyeToKICAgICAgICBfdCA9IGpzb24ubG9hZChvcGVuKGZpcnN0KCdhcmMtYWdpX3Rlc3RfY2hhbGxlbmdlcy5qc29uJykpKQogICAgICAgIGpzb24uZHVtcCh7dDogW3snYXR0ZW1wdF8xJzogW1swXV0sICdhdHRlbXB0XzInOiBbWzBdXX0gZm9yIF8gaW4gX3RbdF1bJ3Rlc3QnXV0gZm9yIHQgaW4gX3R9LCBvcGVuKGYne09VVH0vc3VibWlzc2lvbi5qc29uJywgJ3cnKSkKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZTogbG9nKCdlYXJseSBwbGFjZWhvbGRlciBmYWlsZWQnLCBlKQogICAgbG9nKCdDRkcnLCBqc29uLmR1bXBzKENGRyksICdSRVJVTicsIFJFUlVOLCAnRkFLRScsIEZBS0UpCiAgICBtZGlyID0gTm9uZTsgVE9LID0gTm9uZQogICAgaWYgRkFLRToKICAgICAgICBwb3J0cyA9IFs4MDAxLCA4MDAyXTsgdXNlZCA9IHsnZmFrZSc6IDF9CiAgICBlbGlmIG9zLmVudmlyb24uZ2V0KCdFMl9FWFRfUE9SVCcpOgogICAgICAgIHBvcnRzID0gW2ludChvcy5lbnZpcm9uWydFMl9FWFRfUE9SVCddKV07IHVzZWQgPSB7J2V4dCc6ICd0cHUtdjVlOC1iZjE2J30KICAgICAgICB0cnk6CiAgICAgICAgICAgIGZyb20gdHJhbnNmb3JtZXJzIGltcG9ydCBBdXRvVG9rZW5pemVyCiAgICAgICAgICAgIFRPSyA9IEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKG9zLmVudmlyb25bJ0UyX1RPS19ESVInXSwgdHJ1c3RfcmVtb3RlX2NvZGU9VHJ1ZSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6IGxvZygndG9rZW5pemVyIGxvYWQgZmFpbGVkLCB1c2luZyBidWlsdC1pbiB0ZW1wbGF0ZScsIHN0cihlKVs6MTIwXSkKICAgIGVsc2U6CiAgICAgICAgaW5zdGFsbCgpOyBlbnYoKQogICAgICAgIGNmZ3MgPSBbb3MucGF0aC5kaXJuYW1lKHApIGZvciBwIGluIGdsb2IuZ2xvYihmJ3tJTn0vKiovY29uZmlnLmpzb24nLCByZWN1cnNpdmU9VHJ1ZSldCiAgICAgICAgbWRpciA9IFtjIGZvciBjIGluIGNmZ3MgaWYgJ2dwdHEnIGluIGMubG93ZXIoKSBvciAndzRhMTYnIGluIGMubG93ZXIoKV0gb3IgY2ZncwogICAgICAgIG1kaXIgPSBtZGlyWzBdOyBsb2coJ21vZGVsJywgbWRpcik7IHNoKGYnbHMgLWxhIHttZGlyfSB8IGhlYWQgLTMwJykKICAgICAgICBuZ3B1ID0gaW50KHN1YnByb2Nlc3MucnVuKCdudmlkaWEtc21pIC1MIHwgd2MgLWwnLCBzaGVsbD1UcnVlLCBjYXB0dXJlX291dHB1dD1UcnVlLCB0ZXh0PVRydWUpLnN0ZG91dC5zdHJpcCgpIG9yIDQpCiAgICAgICAgZ3JvdXBzID0gWycsJy5qb2luKHN0cihnKSBmb3IgZyBpbiByYW5nZShpLCBpICsgVFApKSBmb3IgaSBpbiByYW5nZSgwLCBuZ3B1LCBUUCldCiAgICAgICAgcG9ydHMgPSBbODAwMSArIGkgZm9yIGkgaW4gcmFuZ2UobGVuKGdyb3VwcykpXQogICAgICAgIHVzZWQgPSBOb25lCiAgICAgICAgZm9yIHByb2YgaW4gUFJPRklMRVM6CiAgICAgICAgICAgIGxvZygnc3RhcnRpbmcgcHJvZmlsZScsIHByb2YsIGdyb3VwcykKICAgICAgICAgICAgcHJvY3MgPSBbc3RhcnQobWRpciwgZywgcCwgcHJvZikgZm9yIGcsIHAgaW4gemlwKGdyb3VwcywgcG9ydHMpXQogICAgICAgICAgICBpZiB3YWl0X3VwKHByb2NzLCBwb3J0cyk6IHVzZWQgPSBwcm9mOyBicmVhawogICAgICAgICAgICBsb2coJ3Byb2ZpbGUgZmFpbGVkJyk7IFtwLmtpbGwoKSBmb3IgcCBpbiBwcm9jc107IHRpbWUuc2xlZXAoMTUpCiAgICAgICAgICAgIGZvciBwIGluIHBvcnRzOiBzaChmJ3RhaWwgLTI1IHtPVVR9L3NlcnZlcl97cH0ubG9nJykKICAgICAgICBpZiBub3QgdXNlZDogbG9nKCdBTEwgUFJPRklMRVMgRkFJTEVEJyk7IHJldHVybgogICAgICAgIHRyeToKICAgICAgICAgICAgZnJvbSB0cmFuc2Zvcm1lcnMgaW1wb3J0IEF1dG9Ub2tlbml6ZXIKICAgICAgICAgICAgVE9LID0gQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQobWRpciwgdHJ1c3RfcmVtb3RlX2NvZGU9VHJ1ZSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6IGxvZygndG9rZW5pemVyIGxvYWQgZmFpbGVkLCB1c2luZyBidWlsdC1pbiB0ZW1wbGF0ZScsIHN0cihlKVs6MTIwXSkKICAgIGxvZygnc2VydmVycyB1cCcsIHVzZWQpOyBUX0dFTiA9IHRpbWUudGltZSgpCiAgICBFTkQgPSBUMCArIENGR1snSEFSRF9FTkRfUyddIGlmIFJFUlVOIGVsc2UgVF9HRU4gKyBDRkdbJ0NPTU1JVF9CVURHRVRfUyddCiAgICBsZWZ0ID0gbGFtYmRhOiBFTkQgLSB0aW1lLnRpbWUoKQogICAgdHAgPSBmaXJzdCgnYXJjLWFnaV90ZXN0X2NoYWxsZW5nZXMuanNvbicpOyBldnAgPSBmaXJzdCgnYXJjLWFnaV9ldmFsdWF0aW9uX2NoYWxsZW5nZXMuanNvbicpCiAgICBjb21wID0gdHAgaWYgKFJFUlVOIG9yIG5vdCBldnApIGVsc2UgZXZwCiAgICB0YXNrcyA9IGpzb24ubG9hZChvcGVuKGNvbXApKTsgc29scyA9IHt9IGlmIGNvbXAgPT0gdHAgZWxzZSBqc29uLmxvYWQob3BlbihldnAucmVwbGFjZSgnY2hhbGxlbmdlcycsICdzb2x1dGlvbnMnKSkpCiAgICBpZHMgPSBzb3J0ZWQodGFza3MpOyByYW5kb20uUmFuZG9tKDIwMjYpLnNodWZmbGUoaWRzKTsgaWRzID0gaWRzWzpOVEFTS10KICAgIHN1Yl90YXNrcyA9IGpzb24ubG9hZChvcGVuKHRwKSkgICAjIHN1Ym1pc3Npb24gbXVzdCBjb3ZlciB0aGUgdGVzdCBmaWxlJ3MgdGFza3MKICAgIHN1YiA9IHt0OiBbeydhdHRlbXB0XzEnOiBbWzBdXSwgJ2F0dGVtcHRfMic6IFtbMF1dfSBmb3IgXyBpbiBzdWJfdGFza3NbdF1bJ3Rlc3QnXV0gZm9yIHQgaW4gc3ViX3Rhc2tzfQogICAgbG9jayA9IHRocmVhZGluZy5Mb2NrKCkKICAgIGRlZiB3cml0ZV9zdWIoKToKICAgICAgICB0bXAgPSBmJ3tPVVR9L3N1Ym1pc3Npb24uanNvbi50bXAnOyBqc29uLmR1bXAoc3ViLCBvcGVuKHRtcCwgJ3cnKSk7IG9zLnJlcGxhY2UodG1wLCBmJ3tPVVR9L3N1Ym1pc3Npb24uanNvbicpCiAgICB3cml0ZV9zdWIoKTsgbG9nKCdwbGFjZWhvbGRlciBzdWJtaXNzaW9uIHdyaXR0ZW4nLCBsZW4oc3ViKSwgJ3Rhc2tzOyBzb2x2aW5nJywgbGVuKGlkcyksICd0YXNrcyBmcm9tJywgY29tcCkKICAgIGZtdCA9IGxhbWJkYSBnOiAnXG4nLmpvaW4oJyAnLmpvaW4obWFwKHN0ciwgcikpIGZvciByIGluIGcpCiAgICBkZWYgcHJvbXB0KHQsIHgpOgogICAgICAgIHMgPSAnU29sdmUgdGhpcyBBUkMtQUdJIHB1enpsZS4gRWFjaCBncmlkIGlzIHJvd3Mgb2YgZGlnaXRzIDAtOSAoY29sb3JzKSBzZXBhcmF0ZWQgYnkgc3BhY2VzLlxuRmluZCB0aGUgc2luZ2xlIHJ1bGUgdGhhdCBtYXBzIGV2ZXJ5IHRyYWluaW5nIGlucHV0IHRvIGl0cyBvdXRwdXQsIGNoZWNrIGl0IGFnYWluc3QgYWxsIHBhaXJzLCB0aGVuIGFwcGx5IGl0IHRvIHRoZSB0ZXN0IGlucHV0LlxuXG4nCiAgICAgICAgZm9yIGksIHAgaW4gZW51bWVyYXRlKHRbJ3RyYWluJ10pOiBzICs9IGYnRXhhbXBsZSB7aSsxfSBpbnB1dDpcbntmbXQocFsiaW5wdXQiXSl9XG5FeGFtcGxlIHtpKzF9IG91dHB1dDpcbntmbXQocFsib3V0cHV0Il0pfVxuXG4nCiAgICAgICAgcmV0dXJuIHMgKyBmJ1Rlc3QgaW5wdXQ6XG57Zm10KHhbImlucHV0Il0pfVxuXG5HaXZlIHRoZSBmaW5hbCB0ZXN0IG91dHB1dCBncmlkIGF0IHRoZSBlbmQgaW5zaWRlIG9uZSBgYGAgYmxvY2ssIHJvd3Mgb2Ygc3BhY2Utc2VwYXJhdGVkIGRpZ2l0cywgbm90aGluZyBlbHNlIGluIHRoZSBibG9jay4nCiAgICBkZWYgcGFyc2UodHh0KToKICAgICAgICBmb3IgYiBpbiByZXZlcnNlZChyZS5maW5kYWxsKHInYGBgW2Etel0qXG4oLio/KWBgYCcsIHR4dCBvciAnJywgcmUuUykpOgogICAgICAgICAgICByb3dzID0gW1tpbnQoYykgZm9yIGMgaW4gcmUuZmluZGFsbChyJ1xkJywgbG4pXSBmb3IgbG4gaW4gYi5zdHJpcCgpLnNwbGl0bGluZXMoKSBpZiBsbi5zdHJpcCgpXQogICAgICAgICAgICBpZiByb3dzIGFuZCBhbGwobGVuKHIpID09IGxlbihyb3dzWzBdKSBmb3IgciBpbiByb3dzKSBhbmQgbGVuKHJvd3MpIDw9IDMwIGFuZCBsZW4ocm93c1swXSkgPD0gMzA6IHJldHVybiByb3dzCiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGRlZiBnZW5fcHJlZml4KHB0eHQpOgogICAgICAgICMgZXhhY3QgZ2VuZXJhdGlvbiBwcmVmaXggb2YgdGhlIHNlcnZlZCBtb2RlbCAoc3lzdGVtIGVmZm9ydCBsaW5lICsgdXNlciB0dXJuICsgJzx0aGluaz5cbicpCiAgICAgICAgaWYgVE9LIGlzIG5vdCBOb25lOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBrdyA9IHsncmVhc29uaW5nX2VmZm9ydCc6IENGR1snRUZGT1JUJ119IGlmIENGR1snRUZGT1JUJ10gZWxzZSB7fQogICAgICAgICAgICAgICAgcyA9IFRPSy5hcHBseV9jaGF0X3RlbXBsYXRlKFt7J3JvbGUnOiAndXNlcicsICdjb250ZW50JzogcHR4dH1dLCB0b2tlbml6ZT1GYWxzZSwgYWRkX2dlbmVyYXRpb25fcHJvbXB0PVRydWUsICoqa3cpCiAgICAgICAgICAgICAgICByZXR1cm4gcyBpZiBzLmVuZHN3aXRoKCc8dGhpbms+XG4nKSBlbHNlIHMgKyAnPHRoaW5rPlxuJwogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6IGxvZygnYXBwbHlfY2hhdF90ZW1wbGF0ZSBmYWlsZWQnLCBzdHIoZSlbOjEyMF0pCiAgICAgICAgc3lzbCA9IHsnJzogWEhJR0gsICd4aGlnaCc6IFhISUdILCAnbG93JzogTE9XLCAnbWVkaXVtJzogJyd9W0NGR1snRUZGT1JUJ11dCiAgICAgICAgcmV0dXJuICgoJzx8aW1fc3RhcnR8PnN5c3RlbVxuJyArIHN5c2wgKyAnPHxpbV9lbmR8PlxuJykgaWYgc3lzbCBlbHNlICcnKSArICc8fGltX3N0YXJ0fD51c2VyXG4nICsgcHR4dCArICc8fGltX2VuZHw+XG48fGltX3N0YXJ0fD5hc3Npc3RhbnRcbjx0aGluaz5cbicKICAgIGlmIGlkczoKICAgICAgICBsb2coJ2dlbiBwcmVmaXggaGVhZC90YWlsOicsIHJlcHIoZ2VuX3ByZWZpeCgnUFJPTVBUJylbOjIwMF0pLCAnLi4uJywgcmVwcihnZW5fcHJlZml4KCdQUk9NUFQnKVstNDA6XSkpCiAgICAjIC0tLS0gam9iIHF1ZXVlczogYWxsIGF0dGVtcHQtMCBmaXJzdDsgYXR0ZW1wdC0xIGZvciBpbnB1dHMgd2l0aG91dCBhIG5hdHVyYWwgYW5zd2VyIG5leHQ7IHRoZSByZXN0IGxhc3QKICAgIHEwID0gY29sbGVjdGlvbnMuZGVxdWUoKHRpZCwgaiwgMCkgZm9yIHRpZCBpbiBpZHMgZm9yIGogaW4gcmFuZ2UobGVuKHRhc2tzW3RpZF1bJ3Rlc3QnXSkpKQogICAgcWhpLCBxbG8gPSBjb2xsZWN0aW9ucy5kZXF1ZSgpLCBjb2xsZWN0aW9ucy5kZXF1ZSgpOyBsaXZlMCA9IHNldCgpCiAgICBuX2lucHV0cyA9IGxlbihxMCkKICAgIHJlcyA9IHt9OyB0b2sgPSB7J2MnOiAwLCAncCc6IDB9OyBzdG9wID0gdGhyZWFkaW5nLkV2ZW50KCkKICAgIHN0ID0ge3A6IGRpY3Qoa3Y9MC4wLCB3YWl0PTAuMCwgcnVuPTAuMCwgdD0wLjAsIGFkbT0wLCBsYXN0PTAuMCwga3ZtYXg9MC4wLCBoaT0wLCBwb2xscz0wKSBmb3IgcCBpbiBwb3J0c30KICAgIGRlZiBuZXh0X2pvYigpOgogICAgICAgIHdpdGggbG9jazoKICAgICAgICAgICAgaWYgbGVmdCgpIDwgMzAwOiByZXR1cm4gTm9uZQogICAgICAgICAgICBpZiBxMDogeiA9IHEwLnBvcGxlZnQoKTsgbGl2ZTAuYWRkKHpbOjJdKTsgcmV0dXJuIHoKICAgICAgICAgICAgaWYgcWhpOiByZXR1cm4gcWhpLnBvcGxlZnQoKQogICAgICAgICAgICBpZiBsaXZlMDogcmV0dXJuICd3YWl0JwogICAgICAgICAgICBpZiBxbG86IHJldHVybiBxbG8ucG9wbGVmdCgpCiAgICAgICAgICAgIHJldHVybiBOb25lCiAgICBkZWYgYWRtaXQocG9ydCk6CiAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgaWYgbGVmdCgpIDwgMzAwOiByZXR1cm4gRmFsc2UKICAgICAgICAgICAgbm93ID0gdGltZS50aW1lKCkKICAgICAgICAgICAgd2l0aCBsb2NrOgogICAgICAgICAgICAgICAgcyA9IHN0W3BvcnRdOyBnYXAgPSBDRkdbJ1JBTVBfR0FQJ10gaWYgc1snYWRtJ10gPCBDRkdbJ1BFUiddIGVsc2UgQ0ZHWydNSU5fR0FQJ10KICAgICAgICAgICAgICAgIGZyZXNoID0gbm93IC0gc1sndCddIDwgMjAKICAgICAgICAgICAgICAgIGlmIG5vdyAtIHNbJ2xhc3QnXSA+PSBnYXAgYW5kIChub3QgZnJlc2ggb3IgKHNbJ2t2J10gPCBDRkdbJ0tWX0FETUlUJ10gYW5kIHNbJ3dhaXQnXSA8IDEpKToKICAgICAgICAgICAgICAgICAgICBzWydhZG0nXSArPSAxOyBzWydsYXN0J10gPSBub3c7IHJldHVybiBUcnVlCiAgICAgICAgICAgIHRpbWUuc2xlZXAoMSkKICAgIGRlZiBjYWxsKGpvYiwgcG9ydCwgd2FpdF9zKToKICAgICAgICB0aWQsIGosIGEgPSBqb2IKICAgICAgICBpZiBsZWZ0KCkgPCAzMDA6IHJldHVybiBOb25lCiAgICAgICAgcHR4dCA9IHByb21wdCh0YXNrc1t0aWRdLCB0YXNrc1t0aWRdWyd0ZXN0J11bal0pCiAgICAgICAgbXQgPSBtYXgoMjA0OCwgbWluKE1BWFRPSywgTUFYTEVOIC0gaW50KGxlbihwdHh0KSAvIDEuNikgLSA1MTIpKSAgICMgbmV2ZXIgcmVxdWVzdCBwYXN0IHRoZSBjb250ZXh0IGxpbWl0CiAgICAgICAgbXQgPSBtYXgoMTAyNCwgbWluKG10LCBpbnQoKGxlZnQoKSAtIDI0MCkgKiBDRkdbJ1RQU19TVFJFQU0nXSkpKSAgICAgIyBmaW5pc2ggYmVmb3JlIHRoZSBkZWFkbGluZQogICAgICAgIGJvZHkgPSB7J21vZGVsJzogb3MuZW52aXJvbi5nZXQoJ0UyX01PREVMJywgJ20nKSwgJ21lc3NhZ2VzJzogW3sncm9sZSc6ICd1c2VyJywgJ2NvbnRlbnQnOiBwdHh0fV0sICdtYXhfdG9rZW5zJzogbXQsCiAgICAgICAgICAgICAgICAndGVtcGVyYXR1cmUnOiBDRkdbJ1RFTVBTJ11bYV0sICd0b3BfcCc6IDAuOTUsICd0b3Bfayc6IDIwLCAnc2VlZCc6IDEwMDAgKiBhICsgan0KICAgICAgICBpZiBDRkdbJ0VGRk9SVCddOiBib2R5WydjaGF0X3RlbXBsYXRlX2t3YXJncyddID0geydyZWFzb25pbmdfZWZmb3J0JzogQ0ZHWydFRkZPUlQnXX0KICAgICAgICB0MSA9IHRpbWUudGltZSgpOyB0eHQgPSBOb25lOyByZWFzID0gJyc7IGN0ID0gMDsgZmluID0gJ2VycicKICAgICAgICB0cnk6CiAgICAgICAgICAgIHIgPSBwb3N0KHBvcnQsICcvdjEvY2hhdC9jb21wbGV0aW9ucycsIGJvZHksIG1heCg2MCwgbGVmdCgpIC0gMzApKQogICAgICAgICAgICBjaCA9IHJbJ2Nob2ljZXMnXVswXTsgdHh0ID0gY2hbJ21lc3NhZ2UnXS5nZXQoJ2NvbnRlbnQnKSBvciAnJzsgcmVhcyA9IGNoWydtZXNzYWdlJ10uZ2V0KCdyZWFzb25pbmdfY29udGVudCcpIG9yIGNoWydtZXNzYWdlJ10uZ2V0KCdyZWFzb25pbmcnKSBvciAnJwogICAgICAgICAgICBjdCA9IHJbJ3VzYWdlJ11bJ2NvbXBsZXRpb25fdG9rZW5zJ107IGZpbiA9IGNoLmdldCgnZmluaXNoX3JlYXNvbicpCiAgICAgICAgICAgIHdpdGggbG9jazogdG9rWydjJ10gKz0gY3Q7IHRva1sncCddICs9IHJbJ3VzYWdlJ11bJ3Byb21wdF90b2tlbnMnXQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZTogZmluID0gJ2V4YzonICsgc3RyKGUpWzo4MF0KICAgICAgICBnX25hdCA9IHBhcnNlKHR4dCkKICAgICAgICBnID0gZ19uYXQgb3IgcGFyc2UocmVhc1stNDAwMDpdKQogICAgICAgIGZvcmNlZCA9IEZhbHNlCiAgICAgICAgaWYgZyBpcyBOb25lIGFuZCBmaW4gPT0gJ2xlbmd0aCcgYW5kIHJlYXMgYW5kIGxlZnQoKSA+IDEyMDoKICAgICAgICAgICAgIyBhbnN3ZXIgZm9yY2luZzogY2xvc2UgdGhlIHRoaW5raW5nIGJsb2NrIG9uIHRoZSBzZXJ2ZWQgdGVtcGxhdGUgYW5kIGFzayBvbmx5IGZvciB0aGUgZ3JpZCAofjEwLTE4ayBwcmVmaWxsLCA8PTNrIGRlY29kZSkKICAgICAgICAgICAgZnAgPSBnZW5fcHJlZml4KHB0eHQpICsgcmVhc1stMjQwMDA6XS5zdHJpcCgpICsgJ1xuPC90aGluaz5cblxuVGltZSBpcyB1cC4gRmluYWwgdGVzdCBvdXRwdXQgZ3JpZDpcbmBgYFxuJwogICAgICAgICAgICBmYiA9IHsnbW9kZWwnOiBvcy5lbnZpcm9uLmdldCgnRTJfTU9ERUwnLCAnbScpLCAncHJvbXB0JzogZnAsICdtYXhfdG9rZW5zJzogMzA3MiwgJ3RlbXBlcmF0dXJlJzogMC4yLCAndG9wX3AnOiAwLjk1LCAnc3RvcCc6IFsnYGBgJ10sICdzZWVkJzogNyArIGp9CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHIyID0gcG9zdChwb3J0LCAnL3YxL2NvbXBsZXRpb25zJywgZmIsIG1heCg2MCwgbWluKDkwMCwgbGVmdCgpIC0gMzApKSkKICAgICAgICAgICAgICAgIHQyID0gcjJbJ2Nob2ljZXMnXVswXS5nZXQoJ3RleHQnKSBvciAnJwogICAgICAgICAgICAgICAgZyA9IHBhcnNlKCdgYGBcbicgKyB0Mi5zdHJpcCgnXG4nKSArICdcbmBgYCcpOyBmb3JjZWQgPSBnIGlzIG5vdCBOb25lCiAgICAgICAgICAgICAgICB3aXRoIGxvY2s6IHRva1snYyddICs9IHIyWyd1c2FnZSddWydjb21wbGV0aW9uX3Rva2VucyddCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZTogbG9nKCdmb3JjZSBmYWlsZWQnLCBzdHIoZSlbOjEyMF0pCiAgICAgICAgb2sgPSAoZyA9PSBzb2xzW3RpZF1bal0pIGlmIHNvbHMgZWxzZSBOb25lCiAgICAgICAgd2l0aCBsb2NrOgogICAgICAgICAgICBpZiBnIGlzIG5vdCBOb25lIGFuZCB0aWQgaW4gc3ViIGFuZCBqIDwgbGVuKHN1Ylt0aWRdKToKICAgICAgICAgICAgICAgIHNsb3QgPSAnYXR0ZW1wdF8xJyBpZiBhID09IDAgZWxzZSAnYXR0ZW1wdF8yJwogICAgICAgICAgICAgICAgc3ViW3RpZF1bal1bc2xvdF0gPSBnCiAgICAgICAgICAgICAgICBvdGhlciA9ICdhdHRlbXB0XzInIGlmIGEgPT0gMCBlbHNlICdhdHRlbXB0XzEnCiAgICAgICAgICAgICAgICBpZiBzdWJbdGlkXVtqXVtvdGhlcl0gPT0gW1swXV06IHN1Ylt0aWRdW2pdW290aGVyXSA9IGcgICAjIG5ldmVyIGxlYXZlIGFuIGVtcHR5IGF0dGVtcHQKICAgICAgICAgICAgICAgIHdyaXRlX3N1YigpCiAgICAgICAgICAgIHJlc1sodGlkLCBqLCBhKV0gPSB7J2dyaWQnOiBnLCAnb2snOiBvaywgJ2N0b2snOiBjdCwgJ2Zpbic6IGZpbiwgJ2ZvcmNlZCc6IGZvcmNlZCwgJ25hdHVyYWwnOiBnX25hdCBpcyBub3QgTm9uZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnc2Vjcyc6IHJvdW5kKHRpbWUudGltZSgpIC0gdDEpLCAncG9ydCc6IHBvcnQsICd3YWl0Jzogcm91bmQod2FpdF9zKX0KICAgICAgICAgICAgaWYgYSA9PSAwOgogICAgICAgICAgICAgICAgKHFsbyBpZiAoZmluID09ICdzdG9wJyBhbmQgZ19uYXQgaXMgbm90IE5vbmUpIGVsc2UgcWhpKS5hcHBlbmQoKHRpZCwgaiwgMSkpCiAgICAgICAgbG9nKGYne3RpZH0gdHtqfSBhe2F9IHB7cG9ydH0gZmluPXtmaW59IGZvcmNlZD17Zm9yY2VkfSBjdG9rPXtjdH0gcGFyc2VkPXtnIGlzIG5vdCBOb25lfSBvaz17b2t9IHNlY3M9e3JvdW5kKHRpbWUudGltZSgpLXQxKX0gJwogICAgICAgICAgICBmJ3dhaXQ9e3JvdW5kKHdhaXRfcyl9IGFnZ190cHM9e3Rva1siYyJdL21heCgxLCB0aW1lLnRpbWUoKS1UX0dFTik6LjBmfSBxMD17bGVuKHEwKX0gcWhpPXtsZW4ocWhpKX0gcWxvPXtsZW4ocWxvKX0nKQogICAgICAgIHdpdGggb3BlbihmJ3tPVVR9L2UyX3Jhdy5qc29ubCcsICdhJykgYXMgZjoKICAgICAgICAgICAgZi53cml0ZShqc29uLmR1bXBzKHsndGlkJzogdGlkLCAnaic6IGosICdhJzogYSwgJ2Zpbic6IGZpbiwgJ2ZvcmNlZCc6IGZvcmNlZCwgJ25hdHVyYWwnOiBnX25hdCBpcyBub3QgTm9uZSwgJ2N0b2snOiBjdCwgJ29rJzogb2ssCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ210JzogbXQsICdwb3J0JzogcG9ydCwgJ3Rfc3RhcnQnOiByb3VuZCh0MSAtIFRfR0VOKSwgJ3NlY3MnOiByb3VuZCh0aW1lLnRpbWUoKSAtIHQxKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAndHh0JzogKHR4dCBvciAnJylbLTMwMDA6XSwgJ3JlYXNfdGFpbCc6IHJlYXNbLTYwMDA6XX0pICsgJ1xuJykKICAgICAgICBpZiBDRkdbJ1NBVkVfQ0hBSU5TJ106CiAgICAgICAgICAgIGltcG9ydCBnemlwCiAgICAgICAgICAgIHdpdGggbG9jaywgZ3ppcC5vcGVuKGYne09VVH0vY2hhaW5zLmpzb25sLmd6JywgJ2F0JykgYXMgZjoKICAgICAgICAgICAgICAgIGYud3JpdGUoanNvbi5kdW1wcyh7J3RpZCc6IHRpZCwgJ2onOiBqLCAnYSc6IGEsICdmaW4nOiBmaW4sICdjdG9rJzogY3QsICdvayc6IG9rLCAncmVhc29uaW5nJzogcmVhcywgJ2NvbnRlbnQnOiB0eHQgb3IgJyd9KSArICdcbicpCiAgICAgICAgcmV0dXJuIFRydWUKICAgIGRlZiB3b3JrZXIocG9ydCk6CiAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgeiA9IG5leHRfam9iKCkKICAgICAgICAgICAgaWYgeiBpcyBOb25lOiByZXR1cm4KICAgICAgICAgICAgaWYgeiA9PSAnd2FpdCc6IHRpbWUuc2xlZXAoMyk7IGNvbnRpbnVlCiAgICAgICAgICAgIHR3ID0gdGltZS50aW1lKCkKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgaWYgbm90IGFkbWl0KHBvcnQpOiByZXR1cm4KICAgICAgICAgICAgICAgIGNhbGwoeiwgcG9ydCwgdGltZS50aW1lKCkgLSB0dykKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOiBsb2coJ3dvcmtlciBlcnJvcicsIHJlcHIoZSlbOjE2MF0pCiAgICAgICAgICAgIGZpbmFsbHk6CiAgICAgICAgICAgICAgICB3aXRoIGxvY2s6IGxpdmUwLmRpc2NhcmQoels6Ml0pIGlmIHpbMl0gPT0gMCBlbHNlIE5vbmUKICAgIGRlZiBwb2xsZXIoKToKICAgICAgICBsYXN0X2xvZyA9IDAuMAogICAgICAgIHdoaWxlIG5vdCBzdG9wLmlzX3NldCgpOgogICAgICAgICAgICBsaW5lID0gW10KICAgICAgICAgICAgZm9yIHAgaW4gcG9ydHM6CiAgICAgICAgICAgICAgICBzLCBtID0gZ2V0KGYnaHR0cDovLzEyNy4wLjAuMTp7cH0vbWV0cmljcycpCiAgICAgICAgICAgICAgICBpZiBzICE9IDIwMDogY29udGludWUKICAgICAgICAgICAgICAgIGQgPSB7fQogICAgICAgICAgICAgICAgZm9yIGwgaW4gbS5zcGxpdGxpbmVzKCk6CiAgICAgICAgICAgICAgICAgICAgaWYgbm90IGwuc3RhcnRzd2l0aCgndmxsbTonKTogY29udGludWUKICAgICAgICAgICAgICAgICAgICBuYW1lID0gbC5zcGxpdCgneycsIDEpWzBdLnNwbGl0KCcgJywgMSlbMF0KICAgICAgICAgICAgICAgICAgICBpZiBuYW1lIGluIFdBTlQ6CiAgICAgICAgICAgICAgICAgICAgICAgIHRyeTogZFtuYW1lXSA9IGQuZ2V0KG5hbWUsIDAuMCkgKyBmbG9hdChsLnJzcGxpdCgnICcsIDEpWzFdKQogICAgICAgICAgICAgICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOiBwYXNzCiAgICAgICAgICAgICAgICB3aXRoIGxvY2s6CiAgICAgICAgICAgICAgICAgICAgeCA9IHN0W3BdOyB4Wyd0J10gPSB0aW1lLnRpbWUoKTsgeFsncG9sbHMnXSArPSAxCiAgICAgICAgICAgICAgICAgICAgeFsna3YnXSA9IGQuZ2V0KCd2bGxtOmt2X2NhY2hlX3VzYWdlX3BlcmMnLCB4WydrdiddKTsgeFsnd2FpdCddID0gZC5nZXQoJ3ZsbG06bnVtX3JlcXVlc3RzX3dhaXRpbmcnLCAwLjApCiAgICAgICAgICAgICAgICAgICAgeFsncnVuJ10gPSBkLmdldCgndmxsbTpudW1fcmVxdWVzdHNfcnVubmluZycsIDAuMCk7IHhbJ2t2bWF4J10gPSBtYXgoeFsna3ZtYXgnXSwgeFsna3YnXSk7IHhbJ2hpJ10gKz0geFsna3YnXSA+PSAwLjk3CiAgICAgICAgICAgICAgICBsaW5lLmFwcGVuZChmJ3twfSBrdj17eFsia3YiXTouM2Z9IHJ1bj17eFsicnVuIl06LjBmfSB3YWl0PXt4WyJ3YWl0Il06LjBmfSBnZW49e2QuZ2V0KCJ2bGxtOmdlbmVyYXRpb25fdG9rZW5zX3RvdGFsIiwgMCk6LjBmfSAnCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmJ2FjYz17ZC5nZXQoInZsbG06c3BlY19kZWNvZGVfbnVtX2FjY2VwdGVkX3Rva2Vuc190b3RhbCIsIDApOi4wZn0gZHJmPXtkLmdldCgidmxsbTpzcGVjX2RlY29kZV9udW1fZHJhZnRfdG9rZW5zX3RvdGFsIiwgMCk6LjBmfScpCiAgICAgICAgICAgIGlmIHRpbWUudGltZSgpIC0gbGFzdF9sb2cgPj0gNjA6CiAgICAgICAgICAgICAgICBsYXN0X2xvZyA9IHRpbWUudGltZSgpCiAgICAgICAgICAgICAgICB3aXRoIG9wZW4oZid7T1VUfS9tZXRyaWNzLmxvZycsICdhJykgYXMgZjogZi53cml0ZShmJ3t0aW1lLnRpbWUoKS1UX0dFTjouMGZ9ICcgKyAnIHwgJy5qb2luKGxpbmUpICsgJ1xuJykKICAgICAgICAgICAgc3RvcC53YWl0KDQpCiAgICB0aHJlYWRpbmcuVGhyZWFkKHRhcmdldD1wb2xsZXIsIGRhZW1vbj1UcnVlKS5zdGFydCgpCiAgICB0aHMgPSBbdGhyZWFkaW5nLlRocmVhZCh0YXJnZXQ9d29ya2VyLCBhcmdzPShwLCksIGRhZW1vbj1UcnVlKSBmb3IgcCBpbiBwb3J0cyBmb3IgXyBpbiByYW5nZShDRkdbJ1BFUiddKV0KICAgIFt0LnN0YXJ0KCkgZm9yIHQgaW4gdGhzXQogICAgd2hpbGUgYW55KHQuaXNfYWxpdmUoKSBmb3IgdCBpbiB0aHMpIGFuZCBsZWZ0KCkgPiAtNjA6IHRpbWUuc2xlZXAoNSkKICAgIHN0b3Auc2V0KCk7IHdhbGwgPSB0aW1lLnRpbWUoKSAtIFRfR0VOCiAgICB3aXRoIGxvY2s6IHdyaXRlX3N1YigpCiAgICBzY29yZSA9IDAuMDsgcGVyID0ge30KICAgIGZvciB0aWQgaW4gKGlkcyBpZiBzb2xzIGVsc2UgW10pOgogICAgICAgIG4gPSBsZW4odGFza3NbdGlkXVsndGVzdCddKTsgcyA9IHN1bShhbnkoKHJlcy5nZXQoKHRpZCwgaiwgYSkpIG9yIHt9KS5nZXQoJ29rJykgZm9yIGEgaW4gcmFuZ2UoTkFUVCkpIGZvciBqIGluIHJhbmdlKG4pKSAvIG4KICAgICAgICBwZXJbdGlkXSA9IHM7IHNjb3JlICs9IHMKICAgIHAxID0gc3VtKHN1bShib29sKChyZXMuZ2V0KCh0aWQsIGosIDApKSBvciB7fSkuZ2V0KCdvaycpKSBmb3IgaiBpbiByYW5nZShsZW4odGFza3NbdGlkXVsndGVzdCddKSkpIC8gbGVuKHRhc2tzW3RpZF1bJ3Rlc3QnXSkgZm9yIHRpZCBpbiBpZHMpIGlmIHNvbHMgZWxzZSBOb25lCiAgICB2YWxzID0gbGlzdChyZXMudmFsdWVzKCkpOyBmaW5zID0gY29sbGVjdGlvbnMuQ291bnRlcihzdHIodlsnZmluJ10pWzoxMl0gZm9yIHYgaW4gdmFscykKICAgIHJlcCA9IHsnY2ZnJzogQ0ZHLCAncHJvZmlsZSc6IHVzZWQsICd0cCc6IFRQLCAnbWF4dG9rJzogTUFYVE9LLCAnbl90YXNrcyc6IGxlbihpZHMpLCAnbl9pbnB1dHMnOiBuX2lucHV0cywgJ2RvbmUnOiBsZW4odmFscyksCiAgICAgICAgICAgJ2RvbmVfYTAnOiBzdW0oMSBmb3IgayBpbiByZXMgaWYga1syXSA9PSAwKSwgJ2RvbmVfYTEnOiBzdW0oMSBmb3IgayBpbiByZXMgaWYga1syXSA9PSAxKSwKICAgICAgICAgICAncGFzczInOiBzY29yZSAvIGxlbihpZHMpIGlmIHNvbHMgYW5kIGlkcyBlbHNlIE5vbmUsICdwYXNzMSc6IHAxIC8gbGVuKGlkcykgaWYgc29scyBhbmQgaWRzIGVsc2UgTm9uZSwKICAgICAgICAgICAnZm9yY2VkX3BhcnNlZCc6IHN1bShib29sKHYuZ2V0KCdmb3JjZWQnKSkgZm9yIHYgaW4gdmFscyksICdmb3JjZWRfb2snOiBzdW0oYm9vbCh2LmdldCgnZm9yY2VkJykgYW5kIHYuZ2V0KCdvaycpKSBmb3IgdiBpbiB2YWxzKSwKICAgICAgICAgICAnYWdnX3Rwcyc6IHRva1snYyddIC8gd2FsbCwgJ2N0b2snOiB0b2tbJ2MnXSwgJ3dhbGwnOiB3YWxsLCAncGFyc2VkJzogc3VtKHZbJ2dyaWQnXSBpcyBub3QgTm9uZSBmb3IgdiBpbiB2YWxzKSwgJ2ZpbmlzaCc6IGRpY3QoZmlucyksCiAgICAgICAgICAgJ3NlcnZlcnMnOiB7cDoge2s6IHN0W3BdW2tdIGZvciBrIGluICgnYWRtJywgJ2t2bWF4JywgJ2hpJywgJ3BvbGxzJyl9IGZvciBwIGluIHBvcnRzfSwgJ3Blcl90YXNrJzogcGVyfQogICAganNvbi5kdW1wKHJlcCwgb3BlbihmJ3tPVVR9L2UyX3JlcG9ydC5qc29uJywgJ3cnKSwgaW5kZW50PTEpCiAgICBsb2coJ1JFUE9SVCcsIGpzb24uZHVtcHMoe2s6IHYgZm9yIGssIHYgaW4gcmVwLml0ZW1zKCkgaWYgayAhPSAncGVyX3Rhc2snfSkpCiAgICBpZiBub3QgRkFLRSBhbmQgbm90IG9zLmVudmlyb24uZ2V0KCdFMl9FWFRfUE9SVCcpOiBzaChmJ2dyZXAgLWggInRocm91Z2hwdXRcXHxTcGVjdWxhdGl2ZVxcfGFjY2VwdGFuY2UiIHtPVVR9L3NlcnZlcl8qLmxvZyB8IHRhaWwgLTYnKQppZiBfX25hbWVfXyA9PSAnX19tYWluX18nOgogICAgbWFpbigpCg=="
open("/tmp/e3t_client.py", "wb").write(__import__("base64").b64decode(CLIENT_B64))
PHASES = [
    ("low16k_120", dict(E2_EFFORT="low", E2_COMMIT_MODE="census", E2_COMMIT_NTASK="120", E2_COMMIT_MAXTOK="16384", E2_COMMIT_BUDGET_S="9000")),
    ("xhigh48k_60", dict(E2_EFFORT="", E2_COMMIT_MODE="bench", E2_COMMIT_BUDGET_S="12600")),
]
for name, ph in PHASES:
    out = WORK / name; out.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, E2_EXT_PORT=str(PORT), E2_TOK_DIR=str(model_path), E2_MODEL=CFG["served_model_name"], E2_API_KEY=CFG["api_key"],
             E2_OUT=str(out), E2_PER="16", E2_RAMP_GAP="5", E2_MIN_GAP="1", E2_KV_ADMIT="0.95", **ph)
    banner(7, f"E3T phase {name}")
    with open(out / "client.log", "w") as f:
        rc = subprocess.run([sys.executable, "/tmp/e3t_client.py"], env=e, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"   phase {name} rc={rc}"); subprocess.run(f"grep REPORT {out}/client.log | cut -c1-1500", shell=True)
    if server.poll() is not None:
        server_died(server, "stopped", reason="server-exit"); break
server.terminate()
sys.exit(0)
