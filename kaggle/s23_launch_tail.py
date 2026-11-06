# ---- S23: dense Qwen3.8-27B FP8 (qwen3_5) target with its own MTP head; fallback ladder; fail fast ----
target_cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())
q = target_cfg.get("quantization_config", {})
assert target_cfg.get("architectures", [""])[0] == "Qwen3_5ForConditionalGeneration", target_cfg.get("architectures")
assert q.get("quant_method") == "fp8", q.get("quant_method")
print("#S23_TARGET", MODEL_DIR, "| arch", target_cfg["architectures"][0], "| mtp layers", target_cfg.get("text_config", {}).get("mtp_num_hidden_layers"), flush=True)

graph_bs = sorted({1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})
graph_bs = [n for n in graph_bs if n <= CFG["CUDAGRAPH_MAXBS"]]

def build_args(spec, memfrac, cuda_graph=True):
    a = [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",
        "--model-loader-extra-config", '{"enable_multithread_load":false}',
        "--served-model-name", str(CFG["SERVED_NAME"]), "--host", "127.0.0.1",
        "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
        "--dtype", "bfloat16", "--quantization", "fp8",
        "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(memfrac),
        "--context-length", str(CFG["CTX"]), "--page-size", "64",
        "--max-running-requests", str(CFG["MAXREQ"]), "--chunked-prefill-size", str(CFG["CHUNK"]),
        "--max-prefill-tokens", str(CFG["MAX_PREFILL"]),
        "--mamba-ssm-dtype", CFG["SSM_DTYPE"], "--max-mamba-cache-size", str(CFG["MAMBA_CACHE"]),
        "--mamba-radix-cache-strategy", CFG["MAMBA_RADIX"], "--mamba-track-interval", "64",
        "--mamba-backend", CFG["LINEAR_BACKEND"],
        "--linear-attn-decode-backend", CFG["LINEAR_BACKEND"],
        "--linear-attn-prefill-backend", CFG["LINEAR_BACKEND"],
        "--trust-remote-code",
        "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
        "--default-chat-template-kwargs", '{"preserve_thinking":true}',
        "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
        "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]
    if cuda_graph:
        a += ["--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"]), "--cuda-graph-bs-decode", *map(str, graph_bs)]
    else:
        a += ["--disable-cuda-graph"]
    if CFG["PREFETCH_CHECKPOINTS"]: a += ["--weight-loader-prefetch-checkpoints"]
    if CFG["DROP_CACHE_AFTER_LOAD"]: a += ["--weight-loader-drop-cache-after-load"]
    tmpl = Path(MODEL_DIR, "chat_template.jinja")
    if tmpl.is_file(): a += ["--chat-template", str(tmpl)]
    if not CFG["AUTOTUNE"]: a += ["--disable-flashinfer-autotune"]
    if spec:
        if CFG["GDN_MTP_CACHE_MODE"]: a += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]
        a += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(CFG["SPEC_STEPS"]),
            "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1),
            "--speculative-draft-model-path", MODEL_DIR,
            "--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],
            "--speculative-accept-threshold-single", str(CFG["SPEC_ACCEPT_SINGLE"]),
            "--speculative-accept-threshold-acc", str(CFG["SPEC_ACCEPT_ACC"])]
    return a

LADDER = [("A_mtp", dict(spec=CFG["SPEC"], memfrac=CFG["MEMFRAC"])),
          ("B_nospec", dict(spec=False, memfrac=CFG["MEMFRAC"])),
          ("C_safe", dict(spec=False, memfrac=0.86, cuda_graph=False))]
if not CFG["SPEC"]: LADDER = LADDER[1:]

def show_log_tail(offset, n=20000):
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2); f.seek(max(offset, f.tell()-n)); print(f.read().decode(errors="replace"))
    except Exception as exc: print("log tail failed", exc)

def kill_group(p):
    import signal
    try: os.killpg(p.pid, signal.SIGKILL)
    except Exception: pass
    try: p.wait(timeout=30)
    except Exception: pass
    time.sleep(10)

precache_model_thread.start()
url = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health"
server_ready, S23_RUNG, proc = False, None, None
for rung, kw in LADDER:
    args = build_args(**kw)
    print(f"\n#S23_RUNG {rung} start at {int(time.time()-NOTEBOOK_START_TIME)}s\n" + shlex.join(args) + "\n", flush=True)
    log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
    with open(LOG, "ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    t0, last_report, died = time.time(), -30, False
    while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
        if proc.poll() is not None:
            died = True; break
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    server_ready = True; break
        except Exception:
            pass
        elapsed = int(time.time()-t0)
        if elapsed-last_report >= 30:
            print(f"  {rung} {elapsed}s ...", flush=True); last_report = elapsed
        time.sleep(3)
    if server_ready:
        S23_RUNG = rung
        print(f"\n#S23_READY rung={rung} after {int(time.time()-t0)}s (notebook {int(time.time()-NOTEBOOK_START_TIME)}s)", flush=True)
        break
    if died:
        print(f"#S23_RUNG_FAIL {rung} exit={proc.returncode} after {int(time.time()-t0)}s", flush=True)
        show_log_tail(log_offset); kill_group(proc); continue
    # deadline while still loading: release the benchmark as B12 does
    S23_RUNG = rung + "_loading"
    print(f"#S23_DEADLINE rung={rung} still loading at {int(time.time()-NOTEBOOK_START_TIME)}s", flush=True)
    break

if S23_RUNG is None:
    raise RuntimeError("S23: every serving rung failed; stopping the notebook to save GPU quota")

def _post(path, payload, timeout=300):
    rq = urllib.request.Request(f"http://127.0.0.1:{SERVED_MODEL_PORT}{path}", data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.loads(r.read())

if server_ready:
    try:
        t1 = time.time()
        out = _post("/v1/chat/completions", {"model": CFG["SERVED_NAME"], "max_tokens": 64, "temperature": 0,
              "messages": [{"role": "user", "content": "Answer in one short sentence: what is 2 + 2?"}],
              "chat_template_kwargs": {"enable_thinking": False}})
        print("#S23_SMOKE", repr(out["choices"][0]["message"].get("content"))[:200], f"{time.time()-t1:.1f}s", flush=True)
    except Exception as exc:
        print("#S23_SMOKE failed", repr(exc)[:500], flush=True)
    try:
        def _tok_count(messages):
            req = dict(model=CFG["SERVED_NAME"], messages=messages, add_generation_prompt=True,
                       chat_template_kwargs={"preserve_thinking": True})
            r = _post("/v1/tokenize", req, timeout=60)
            return int(r.get("count", len(r.get("tokens", []))))
        marker = " ".join(["zebra quartz"] * 40)
        hist = lambda key: [{"role": "user", "content": "Board A."},
                            dict({"role": "assistant", "content": "Moving up."}, **({key: marker} if key else {})),
                            {"role": "user", "content": "Board B."}]
        counts = {k or "none": _tok_count(hist(k)) for k in (None, "reasoning_content", "reasoning")}
        renders = [k for k in ("reasoning_content", "reasoning") if counts[k] >= counts["none"] + 40]
        if os.environ.get("ARC3_REASONING_HISTORY_KEY", "") not in renders and renders:
            os.environ["ARC3_REASONING_HISTORY_KEY"] = renders[0]
        print(f"#S23_REASONING_PROBE counts={counts} renders={renders} key={os.environ.get('ARC3_REASONING_HISTORY_KEY')}", flush=True)
    except Exception as exc:
        print(f"#S23_REASONING_PROBE failed: {exc!r}"[:500], flush=True)
    try:
        lines = Path(LOG).read_text(errors="replace").splitlines()
        hits = [l for l in lines if any(k in l for k in ("KV Cache", "max_total_num_tokens", "Mamba Cache", "Load weight end", "max_running_requests", "avail mem"))]
        for line in hits[:40]:
            print("serve>", line[:300])
    except Exception:
        pass
