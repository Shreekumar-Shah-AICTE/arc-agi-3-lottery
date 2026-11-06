# ---- S24 (D68): Gemma-4-31B-it QAT W4A16 (Google's compressed-tensors checkpoint) + Google's QAT assistant (Frozen-KV MTP) ----
# Fallback ladder A_mtp -> B_nospec -> C_safe; each rung must pass a smoke gate; fail fast if nothing serves.
target_cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())
q = target_cfg.get("quantization_config", {})
assert target_cfg.get("architectures", [""])[0] == "Gemma4ForConditionalGeneration", target_cfg.get("architectures")
assert q.get("quant_method") == "compressed-tensors", q.get("quant_method")
_w = q["config_groups"]["group_0"]["weights"]
assert (_w["num_bits"], _w["group_size"], _w["symmetric"]) == (4, 32, True), _w
_tc = target_cfg["text_config"]
print("#S24_TARGET", MODEL_DIR, "| layers", _tc["num_hidden_layers"], "| sliding", _tc["sliding_window"],
      "| ignore", len(q.get("ignore", [])), flush=True)
draft_cfg = json.loads(Path(DRAFT_MODEL_DIR, "config.json").read_text())
assert draft_cfg.get("architectures", [""])[0] == "Gemma4AssistantForCausalLM", draft_cfg.get("architectures")
print("#S24_DRAFT", DRAFT_MODEL_DIR, "| layers", draft_cfg["text_config"]["num_hidden_layers"],
      "| kv shared", draft_cfg["text_config"].get("num_kv_shared_layers"), flush=True)

# Metadata-only target view (same pattern as B12's draft view; weights are symlinked, never copied).
# 1) The checkpoint keeps the vision tower in BF16 under HF '*.q_proj.linear' names, which SGLang's fused
#    vision QKV / GateUp prefixes do not match; regex ignores keep those layers unquantized.
# 2) Google's 2026-07-09 chat template (fixes tool-calling loops, turn closures, thinking order) replaces the
#    June template shipped with the Kaggle checkpoint.
def prepare_target_view(src, work_root):
    src, work_root = Path(src), Path(work_root)
    cfg = copy.deepcopy(target_cfg)
    ig = cfg["quantization_config"].setdefault("ignore", [])
    for pat in ("re:.*vision_tower\\..*", "re:.*embed_vision\\..*"):
        if pat not in ig: ig.append(pat)
    view = work_root / "target-view"
    if view.exists(): shutil.rmtree(view)
    view.mkdir(parents=True)
    (view / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for p in sorted(src.iterdir()):
        if p.name in ("config.json", "chat_template.jinja") or not p.is_file(): continue
        (view / p.name).symlink_to(p.resolve())
    (view / "chat_template.jinja").write_text(GEMMA4_CHAT_TEMPLATE)
    print("SGLang target view:", view, sorted(x.name for x in view.iterdir()), flush=True)
    return str(view)

TARGET_VIEW = prepare_target_view(MODEL_DIR, PREFIX)
graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})
graph_bs = [n for n in graph_bs if n <= CFG["CUDAGRAPH_MAXBS"]]

def build_args(spec, memfrac, cuda_graph=True, kvdtype=None):
    a = [SGLANG, "serve", "--model-path", TARGET_VIEW, "--load-format", "safetensors",
        "--model-loader-extra-config", '{"enable_multithread_load":false}',
        "--served-model-name", str(CFG["SERVED_NAME"]), "--host", "127.0.0.1",
        "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
        "--dtype", "bfloat16", "--quantization", "compressed-tensors",
        "--attention-backend", "triton",
        "--kv-cache-dtype", kvdtype or CFG["KVDTYPE"], "--mem-fraction-static", str(memfrac),
        "--context-length", str(CFG["CTX"]), "--page-size", "64",
        "--swa-full-tokens-ratio", str(CFG["SWA_RATIO"]),
        "--max-running-requests", str(CFG["MAXREQ"]), "--chunked-prefill-size", str(CFG["CHUNK"]),
        "--max-prefill-tokens", str(CFG["MAX_PREFILL"]),
        "--trust-remote-code",
        "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
        "--reasoning-parser", "gemma4", "--tool-call-parser", "gemma4",
        "--chat-template", str(Path(TARGET_VIEW, "chat_template.jinja")),
        "--default-chat-template-kwargs", '{"preserve_thinking":true}',
        "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
        "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]
    if cuda_graph:
        a += ["--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"]), "--cuda-graph-bs-decode", *map(str, graph_bs)]
    else:
        a += ["--disable-cuda-graph"]
    if CFG["PREFETCH_CHECKPOINTS"]: a += ["--weight-loader-prefetch-checkpoints"]
    if CFG["DROP_CACHE_AFTER_LOAD"]: a += ["--weight-loader-drop-cache-after-load"]
    if not CFG["AUTOTUNE"]: a += ["--disable-flashinfer-autotune"]
    if spec:
        # NEXTN + a Gemma4AssistantForCausalLM draft is promoted to FROZEN_KV_MTP (the assistant reads target KV).
        a += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(CFG["SPEC_STEPS"]),
            "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1),
            "--speculative-draft-model-path", str(DRAFT_MODEL_DIR),
            "--speculative-draft-model-quantization", "unquant",
            "--speculative-accept-threshold-single", str(CFG["SPEC_ACCEPT_SINGLE"]),
            "--speculative-accept-threshold-acc", str(CFG["SPEC_ACCEPT_ACC"])]
    return a

LADDER = [("A_mtp", dict(spec=CFG["SPEC"], memfrac=CFG["MEMFRAC"])),
          ("B_nospec", dict(spec=False, memfrac=CFG["MEMFRAC"])),
          ("C_safe", dict(spec=False, memfrac=0.86, cuda_graph=False, kvdtype="auto"))]
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

def _post(path, payload, timeout=300):
    rq = urllib.request.Request(f"http://127.0.0.1:{SERVED_MODEL_PORT}{path}", data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.loads(r.read())

def _smoke_image_url():
    import base64, io
    from PIL import Image
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    for x in range(16, 48):
        for y in range(16, 48): img.putpixel((x, y), (230, 20, 20))
    img = img.resize((640, 640), Image.NEAREST)  # the harness sends 10x-upscaled boards
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

SMOKE_TOOLS = [{"type": "function", "function": {"name": "python", "description": "Run Python code and return its stdout.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source to run"}},
                   "required": ["code"]}}}]

def smoke_gate(rung):
    """Plain answer, then image + thinking + tool call (the harness's request shape). Returns (plain_ok, tool_ok)."""
    plain_ok = tool_ok = False
    try:
        t1 = time.time()
        out = _post("/v1/chat/completions", {"model": CFG["SERVED_NAME"], "max_tokens": 64, "temperature": 0,
              "messages": [{"role": "user", "content": "Answer in one short sentence: what is 2 + 2?"}],
              "chat_template_kwargs": {"enable_thinking": False}})
        txt = out["choices"][0]["message"].get("content") or ""
        plain_ok = ("4" in txt) or ("four" in txt.lower())
        print(f"#S24_SMOKE {rung} ok={plain_ok}", repr(txt)[:200], f"{time.time()-t1:.1f}s", flush=True)
    except Exception as exc:
        print(f"#S24_SMOKE {rung} failed", repr(exc)[:500], flush=True)
    if not plain_ok:
        return plain_ok, tool_ok
    try:
        url = _smoke_image_url()
    except Exception as exc:
        url = None; print("#S24_TOOL_SMOKE no PIL image:", repr(exc)[:200], flush=True)
    content = ([{"type": "image_url", "image_url": {"url": url}}] if url else []) + [{"type": "text", "text":
        "In one short sentence, say what colour the square in the image is. Then use the python tool to compute "
        "17 * 25 - do not compute it yourself."}]
    payload = {"model": CFG["SERVED_NAME"], "max_tokens": 4096, "temperature": float(os.environ.get("LOCAL_ANALYZER_TEMPERATURE", "1.0")),
               "top_p": 0.95, "top_k": int(os.environ.get("LOCAL_ANALYZER_TOP_K", "64")),
               "messages": [{"role": "user", "content": content}], "tools": SMOKE_TOOLS, "tool_choice": "auto",
               "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
    for attempt in range(3):
        try:
            t1 = time.time()
            out = _post("/v1/chat/completions", payload, timeout=600)
            m = out["choices"][0]["message"]; tcs = m.get("tool_calls") or []
            args = tcs[0]["function"].get("arguments") if tcs else None
            print(f"#S24_TOOL_SMOKE {rung} try={attempt} tool_calls={len(tcs)} args={str(args)[:160]!r} "
                  f"reasoning_chars={len(m.get('reasoning_content') or '')} content={str(m.get('content'))[:200]!r} "
                  f"finish={out['choices'][0].get('finish_reason')} usage={out.get('usage')} {time.time()-t1:.1f}s", flush=True)
            seen = ((m.get("content") or "") + " " + (m.get("reasoning_content") or "")).lower()
            print(f"#S24_VISION_CHECK {rung} try={attempt} says_red={'red' in seen}", flush=True)
            if tcs and tcs[0]["function"].get("name") == "python":
                tool_ok = True; break
        except Exception as exc:
            print(f"#S24_TOOL_SMOKE {rung} try={attempt} failed", repr(exc)[:500], flush=True)
    return plain_ok, tool_ok

precache_model_thread.start()
url = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health"
server_ready, S24_RUNG, proc = False, None, None
for rung, kw in LADDER:
    args = build_args(**kw)
    print(f"\n#S24_RUNG {rung} start at {int(time.time()-NOTEBOOK_START_TIME)}s\n" + shlex.join(args) + "\n", flush=True)
    log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
    with open(LOG, "ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    t0, last_report, died, server_ready = time.time(), -30, False, False
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
        print(f"\n#S24_READY rung={rung} after {int(time.time()-t0)}s (notebook {int(time.time()-NOTEBOOK_START_TIME)}s)", flush=True)
        plain_ok, tool_ok = smoke_gate(rung)
        if plain_ok and tool_ok:
            S24_RUNG = rung
            break
        print(f"#S24_GATE_FAIL {rung} plain={plain_ok} tool={tool_ok}", flush=True)
        show_log_tail(log_offset, 12000); kill_group(proc); server_ready = False
        if plain_ok and rung != "A_mtp":
            # serving works but the harness request shape (image + thinking + tool call) does not: another rung won't fix it
            raise RuntimeError("S24: Gemma tool-call/image smoke failed on a no-spec rung; stopping to save GPU quota")
        continue
    if died:
        print(f"#S24_RUNG_FAIL {rung} exit={proc.returncode} after {int(time.time()-t0)}s", flush=True)
        show_log_tail(log_offset); kill_group(proc); continue
    # deadline while still loading: release the benchmark as B12 does
    S24_RUNG = rung + "_loading"
    print(f"#S24_DEADLINE rung={rung} still loading at {int(time.time()-NOTEBOOK_START_TIME)}s", flush=True)
    break

if S24_RUNG is None:
    raise RuntimeError("S24: every serving rung failed; stopping the notebook to save GPU quota")

if server_ready:
    try:
        def _tok_count(messages):
            req = dict(model=CFG["SERVED_NAME"], messages=messages, add_generation_prompt=True,
                       chat_template_kwargs={"preserve_thinking": True, "enable_thinking": True})
            r = _post("/v1/tokenize", req, timeout=60)
            return int(r.get("count", len(r.get("tokens", []))))
        marker = " ".join(["zebra quartz"] * 40)
        call = [{"id": "c1", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "print(1)"})}}]
        hist = lambda key: [{"role": "user", "content": "Board A."},
                            dict({"role": "assistant", "content": "", "tool_calls": call}, **({key: marker} if key else {})),
                            {"role": "tool", "tool_call_id": "c1", "content": "1"},
                            {"role": "user", "content": "Board B."}]
        counts = {k or "none": _tok_count(hist(k)) for k in (None, "reasoning_content", "reasoning")}
        renders = [k for k in ("reasoning_content", "reasoning") if counts[k] >= counts["none"] + 40]
        if os.environ.get("ARC3_REASONING_HISTORY_KEY", "") not in renders and renders:
            os.environ["ARC3_REASONING_HISTORY_KEY"] = renders[0]
        print(f"#S24_REASONING_PROBE counts={counts} renders={renders} key={os.environ.get('ARC3_REASONING_HISTORY_KEY')}", flush=True)
    except Exception as exc:
        print(f"#S24_REASONING_PROBE failed: {exc!r}"[:500], flush=True)
    try:
        lines = Path(LOG).read_text(errors="replace").splitlines()
        keys = ("KV Cache", "max_total_num_tokens", "swa", "SWA", "Load weight end", "max_running_requests", "avail mem",
                "FROZEN_KV_MTP", "attention backend", "compressed")
        hits = [l for l in lines if any(k in l for k in keys)]
        for line in hits[:50]:
            print("serve>", line[:300])
    except Exception:
        pass
