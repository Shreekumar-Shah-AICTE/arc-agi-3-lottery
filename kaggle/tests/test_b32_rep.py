import json, os, hashlib
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPS = [("b31_spec2_fb25", "b32_spec2_r2"), ("b31_spec2_fb25", "b32_spec2_r3"),
        ("b28_ctl_fb25", "b32_ctl_r2"), ("b28_ctl_fb25", "b32_ctl_r3")]
ok = 0; n = 0
for src, dst in REPS:
    sm = json.load(open(os.path.join(root, src, "kernel-metadata.json")))
    dm = json.load(open(os.path.join(root, dst, "kernel-metadata.json")))
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    checks = {
        "notebook bytes identical": h(os.path.join(root, src, sm["code_file"])) == h(os.path.join(root, dst, dm["code_file"])),
        "only id/title/code_file differ": {k: v for k, v in sm.items() if k not in ("id", "title", "code_file")} == {k: v for k, v in dm.items() if k not in ("id", "title", "code_file")},
        "slug == title": dm["id"].split("/")[1] == dm["title"],
        "gpu on, internet off, RTX Pro 6000": dm["enable_gpu"] and not dm["enable_internet"] and dm["machine_shape"] == "NvidiaRtxPro6000",
    }
    for k, v in checks.items():
        n += 1; ok += bool(v); print("PASS" if v else "FAIL", dst, k)
print(f"{ok}/{n} PASS")
