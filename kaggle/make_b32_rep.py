"""B32 (S34, D87): byte-identical replicates of B31-SPEC2-FB25 and B28-CTL-FB25 under new slugs.
Only kernel-metadata id/title/code_file change; the notebook bytes are copied unchanged."""
import json, os, shutil, sys
REPS = [("b31_spec2_fb25", "b32_spec2_r2"), ("b31_spec2_fb25", "b32_spec2_r3"),
        ("b28_ctl_fb25", "b32_ctl_r2"), ("b28_ctl_fb25", "b32_ctl_r3")]
root = os.path.dirname(os.path.abspath(__file__))
for src, dst in REPS:
    s, d = os.path.join(root, src), os.path.join(root, dst)
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    meta = json.load(open(os.path.join(s, "kernel-metadata.json")))
    slug = "arc3-" + dst.replace("_", "-")
    shutil.copyfile(os.path.join(s, meta["code_file"]), os.path.join(d, dst + ".ipynb"))
    meta.update(id="shreekumarshah/" + slug, title=slug, code_file=dst + ".ipynb")
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    print("built", d, "->", meta["id"])
