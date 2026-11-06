Backups of Kaggle run outputs needed to rebuild analyses if the sandbox is wiped.
- runs/<run>.benchmark.json.gz: taaf benchmark.json (full per-action histories) for each ARC-AGI-3 commit run. b3v2 = X2 replicate (the b3 v1 file was overwritten, lost).
  Restore: mkdir -p /data/kaggle_ro/out/<run> && gunzip -c data/runs/<run>.benchmark.json.gz > /data/kaggle_ro/out/<run>/benchmark.json
- e1c/: E1c report and raw attempts.
- lab/forklab/franzen-harness-changes.patch = Franzen MS2 harness patch (same as cell 2 of kaggle/b1_franzen/b1.ipynb).
