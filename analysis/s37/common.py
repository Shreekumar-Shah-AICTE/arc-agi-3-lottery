"""Shared loaders for the paper analyses (run from any directory)."""
import csv, gzip, io, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CTL = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']          # B12 harness, control config
SPEC2 = ['b31_spec2_fb25', 'b32_spec2_r2', 'b32_spec2_r3']             # speculative depth 2 (D87 NOISE)
# Runs that did not play all 25 games: Appendix A rows 31-32 (lf52, and lf52 + su15, never admitted) and the occupancy
# run of row 30 (lf52 never admitted, D84; found again on 4 Nov while building Appendix B). Their level totals are not
# comparable with full runs, so run-level comparisons and Fig. 3 leave them out.
INVALID = ('b31_spec2_fb25', 'b31_spec1_fb25', 'b29_occ_fb25')
SPEC2_VALID = [r for r in SPEC2 if r not in INVALID]
B12_LB = [33.37, 29.69, 28.24, 26.76, 26.21]                            # 5 scored runs of arc3-b12-x6dprime v1

# The paper's 39 development runs (Appendix B), frozen on 4 Nov 2026. Runs added later (the 2x-budget V5 run v5_fb50,
# the lever 5 runs t1_fb25_nospec and t1c_fb25_ctl) have their own scripts, so every script that scans data/runs
# uses dev_run_files() and a new download cannot change the paper's numbers.
DEV_RUNS = ('b10_x9', 'b12_fb25', 'b12_ht', 'b12_ht2', 'b12_len', 'b12_sdt', 'b12_sdt_r2', 'b12_x7b', 'b12d25', 'b12d25_r2',
            'b13_x6d16', 'b15_vllm', 'b15_vllm_r2', 'b15c_fb25', 'b1_franzen', 'b1c_ctrl', 'b23_d27', 'b23_tc27', 'b24_g4',
            'b28_ctl_fb25', 'b28_tc_fb25', 'b29_occ_fb25', 'b2t_spec4', 'b30_safe', 'b31_spec1_fb25', 'b31_spec2_fb25',
            'b32_ctl_r2', 'b32_ctl_r3', 'b32_spec2_r2', 'b32_spec2_r3', 'b3_x2', 'b4_x6', 'b5_x2ho', 'b5c_ctrlho', 'b7_x6ho',
            'b7c_ctrlho2', 'b8_x7', 'b9_x8', 'f1_forklab')

# The 16 B12-family development runs (control plus variants judged NOISE or null; same list as v2_variance_family.py).
B12_FAMILY = ('b12_fb25', 'b12_ht', 'b12_ht2', 'b12_len', 'b12_sdt', 'b12_sdt_r2', 'b12_x7b', 'b15c_fb25', 'b28_ctl_fb25',
              'b29_occ_fb25', 'b31_spec1_fb25', 'b31_spec2_fb25', 'b32_ctl_r2', 'b32_ctl_r3', 'b32_spec2_r2', 'b32_spec2_r3')

def dev_run_files():
    """The 39 development-run files, in file-name order (the order the earlier glob gave)."""
    fs = [HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz' for r in DEV_RUNS]
    missing = [f.name for f in fs if not f.exists()]
    assert len(fs) == 39 and not missing, missing
    return sorted(fs, key=lambda f: f.name)

def load_run(name):
    """Per game: (final_score, levels_completed, number_of_levels)."""
    with gzip.open(HERE / 'data' / 'runs' / f'{name}.benchmark.json.gz', 'rt') as f:
        d = json.load(f)
    return {g['game_id'].split('-')[0]: (g['final_score'], g['levels_completed'], g['number_of_levels']) for g in d['game_runs']}

def load_lb(track='arc-agi-3', which='final'):
    # Deterministic choice (4 Nov): the final board (file name contains 'final') unless `which` names another stamp.
    files = sorted((HERE / 'data' / 'lb').glob(f'{track}_public_*.csv.gz'))
    pick = [f for f in files if which in f.name] or files[-1:]
    p = pick[-1]
    with gzip.open(p, 'rt') as f:
        rows = list(csv.DictReader(f))
    return sorted((float(r['Score']) for r in rows if r.get('Score')), reverse=True)
