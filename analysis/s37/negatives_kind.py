"""Kind of each of the 32 negative ledger rows (7 Nov 2026, after the 7 Nov outside reviews asked what the 32 contain).
Used by table_ledger_negatives.py (Appendix A, column Kind) and review_checks_7nov.py (counts for Section 4.5)."""
KIND = {
    # ARC-AGI-2 rows
    'E1b-qwen': 'ARC-AGI-2', 'A1-LB': 'ARC-AGI-2', 'A2-LB': 'ARC-AGI-2', 'E3T': 'ARC-AGI-2', 'E2.1': 'ARC-AGI-2',
    'P1B-DOTS3': 'ARC-AGI-2',
    # identical code resubmitted or rerun: a noise sample, not a change
    'B3v2-X2-rep': 'rerun', 'B1-rerun-LB': 'rerun',
    # simulations or log analyses on CPU, no new GPU run
    'R4-FS2-G0': 'CPU only', 'K1-WASTE': 'CPU only', 'K2-DRAIN': 'CPU only', 'K5-EFFIC': 'CPU only', 'K-SURPRISE': 'CPU only',
    # second row for a change already listed (HT-FB25, G4-D25)
    'HT-FB25-res': 'readout', 'G4-D25-res': 'readout',
}
GPU = 'GPU change'   # every other row: a distinct change to the ARC-AGI-3 agent or its serving, run on the GPU
def kind(ledger_id): return KIND.get(ledger_id, GPU)
