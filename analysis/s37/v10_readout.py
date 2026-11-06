"""V10 readout (written 4 Nov 2026 ~16:30 UTC before T1 and T1c exist; committed with the seal). Scores sealed/v10_supply_prediction.txt
exactly as written. Usage: python3 v10_readout.py [T1 benchmark.json(.gz)] [T1c benchmark.json(.gz)]
(defaults data/runs/t1_fb25_nospec.benchmark.json.gz and data/runs/t1c_fb25_ctl.benchmark.json.gz)."""
import gzip, json, re, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
S_CTL = 4.5585e6        # mean total generated tokens of the four FB25 controls (4.601M, 4.582M, 4.562M, 4.489M)
L1, BAND1 = 102.8, (92.7, 112.8)            # control at r = 1 and its 90% band (sealed table, last row)
# sealed P-V10-1 table: r, predicted levels, 90% band
TAB = [(0.40, 48.2, 41.6, 54.9), (0.45, 57.8, 50.4, 65.1), (0.50, 65.0, 57.2, 72.8), (0.55, 69.2, 61.2, 77.3),
       (0.60, 77.2, 68.8, 85.7), (0.65, 81.5, 72.3, 90.7), (0.70, 85.5, 76.2, 94.8), (0.75, 88.2, 79.2, 97.3),
       (0.80, 91.8, 81.7, 101.8), (0.85, 95.2, 83.9, 106.6), (0.90, 97.2, 86.9, 107.6), (0.95, 101.0, 91.3, 110.7),
       (1.00, 102.8, 92.7, 112.8)]
def load(p):
    op = gzip.open if str(p).endswith('.gz') else open
    return json.load(op(p, 'rt'))['game_runs']
def summary(p):
    V = load(p)
    tok = [int(re.search(r'tokens=(\d+)', g.get('solver_note') or 'tokens=0').group(1)) for g in V]
    lev = sum(g['levels_completed'] for g in V); won = sum(g['levels_completed'] >= g['number_of_levels'] for g in V)
    valid = len(V) == 25 and all(t > 0 for t in tok)
    return dict(n=len(V), tokens=sum(tok), levels=lev, won=won, valid=valid)
def interp(r):
    rs = [t[0] for t in TAB]
    return tuple(float(np.interp(r, rs, [t[i] for t in TAB])) for i in (1, 2, 3))
p1 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'data' / 'runs' / 't1_fb25_nospec.benchmark.json.gz'
p2 = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'data' / 'runs' / 't1c_fb25_ctl.benchmark.json.gz'
t1 = summary(p1)
print(f'T1 {p1.name}: games {t1["n"]}, tokens {t1["tokens"]/1e6:.3f}M, levels {t1["levels"]}, completed {t1["won"]}, '
      f'VALID {t1["valid"]} (also check the kernel log for server crashes)')
r = t1['tokens'] / S_CTL
print(f'measured supply ratio r = {r:.3f}')
if not t1['valid'] or r < 0.40 or r > 1.00:
    print('P-V10-1 UNTESTED (invalid run or r outside 0.40-1.00)')
else:
    L, lo, hi = interp(r)
    v = 'PASS' if lo <= t1['levels'] <= hi else ('ABOVE (not token-limited at this supply)' if t1['levels'] > hi
                                                 else 'BELOW (slower serving costs more than its tokens)')
    print(f'P-V10-1 total levels {t1["levels"]} vs L(r) {L:.1f} (90% {lo:.1f}-{hi:.1f}): {v}')
if p2.exists():
    c = summary(p2)
    ok = BAND1[0] <= c['levels'] <= BAND1[1] and abs(c['tokens'] / S_CTL - 1) <= 0.05
    print(f'T1c {p2.name}: games {c["n"]}, tokens {c["tokens"]/1e6:.3f}M ({c["tokens"]/S_CTL:.3f} of control mean), '
          f'levels {c["levels"]}, completed {c["won"]}, VALID {c["valid"]}')
    print(f'P-V10-2 drift check (levels {BAND1[0]}-{BAND1[1]}, tokens within 5%): {"no drift" if ok else "DRIFT"}')
    if not ok and c['valid'] and t1['valid']:
        r2 = t1['tokens'] / c['tokens']
        if 0.40 <= r2 <= 1.00:
            L2, lo2, hi2 = (x * c['levels'] / L1 for x in interp(r2))
            print(f'  rescaled to T1c: r\' {r2:.3f}, L\' {L2:.1f} (90% {lo2:.1f}-{hi2:.1f}); T1 {t1["levels"]}: '
                  f'{"inside" if lo2 <= t1["levels"] <= hi2 else "outside"} (flagged: drift)')
else:
    print('T1c output not found: P-V10-2 not scored yet')
