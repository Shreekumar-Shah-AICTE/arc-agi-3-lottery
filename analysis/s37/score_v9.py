"""Score the sealed ARC-AGI-2 private-board prediction V9 (sealed/v9_arc2_private_lb_prediction.txt). Written 4 Nov 2026
~16:55 UTC, before the ARC-AGI-2 private board was published. Definitions are copied from the sealed file:
  C = teams with a final public score in [27, 35) on the 4 Nov 11:16 UTC snapshot; n = its SubmissionCount.
  Spearman uses average ranks for ties (the ordinal-rank variant used in the simulation is printed too).
  Public top 10% of C = S >= 90th percentile of S within C (k teams); private top = the top k of C by private score.
Usage: python3 score_v9.py PUBLIC_FINAL.csv[.gz] PRIVATE.csv[.gz]"""
import csv, gzip, sys
import numpy as np
from score_sealed import read, avg_rank, ord_rank, spearman
LO, HI = 27.0, 35.0
# sealed table: HB fits at each sigma (v9_seal_table.out), means of P1..P4
SIG = [0.30, 0.45, 0.60, 0.81, 1.00, 1.30]
TAB = {'P1': [0.953, 0.895, 0.818, 0.689, 0.518, 0.279], 'P2': [0.792, 0.690, 0.597, 0.483, 0.367, 0.225],
       'P3': [2.442, 2.208, 2.030, 1.577, 1.208, 0.718], 'P4': [0.169, 0.339, 0.482, 0.722, 1.082, 1.558]}
PRIMARY = {'P1': (0.82, 0.65, 0.91), 'P2': (0.60, 0.43, 0.73), 'P3': (2.0, 1.0, 2.3), 'P4': (0.48, 0.29, 0.88)}
LESS_NOISE_IF_ABOVE = {'P1': True, 'P2': True, 'P3': True, 'P4': False}
def implied_sigma(key, x):
    y = np.array(TAB[key]); s = np.array(SIG)
    if y[0] > y[-1]: y, s = y[::-1], s[::-1]                 # make y increasing
    if x < y[0]: return f'{"<" if s[0] < s[-1] else ">"}{s[0]:.2f}', None
    if x > y[-1]: return f'{">" if s[-1] > s[0] else "<"}{s[-1]:.2f}', None
    v = float(np.interp(x, y, s)); return f'{v:.2f}', v
def main(pub_path, prv_path):
    pub = [r for r in read(pub_path) if r.get('Score')]
    prv = {r['TeamId']: float(r['Score']) for r in read(prv_path) if r.get('Score') not in (None, '')}
    S_all = np.array([float(r['Score']) for r in pub]); N_all = np.array([int(r['SubmissionCount']) for r in pub])
    ids = [r['TeamId'] for r in pub]; names = [r['TeamName'] for r in pub]
    have = np.array([i in prv for i in ids]); P_all = np.array([prv.get(i, np.nan) for i in ids])
    inC = (S_all >= LO) & (S_all < HI); m = inC & have
    print(f'public board {len(pub)} teams; C {inC.sum()}; C without a private score {(inC & ~have).sum()}')
    S, N, P = S_all[m], N_all[m].astype(float), P_all[m]; NM = [x for x, k in zip(names, m) if k]
    TOP = S >= np.quantile(S, 0.9); k = int(TOP.sum())
    def stats(rk):
        topP = ord_rank(P) >= len(P) - k
        return dict(P1=spearman(S, P, rk), P2=float((TOP & topP).sum() / k),
                    P3=float(P[N >= 64].mean() - P[N == 1].mean()), P4=float((S - P)[TOP].mean() - (S - P).mean()))
    st, so = stats(avg_rank), stats(ord_rank)
    top40 = (S_all >= 40) & have; p5 = spearman(S_all[top40], P_all[top40])
    print(f'C matched {len(S)} (n=1 {(N == 1).sum()}, n>=64 {(N >= 64).sum()}); k = {k}; public gap {S[N >= 64].mean() - S[N == 1].mean():.2f}')
    inside, side, sig = 0, [], []
    for key in ('P1', 'P2', 'P3', 'P4'):
        pv, lo, hi = PRIMARY[key]; x = st[key]; ins = lo <= x <= hi; inside += ins
        if not ins: side.append('less' if (x > hi) == LESS_NOISE_IF_ABOVE[key] else 'more')
        txt, v = implied_sigma(key, x); sig.append(v)
        print(f'{key} observed {x:.3f} (ordinal {so[key]:.3f}); primary {pv} [{lo}, {hi}] -> {"inside" if ins else "OUTSIDE"}; implied sigma {txt}')
    fails = max(side.count('less'), side.count('more')) >= 2
    print('primary sigma 0.6:', 'PASSES' if inside >= 3 else ('FAILS (' + ('less' if side.count('less') >= 2 else 'more') + ' noise than measured)' if fails else 'mixed'))
    good = [v for v in sig if v is not None]
    if len(good) == 4: r = max(good) / min(good); print(f'implied sigma spread max/min = {r:.2f} -> structure {"CONSISTENT" if r <= 1.5 else "INCONSISTENT"} (rule: <= 1.5); median {np.median(good):.2f}')
    else: print('implied sigma: at least one statistic is outside the table (sigma < 0.30 or > 1.30) -> structure check fails')
    print(f'P5 Spearman among public >= 40 ({top40.sum()} teams): {p5:.3f} -> {"FAIL" if p5 < 0.90 else "pass"} (fails if below 0.90)')
    if 'Shreekumar Shah' in NM:
        j = NM.index('Shreekumar Shah'); print(f'P6 our private percentile in C {(P < P[j]).mean():.3f} (forecast 0.43, 0.14 to 0.77)')
if __name__ == '__main__': main(*sys.argv[1:3])
