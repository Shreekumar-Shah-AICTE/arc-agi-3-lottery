"""Score the sealed ARC-AGI-3 private-board prediction (sealed/arc3_private_lb_prediction.txt, SHA-256 2c5b49a9...0ef7).
Written 4 Nov 2026 before the private board was opened by us. Definitions are copied from the sealed file:
  C = teams with final public score in [20, 40); n = SubmissionCount on the final public board.
  Spearman uses average ranks for ties (an ordinal-rank variant, as in v8_sealed_prediction.py, is printed too).
  Top 10% of C on the public board = S >= 90th percentile of S (as in v8); private top = the same number k of teams
  by private rank.  D = mean(private - public) over C teams with n = 1.
Usage: python3 score_sealed.py PUBLIC_FINAL.csv[.gz] PRIVATE.csv[.gz] OUR_SUBMISSIONS.csv"""
import csv, gzip, sys, math
import numpy as np

def read(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8-sig') as fh: return list(csv.DictReader(fh))

def avg_rank(v):
    v = np.asarray(v, float); o = np.argsort(v, kind='mergesort'); r = np.empty(len(v)); r[o] = np.arange(len(v), dtype=float)
    sv = v[o]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]: j += 1
        if j > i: r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r
def ord_rank(v):
    o = np.argsort(v, kind='mergesort'); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
def spearman(a, b, rk=avg_rank): return float(np.corrcoef(rk(a), rk(b))[0, 1])

PRED = {  # name: (H value, H lo, H hi, S value, fail test as text, fail lambda)
    'P1': (0.39, 0.28, 0.57, 0.90, 'above 0.70', lambda x: x > 0.70),
    'P2': (0.30, 0.20, 0.42, 0.71, 'above 0.55', lambda x: x > 0.55),
    'P3': (0.9, 0.0, 1.7, 3.6, 'above 2.5', lambda x: x > 2.5),
    'P4': (2.7, 1.9, 3.3, 0.70, 'below 1.3', lambda x: x < 1.3),
    'P5': (0.89, 0.78, 0.96, None, 'below 0.60', lambda x: x < 0.60),
}
FAMILY = ['56832595', '57052442', '56858951', '56919974', '57153031', '57760408', '57815956', '57904835']
OURS = 'Shreekumar Shah'

def main(pub_path, prv_path, subs_path=None, boot=2000):
    pub = [r for r in read(pub_path) if r.get('Score')]
    prv = {r['TeamId']: float(r['Score']) for r in read(prv_path) if r.get('Score') not in (None, '')}
    S_all = np.array([float(r['Score']) for r in pub]); N_all = np.array([int(r['SubmissionCount']) for r in pub])
    ids = [r['TeamId'] for r in pub]; names = [r['TeamName'] for r in pub]
    have = np.array([i in prv for i in ids])
    P_all = np.array([prv.get(i, np.nan) for i in ids])
    inC = (S_all >= 20) & (S_all < 40)
    print(f'final public board: {len(pub)} teams; C = {inC.sum()} teams; private score missing for {(inC & ~have).sum()} of C '
          f'and {(~have).sum()} overall; private-only teams {len(set(prv) - set(ids))}')
    m = inC & have
    S, N, P = S_all[m], N_all[m].astype(float), P_all[m]; NM = [x for x, k in zip(names, m) if k]
    TOP = S >= np.quantile(S, 0.9); k = int(TOP.sum())
    def stats(S, N, P, TOP, rk=avg_rank):
        k = int(TOP.sum()); topP = ord_rank(P) >= len(P) - k
        G1, G64 = N == 1, N >= 64
        return dict(P1=spearman(S, P, rk), P2=float((TOP & topP).sum() / k),
                    P3=float(P[G64].mean() - P[G1].mean()),
                    P4=float((S - P)[TOP].mean() - (S - P).mean()))
    st = stats(S, N, P, TOP)
    st_ord = stats(S, N, P, TOP, ord_rank)
    top40 = (S_all >= 40) & have
    st['P5'] = spearman(S_all[top40], P_all[top40]); st_ord['P5'] = spearman(S_all[top40], P_all[top40], ord_rank)
    # bootstrap over teams of C (sampling uncertainty of the observed statistics; not part of the decision rule)
    rng = np.random.default_rng(20261104); bs = {key: [] for key in ('P1', 'P2', 'P3', 'P4')}
    for _ in range(boot):
        ii = rng.integers(0, len(S), len(S)); Sb, Nb, Pb = S[ii], N[ii], P[ii]
        Tb = Sb >= np.quantile(Sb, 0.9)
        if (Nb >= 64).sum() == 0 or (Nb == 1).sum() == 0: continue
        for key, val in stats(Sb, Nb, Pb, Tb).items(): bs[key].append(val)
    print(f'C after matching: {len(S)} teams (n=1: {(N == 1).sum()}, n>=64: {(N >= 64).sum()}); public top-10% k = {k} '
          f'(cut {np.quantile(S, 0.9):.2f}); teams with public >= 40 matched: {top40.sum()}')
    print(f'public gap mean(n>=64) - mean(n=1) in C = {S[N >= 64].mean() - S[N == 1].mean():.2f}; '
          f'C mean public {S.mean():.2f}, private {P.mean():.2f}, SD public {S.std(ddof=1):.2f}, private {P.std(ddof=1):.2f}')
    crossed, inside = 0, 0
    print('pred  observed  (ordinal)  boot 90%        H value [range]       S      fails if     verdict')
    for key in ('P1', 'P2', 'P3', 'P4', 'P5'):
        hv, lo, hi, sv, ftxt, ff = PRED[key]; x = st[key]; f = ff(x); ins = lo <= x <= hi
        if key != 'P5': crossed += f; inside += ins
        b = bs.get(key); bt = f'[{np.quantile(b, 0.05):.2f}, {np.quantile(b, 0.95):.2f}]' if b else '            '
        print(f'{key}   {x:8.3f}  ({st_ord[key]:7.3f})  {bt:15s} {hv:5.2f} [{lo:.2f}, {hi:.2f}]   {sv if sv is not None else "-":>5}  {ftxt:11s}  '
              f'{"CROSSED" if f else ("inside H range" if ins else "outside H range, not crossed")}')
    verdict = 'H FAILS' if crossed >= 2 else ('H PASSES' if crossed == 0 and inside >= 3 else 'H neither passes nor fails (mixed)')
    print(f'decision rule on P1-P4: crossed {crossed}, inside H range {inside} -> {verdict}')
    if st['P1'] < 0.25 or st['P2'] < 0.17: print('note: P1 < 0.25 or P2 < 17%: more noise than measured (sealed note)')
    G1 = N == 1; D = float((P - S)[G1].mean()); print(f'D = mean(private - public) over C with n = 1: {D:.2f} ({G1.sum()} teams)')
    if OURS in NM:
        j = NM.index(OURS); print(f'our team: public {S[j]:.2f}, private {P[j]:.2f}; public pct in C {(S < S[j]).mean():.3f}, '
                                  f'P6c private pct in C {(P < P[j]).mean():.3f} (forecast 0.74, 0.15 to 0.99)')
    if subs_path:
        sub = {r['ref']: r for r in read(subs_path)}
        fam = [(ref, sub.get(ref, {}).get('publicScore'), sub.get(ref, {}).get('privateScore')) for ref in FAMILY]
        for ref, pu, pr in fam: print(f'  {ref}: public {pu}, private {pr}')
        pv = np.array([float(pr) for _, _, pr in fam if pr not in (None, '', 'None')])
        if len(pv) == len(FAMILY):
            sd = float(pv.std(ddof=1)); print(f'P6a SD of 8 family private scores {sd:.2f} (H 1.7 to 4.2; fails outside 1.2 to 5.5) -> '
                                              f'{"CROSSED" if not 1.2 <= sd <= 5.5 else ("inside" if 1.7 <= sd <= 4.2 else "outside range, not crossed")}; mean {pv.mean():.2f}')
            b = pv[0] - D; print(f'P6b 56832595 private - D = {b:.2f} (forecast 28, 22.5 to 34.5) -> {"inside" if 22.5 <= b <= 34.5 else "outside"}')
        else: print('P6a/P6b: private scores missing for some family runs')
    return st

if __name__ == '__main__':
    main(*sys.argv[1:4])
