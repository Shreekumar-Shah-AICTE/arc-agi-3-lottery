"""Did our changes beat the base notebook on the leaderboard? (exploratory, 5 Nov 2026)
B1 = fork of dfranzen's Milestone 2 notebook, submitted twice (submissions 2 and 3 of Table 1); X6 = B1 + solved-level
ledger, once; B12 = X6 ledger v2 + D' slot priority, five byte-identical draws. Scores are the final public ARC-AGI-3
leaderboard scores (paper/tables/t1_arc3_submissions.md). Per-submission SD: B12's own (2.87, Section 4.6) and pooled.
Run: python3 analysis/s37/b1_vs_b12.py > analysis/s37/out/b1_vs_b12.out"""
import math, statistics
from common import B12_LB
B1 = [28.30, 26.72]; X6 = [29.29]
m1, m12 = statistics.mean(B1), statistics.mean(B12_LB); d = m12 - m1
sd12 = statistics.stdev(B12_LB)
sdp = math.sqrt((sum((x - m1) ** 2 for x in B1) + sum((x - m12) ** 2 for x in B12_LB)) / (len(B1) + len(B12_LB) - 2))
print(f'B1 mean {m1:.2f} (n={len(B1)}), B12 mean {m12:.2f} (n={len(B12_LB)}), X6 {X6[0]:.2f} (n=1); B12 - B1 = {d:.2f} points')
for lab, sd in (('B12 SD', sd12), ('pooled SD', sdp)):
    se = sd * math.sqrt(1 / len(B1) + 1 / len(B12_LB))
    p = math.erfc(abs(d / se) / math.sqrt(2))
    n = 2 * (1.959964 + 0.841621) ** 2 * sd ** 2 / d ** 2
    print(f'  {lab} {sd:.2f}: SE {se:.2f}, z {d / se:.2f}, two-sided p {p:.2f}; submissions per arm to detect a true {d:.2f} '
          f'with power 0.8: {math.ceil(n)}')
se6 = sd12 * math.sqrt(1 + 1 / len(B1))
print(f'X6 - B1 = {X6[0] - m1:.2f} (SE {se6:.2f} at the B12 SD); B12 - X6 = {m12 - X6[0]:.2f}')
