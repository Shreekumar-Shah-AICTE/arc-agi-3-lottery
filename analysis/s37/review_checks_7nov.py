"""Checks asked for by the 7 Nov 2026 outside reviews (exploratory; written after every outcome used here was known).
(1) Drift: do our ARC-AGI-3 leaderboard scores fall with the submission date? Least squares on Table 1.
(2) V5 against its sealed log-linear alternative: likelihood ratio at the observed total, normal errors with the SD
    implied by the sealed 90% interval for both predictions (the sealed file gives no interval for the alternative).
(3) Kinds of the 32 negative ledger rows (negatives_kind.py).
Writes out/review_checks_7nov.out."""
import math, re
from datetime import datetime
from common import HERE, ROOT
from negatives_kind import kind
import csv

def ols(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x); b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    res = [c - my - b * (a - mx) for a, c in zip(x, y)]; s2 = sum(r * r for r in res) / (n - 2)
    return b, math.sqrt(s2 / sxx), n - 2

def t_p(t, df, steps=200000):   # two-sided p of Student t by Simpson integration of the density from |t| to 60
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)); f = lambda u: c * (1 + u * u / df) ** (-(df + 1) / 2)
    a, b = abs(t), 60.0; h = (b - a) / steps
    s = f(a) + f(b) + sum((4 if i % 2 else 2) * f(a + i * h) for i in range(1, steps))
    return 2 * s * h / 3

rows = [l.split('|') for l in open(ROOT / 'paper/tables/t1_arc3_submissions.md', encoding='utf-8') if re.match(r'\| \d+ \|', l)]
subs = [(datetime.strptime(r[2].strip(), '%Y-%m-%d %H:%M'), r[4].strip(), float(r[5])) for r in rows]
t0 = subs[0][0]; day = lambda d: (d - t0).total_seconds() / 86400
b12 = [(day(d), s) for d, w, s in subs if w.startswith('B12')]
b, se, df = ols([a for a, _ in b12], [s for _, s in b12]); p = t_p(b / se, df)
b_wo, _, _ = ols([a for a, _ in b12[1:]], [s for _, s in b12[1:]])
first_b12 = min(d for d, w, s in subs if w.startswith('B12'))
var = [s for d, w, s in subs if d > first_b12 and not w.startswith(('B12', 'A1'))]
# B12 and its variants (all entries from the first B12 on, A1 excluded): score = a + slope * day + effect * is_variant
grp = [(day(d), 0.0 if w.startswith('B12') else 1.0, s) for d, w, s in subs if d >= first_b12 and not w.startswith('A1')]
import numpy as np
X = np.array([[1.0, a, v] for a, v, _ in grp]); Y = np.array([s for *_, s in grp])
beta, *_ = np.linalg.lstsq(X, Y, rcond=None); r = Y - X @ beta; dfg = len(Y) - 3
cov = (r @ r) / dfg * np.linalg.inv(X.T @ X); seg = float(np.sqrt(cov[1, 1])); sev = float(np.sqrt(cov[2, 2]))
pg = t_p(beta[1] / seg, dfg); pv = t_p(beta[2] / sev, dfg)
out = []
out.append(f"drift B12 draws n {len(b12)} slope {b:.2f} points per day se {se:.2f} p {p:.2f}; without the first draw slope {b_wo:.2f}")
out.append(f"drift B12 and variants n {len(grp)} slope {beta[1]:.2f} points per day se {seg:.2f} p {pg:.2f}; variant effect {beta[2]:.2f} se {sev:.2f} p {pv:.2f}")
out.append(f"post-B12 variants n {len(var)} mean {sum(var) / len(var):.2f} vs B12 mean {sum(s for _, s in b12) / len(b12):.2f}")
# (2) V5 likelihood ratio (sealed numbers from sealed/v5_budget_prediction.txt and the readout)
pred, lo, hi, alt, obs = 127.0, 110.0, 145.0, 140.0, 141.0
sd = (hi - lo) / (2 * 1.6449)
lr = math.exp(((obs - pred) ** 2 - (obs - alt) ** 2) / (2 * sd * sd))
out.append(f"v5 likelihood ratio log-linear over crack model {lr:.1f} (sd {sd:.1f} from the sealed 90% interval)")
# (3) kinds of the negative rows
neg = list(csv.DictReader(open(HERE / 'out' / 'ledger_negatives.csv', encoding='utf-8')))
from collections import Counter
C = Counter(kind(r['id']) for r in neg)
out.append('negative rows ' + ' '.join(f"{k.replace(' ', '_')} {n}" for k, n in sorted(C.items())) + f" total {len(neg)}")
(HERE / 'out' / 'review_checks_7nov.out').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
