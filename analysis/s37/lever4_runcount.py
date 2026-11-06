"""Lever 4 (4 Nov 2026): how many runs does an ARC-AGI-3 comparison need?
Inputs are our measured run-to-run spreads for byte-identical code:
  leaderboard: 5 identical B12 submissions, SD 2.87 points (score 0-100, 55 hidden games per half);
  dev: 6 same-agent FB25 runs (25 public games x 121 min), SD 5.3 levels (mean 102.5) - lever3_noise.py.
Two-arm comparison, n runs per arm, two-sided alpha 0.05, power 0.8: n = 2 (z_.975 + z_.8)^2 sigma^2 / delta^2.
Best-of-n: the expected maximum of n independent draws is mu + sigma * e_n (e_n = expected max of n standard normals,
computed exactly by numerical integration),
which is what a team that resubmits identical code and keeps its best public score should expect to see."""
import math
import numpy as np
Z = 1.959964 + 0.841621
def runs_per_arm(sigma, delta): return math.ceil(2 * Z * Z * sigma * sigma / (delta * delta))
def e_max(n):   # exact: E[max of n standard normals] = integral of x n phi(x) Phi(x)^(n-1) dx (was Monte Carlo before 4 Nov ~18:05 UTC)
    x = np.linspace(-12, 12, 240001); phi = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    return float(np.sum(x * n * phi * Phi ** (n - 1)) * (x[1] - x[0]))
SIG_LB, SIG_DEV, DEV_MEAN = 2.87, 5.3, 102.5
print('Runs per arm to detect a true difference (power 0.8, two-sided 0.05)')
print('  leaderboard (SD %.2f points per submission):' % SIG_LB)
for d in (1, 2, 3, 5, 8):
    print(f'    +{d} points: {runs_per_arm(SIG_LB, d)} submissions per arm')
print('  dev, 25 public games x 121 min (SD %.1f levels per run, mean %.1f):' % (SIG_DEV, DEV_MEAN))
for pct in (2, 5, 10, 20):
    d = DEV_MEAN * pct / 100
    print(f'    +{pct}% levels (+{d:.1f}): {runs_per_arm(SIG_DEV, d)} runs per arm')
print('\nOne run per arm: the smallest difference detectable at power 0.8 = %.1f points (LB), %.1f levels = %.0f%% (dev)' % (
    Z * SIG_LB * math.sqrt(2), Z * SIG_DEV * math.sqrt(2), 100 * Z * SIG_DEV * math.sqrt(2) / DEV_MEAN))
print('\nBest of n identical submissions (expected public best above the mean, sigma %.2f):' % SIG_LB)
for n in (1, 2, 5, 10, 20, 50, 100, 200):
    e = e_max(n); print(f'    n {n:3d}: +{e * SIG_LB:.2f} points (e_n {e:.2f})')

# Added 4 Nov ~18:10 UTC. (1) The counts above use the normal approximation; with the SD estimated from the same runs
# the test is a t-test: simulated power and the smallest n reaching 0.8 (seed 0). (2) The SDs come from 5 and 6 runs;
# 90% chi-square interval for each SD (normal runs assumed) and the runs per arm at its ends.
def t_power(sigma, delta, n, rng, R=40000):
    def ts(d):
        a, b = rng.normal(0, sigma, (R, n)), rng.normal(d, sigma, (R, n))
        return (b.mean(1) - a.mean(1)) / np.sqrt((a.var(1, ddof=1) + b.var(1, ddof=1)) / n)
    crit = np.quantile(np.abs(ts(0.0)), 0.95)
    return float((np.abs(ts(delta)) > crit).mean())
rng = np.random.default_rng(0)
print('\nt-test check (simulated, 40000 pairs per n, seed 0): formula n -> smallest n with power >= 0.8')
for label, sigma, delta in (('LB +3 points', SIG_LB, 3), ('LB +5 points', SIG_LB, 5),
                            ('dev +10% levels', SIG_DEV, 0.10 * DEV_MEAN), ('dev +20% levels', SIG_DEV, 0.20 * DEV_MEAN)):
    n0 = max(2, runs_per_arm(sigma, delta)); n = n0
    while t_power(sigma, delta, n, rng) < 0.8: n += 1
    print(f'  {label}: formula {n0}, t-test {n}')
def p_lower(a, x):   # regularized lower incomplete gamma, series
    s = term = 1.0 / a; k = 0
    while abs(term) > 1e-15 * abs(s): k += 1; term *= x / (a + k); s += term
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
def chi2_q(p, df):
    lo, hi = 1e-9, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if p_lower(df / 2, mid / 2) < p else (lo, mid)
    return (lo + hi) / 2
print('\nSD uncertainty (90% chi-square interval, normal runs assumed)')
for label, s, k, d, unit in (('LB', SIG_LB, 5, 3.0, 'points'), ('dev', SIG_DEV, 6, 0.10 * DEV_MEAN, 'levels')):
    lo, hi = s * math.sqrt((k - 1) / chi2_q(0.95, k - 1)), s * math.sqrt((k - 1) / chi2_q(0.05, k - 1))
    print(f'  {label} SD {s} from {k} runs: {lo:.2f} to {hi:.2f} {unit}; runs per arm for +{d:.1f}: '
          f'{runs_per_arm(lo, d)} to {runs_per_arm(hi, d)} (point estimate {runs_per_arm(s, d)})')
