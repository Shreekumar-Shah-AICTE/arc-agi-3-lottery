"""A/A calibration of the crack-cost test (6 Nov 2026, exploratory, CPU only; written after the outside review asked for it).

Question: with no change at all, how often does the crack-cost test of Section 4.5 flag a run? Each of the 6 valid
same-agent runs is treated in turn as a one-run "variant" and tested the way the variants were tested
(z = (b - mean of the control runs) / (SD * sqrt(1/k + 1/n_ctl)), k = 1), with the run itself left out of both the
control mean and the A/A SD, so that it cannot vouch for itself. Input: the per-run b table of
out/lever4b_two_numbers.out (lognormal AFT with level fixed effects; the variants' z values there use SD 0.051 from all 6).
"""
import math, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
txt = (HERE / 'out' / 'lever4b_two_numbers.out').read_text()
b = {m.group(1): float(m.group(2)) for m in re.finditer(r'^(b\w+)\s+\d+\s+[\d.]+\s+([+-][\d.]+)\s', txt, re.M)}
CTL = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']
SAME = CTL + ['b32_spec2_r2', 'b32_spec2_r3']          # spec2 is a no-op for cracking (D87 NOISE); spec2 A is invalid
def sd(v):
    m = sum(v) / len(v); return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
print('A/A calibration of the crack-cost test on the 6 valid same-agent runs (leave-one-out)')
zs = []
for r in SAME:
    ctl = [b[c] for c in CTL if c != r]
    s = sd([b[x] for x in SAME if x != r])
    z = (b[r] - sum(ctl) / len(ctl)) / (s * math.sqrt(1 + 1 / len(ctl)))
    zs.append(z)
    print(f'  {r:14s} b {b[r]:+.3f}  control mean {sum(ctl)/len(ctl):+.3f} (n {len(ctl)})  A/A SD without it {s:.3f}  z {z:+.2f}')
n2 = sum(abs(z) >= 2 for z in zs); nb = sum(abs(z) >= 2.64 for z in zs)
print(f'runs with |z| >= 2: {n2} of {len(zs)}; past the Bonferroni line |z| >= 2.64: {nb} of {len(zs)}; largest |z| {max(abs(z) for z in zs):.2f}')
p2 = math.erfc(2 / math.sqrt(2)); pb = math.erfc(2.64 / math.sqrt(2))
print(f'normal-theory rate per run: |z| >= 2 {100*p2:.1f}%, |z| >= 2.64 {100*pb:.1f}%  (expected of 6: {6*p2:.2f} and {6*pb:.2f})')
# What the variant tests of Section 4.5 would say at the five-run SD (b32_spec2_r3 left out), same z formula:
s5 = sd([b[x] for x in SAME if x != 'b32_spec2_r3'])
var = re.findall(r'^\s+(\S+) \((\w+)\)\s+runs (\d+)\s+mean b ([+-][\d.]+)\s+z ([+-][\d.]+)', txt, re.M)
cross = 0
print(f'variant tests at the five-run A/A SD {s5:.3f} (control mean 0 by construction, n_ctl 4):')
for name, kind, k, mb, z6 in var:
    z5 = float(mb) / (s5 * math.sqrt(1 / int(k) + 1 / 4)); cross += abs(z5) >= 2.64
    print(f'  {name:12s} runs {k}  mean b {mb}  z at SD 0.051 {z6}  z at SD {s5:.3f} {z5:+.1f}')
within = [abs(z) for r, z in zip(SAME, zs) if r != 'b32_spec2_r3']
print(f'five runs other than b32_spec2_r3: largest |z| {max(within):.2f}; variants past |z| 2.64 at the five-run SD: {cross} of {len(var)}')
