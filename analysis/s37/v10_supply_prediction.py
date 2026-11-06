"""V10 (lever 5 plan T1, S37 4 Nov 2026): prediction for the FB25 control with speculative decoding off.
Sealed 4 Nov 2026 ~16:45 UTC as sealed/v10_supply_prediction.txt (its table is this script's output), before T1 and
T1c were launched. Endgame assumption (stated in the seal): the harness's D' endgame fades the levels-left bonus over the
last 40% of each game's budget, from minute 72.6 in T1 and in the controls alike, so in token terms T1 enters it earlier;
we assume it reshuffles levels between games without changing the total at a given token count much.
Model (token-time equivalence, as in lever4b): the same agent with token supply ratio r = S/S_ctl behaves like the
control run stopped at r x 121 min, so total levels = L(r), the control clock curve. Band: run-to-run SD at m = r (the
four control runs, floored by the six-run SD 5.3 scaled as sqrt(levels)), plus the curve's own SE (SD / sqrt 4);
90% = +-1.645 x combined SD."""
import gzip, json, math, sys
import numpy as np
from common import HERE, CTL
T_END = 121 * 60
per_run = {}
for run in CTL:
    d = json.load(gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt')); ts = []
    for g in d['game_runs']:
        h = g.get('history') or []; apl = g.get('actions_per_level') or []; lc = g.get('levels_completed') or 0
        cum = np.cumsum([0] + list(apl)); wc = [x.get('wallclock_seconds') or 0 for x in h]
        ts += [wc[cum[i + 1] - 1] for i in range(lc) if 0 <= cum[i + 1] - 1 < len(wc)]
    per_run[run] = np.array(ts)
ms = np.round(np.arange(0.40, 1.0001, 0.05), 2)
M = np.array([[(per_run[r] <= m * T_END).sum() for m in ms] for r in CTL], float)   # runs x m
mean, sd_obs = M.mean(0), M.std(0, ddof=1)
# Four runs give a noisy SD per m; floor it with the six-run end-of-run SD (5.3 levels, lever3_noise.out) scaled as
# sqrt(levels), i.e. SD(m) = max(observed, 5.3 x sqrt(L(m) / L(1))).
sd = np.maximum(sd_obs, 5.3 * np.sqrt(mean / mean[-1]))
band = 1.645 * np.sqrt(sd ** 2 + (sd / math.sqrt(len(CTL))) ** 2)
out = ['V10: FB25 control with speculative decoding off; 25 public games x 121 min, 10 streams.',
       'Conditional prediction, given the measured supply ratio r = total generated tokens / control mean:',
       '  r     levels  90% band   (control runs at m = r: mean, observed SD)']
for m, a, s_, b in zip(ms, mean, sd_obs, band):
    out.append(f'  {m:.2f}  {a:6.1f}   {a - b:5.1f}-{a + b:5.1f}   ({a:.1f}, {s_:.1f})')
lo, hi = 0.55, 0.90   # prior range for r with spec off (INFERENCE: no spec-off run on this stack yet)
i_lo, i_hi = np.searchsorted(ms, lo), np.searchsorted(ms, hi)
out.append(f'Unconditional (prior r {lo}-{hi}): levels {mean[i_lo] - band[i_lo]:.0f} to {mean[i_hi] + band[i_hi]:.0f}; '
           f'control (r = 1) {mean[-1]:.1f}.')
out.append('Rule: token-time equivalence SUPPORTED if observed levels fall inside the band at the measured r; levels above'
           ' the band = the agent is not token-limited at this supply; below = slower serving costs more than its tokens.')
txt = '\n'.join(out) + '\n'
dest = sys.argv[1] if len(sys.argv) > 1 else str(HERE / 'out' / 'v10_supply_prediction.out')
open(dest, 'w').write(txt); print(txt, end=''); print('written to', dest)
