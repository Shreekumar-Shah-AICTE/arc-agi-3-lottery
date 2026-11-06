"""Games that never start, with and without the V11 level-1 guard, at full token supply and at the V11 run's supply
(exploratory; written 5 Nov 2026 ~17:20 UTC, after the V11 readout). Uses sched_endgame_sim.py unchanged (the same
40 worlds x 10 runs, end-of-run rule on, 121 minutes per game); only R1 is scaled, by 0.86 (V11 generated 3.925M tokens,
0.86 of the FB25 control mean)."""
import numpy as np
import sched_endgame_sim as S

BASE = S.R1
print('Never-started and zero-level games per run, 40 worlds x 10 runs, 121 minutes, end-of-run rule on')
for scale in (1.0, 0.86):
    S.R1 = BASE * scale
    for lab, guard in (('no guard', False), ('guard', True)):
        never, zeros, tot, two = [], [], [], []
        for w, need in S.worlds(40, 10):
            t, lv, st, _ = S.run(need, 121, True, guard)
            n = sum(s0 is None for s0 in st); never.append(n); two.append(n >= 2)
            zeros.append(sum(1 for x, s0 in zip(lv, st) if s0 is not None and x == 0)); tot.append(t)
        print(f'  supply x{scale:.2f}, {lab:8s}: never started {np.mean(never):.2f} per run (2 or more in {100 * np.mean(two):.0f}% of runs); '
              f'zero-level games among started {np.mean(zeros):.2f}; levels {np.mean(tot):.1f}')
