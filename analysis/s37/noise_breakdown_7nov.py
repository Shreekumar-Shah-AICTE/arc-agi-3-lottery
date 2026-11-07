"""Noise breakdown with the harness simulator (item o of the 7 Nov reviews; exploratory; written 7 Nov 2026 ~13:05 UTC).
Question: how much of the run-to-run noise in a run's level total comes from the crack waits themselves, and how much from
the harness's slot rule (shelving games whose current level runs long, and starting some games late)?
Model and harness as in sched_endgame_sim.py: crack-model H2 worlds (level log-medians MU + 0.80 N(0,1) and walls (q 0.06)
fixed per world, run luck s 0.49 redrawn every run), the 25 public games, 10 slots, R1 tokens per minute per held slot, a
drain every 58k tokens, 121 minutes, the end-of-run rule on. Four schedulers share the random numbers (paired):
  PS      processor sharing: every unfinished game gets an equal share of the 10 slots' tokens from minute 0 (no queue,
          no slot rule), so only the crack waits vary;
  FIFO    10 slots taken in turn: a game hands its slot back at every drain and joins the back of the queue;
  D'+st   the slot rule D', except that games not yet started go first (so every game starts within about 32 minutes);
  D'      the slot rule as in the B12 harness;
  D'+g    D' with the V11 level-1 guard (a game on level 1 with 0 < tokens < 150k on it gets top priority), for comparison.
Run-to-run variance of the level total within a world (NR runs per world), averaged over worlds. Sequential shares:
crack waits = PS, turn-taking = FIFO - PS, shelving = D'+st - FIFO, late starts = D' - D'+st (order-dependent along this
chain), and paired differences of the mean level total (SE over worlds). Per game: zero-level games per run, and the share of the summed per-game variance carried
by the 0 / at-least-1 split (walls.py section 7; observed 49%, bootstrap 90% interval 39% to 57%, over seven runs).
Usage: python3 noise_breakdown_7nov.py [worlds] [runs per world]"""
import math, sys, time
from collections import deque
import numpy as np
from sched_endgame_sim import NL, d_parts, R1, TPA, SLOTS, DT, worlds

def run(need, mode, budget=121.0, D=58e3, endgame=True):
    G = len(NL); lvl = [0] * G; t = [0.0] * G; ctx = [0.0] * G; spent = [0.0] * G
    fin = [False] * G; start = [None] * G
    steps = int(budget / DT)
    def crack(i):
        if t[i] >= need[i][lvl[i]]:
            spent[i] += need[i][lvl[i]]; lvl[i] += 1; t[i] = 0.0
            if lvl[i] >= NL[i]: fin[i] = True
            return True
        return False
    if mode == 'PS':
        for s in range(steps):
            act = [i for i in range(G) if not fin[i]]
            if not act: break
            dtok = R1 * DT * min(SLOTS, len(act)) / len(act)
            for i in act:
                if start[i] is None: start[i] = s * DT
                t[i] += dtok; crack(i)
        return sum(lvl), lvl, start
    hold = set()
    if mode == 'FIFO':
        queue = deque(range(G))
        def fill(m, back=None):
            if back is not None: queue.append(back)
            while len(hold) < SLOTS and queue:
                j = queue.popleft(); hold.add(j)
                if start[j] is None: start[j] = m
    else:
        queue = set(range(G))
        def phi(m):
            if not endgame or m < 0.6 * budget: return 1.0
            return max(0.0, (budget - m) / (0.4 * budget))
        def prio(i, m):
            if mode == "D'+st" and start[i] is None: return 1e9
            if mode == "D'+g" and lvl[i] == 0 and 0 < t[i] < 150e3: return 5e8
            pace = spent[i] / lvl[i] if lvl[i] else 0.0
            u, b, _ = d_parts(lvl[i] + 1, t[i] / TPA, t[i], pace, NL[i])
            return u + b * phi(m)
        def fill(m, back=None):
            cand = set(queue) | ({back} if back is not None else set())
            while len(hold) < SLOTS and cand:
                j = max(cand, key=lambda i: (prio(i, m), -i)); cand.discard(j); queue.discard(j); hold.add(j)
                if start[j] is None: start[j] = m
            queue.update(cand)
    fill(0.0)
    for s in range(steps):
        m = s * DT
        for i in sorted(hold):
            if i not in hold: continue
            dtok = R1 * DT; t[i] += dtok; ctx[i] += dtok
            crack(i)
            if fin[i]:
                hold.discard(i); fill(m); continue
            if ctx[i] >= D:
                ctx[i] -= D; hold.discard(i); fill(m, back=i)
    return sum(lvl), lvl, start

def zshare(M):          # walls.py section 7: share of summed per-game variance carried by the 0 / at-least-1 split
    t = b = 0.0
    for j in range(M.shape[1]):
        lv = M[:, j]; p0 = (lv == 0).mean(); t += lv.var()
        if 0 < p0 < 1: b += p0 * (1 - p0) * lv[lv > 0].mean() ** 2
    return b / t if t else float('nan')

if __name__ == '__main__':
    NW, NR = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (60, 7)
    MODES = ('PS', 'FIFO', "D'+st", "D'", "D'+g")
    t0 = time.time()
    TOT = {k: np.zeros((NW, NR)) for k in MODES}; LV = {k: np.zeros((NW, NR, len(NL))) for k in MODES}
    Z = {k: np.zeros((NW, NR)) for k in MODES}; LAST = {k: np.zeros((NW, NR)) for k in MODES}
    seen = {}
    for w, need in worlds(NW, NR, seed=11):
        r = seen.get(w, 0); seen[w] = r + 1
        for k in MODES:
            tot, lv, st = run(need, k)
            TOT[k][w, r] = tot; LV[k][w, r] = lv
            Z[k][w, r] = sum(1 for x, s0 in zip(lv, st) if s0 is not None and x == 0)
            LAST[k][w, r] = max(s0 for s0 in st if s0 is not None)
    print(f"Noise breakdown: {NW} H2 worlds x {NR} runs, paired across 5 schedulers; 121 minutes; {time.time() - t0:.0f} s")
    print('observed (seven 121-minute runs): run SD of levels 5.3 on six runs, 6.0 with the seventh; 2.43 zero-level games '
          'per run; last start minute 69 to 108; 0 / at-least-1 split 49% of the summed per-game variance (39% to 57%)')
    V = {k: TOT[k].var(axis=1, ddof=1) for k in MODES}          # within-world run variance per world
    rng = np.random.default_rng(5); B = 2000; idx = rng.integers(0, NW, (B, NW))
    def ci(x):
        bs = x[idx].mean(axis=1); return np.percentile(bs, 5), np.percentile(bs, 95)
    for k in MODES:
        zs = np.array([zshare(LV[k][w]) for w in range(NW)])
        lo, hi = ci(V[k])
        print(f"  {k:6s} levels {TOT[k].mean():6.1f}; run SD {math.sqrt(V[k].mean()):4.1f} (variance {V[k].mean():5.1f}, "
              f"90% {lo:5.1f} to {hi:5.1f}); zero-level games {Z[k].mean():4.2f}; last start {LAST[k].mean():5.1f} min; "
              f"0/1 split share {100 * np.nanmean(zs):3.0f}% (worlds 10-90% {100 * np.nanpercentile(zs, 10):.0f}-{100 * np.nanpercentile(zs, 90):.0f}%)")
    full = V["D'"]
    def share(a, b=None):
        x = V[a] - (V[b] if b else 0.0)
        bs = x[idx].mean(axis=1) / full[idx].mean(axis=1)
        return 100 * x.mean() / full.mean(), 100 * np.percentile(bs, 5), 100 * np.percentile(bs, 95)
    print("  shares of the D' run variance (bootstrap over worlds, 90%):")
    for lab, a, b in (('crack waits (PS)', 'PS', None), ('turn-taking (FIFO - PS)', 'FIFO', 'PS'),
                      ("shelving (D'+st - FIFO)", "D'+st", 'FIFO'), ("late starts (D' - D'+st)", "D'", "D'+st")):
        m_, lo, hi = share(a, b); print(f'    {lab:26s} {m_:5.0f}% ({lo:.0f}% to {hi:.0f}%)')
    m_, lo, hi = share("D'", 'PS'); print(f"    slot rule and queue together (D' - PS) {m_:.0f}% ({lo:.0f}% to {hi:.0f}%)")
    m_, lo, hi = share("D'", 'FIFO'); print(f"    priorities alone (D' - FIFO) {m_:.0f}% ({lo:.0f}% to {hi:.0f}%)")
    print('  paired differences of the mean level total (SE over worlds):')
    for a, b in (("D'", 'PS'), ("D'", 'FIFO'), ("D'+st", "D'"), ("D'+g", "D'")):
        d = (TOT[a] - TOT[b]).mean(axis=1); print(f'    {a} minus {b}: {d.mean():+.2f} levels (SE {d.std(ddof=1) / math.sqrt(NW):.2f})')
