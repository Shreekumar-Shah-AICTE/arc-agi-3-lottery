"""Scheduler simulation with and without the harness's end-of-run rule (exploratory; written 5 Nov 2026 ~16:20 UTC after
the operator's OK at 16:04 UTC, before any V11 output was read).
Question: how much does the end-of-run rule (phi: the levels-left bonus B of the slot rule D' fades linearly to 0 over the
last 40% of each game's budget; Appendix C) change the level total at 121 and 242 minutes per game?
Model: crack-model H2 worlds (Section 3.4: level log-medians MU + 0.80 N(0,1), run luck s 0.49, walls fixed per world with
q 0.06), the 25 public games' level counts, 10 slots, each held slot generating R1 tokens per minute (an FB25 run's 4.45M
tokens over 121 minutes and 10 slots), actions = tokens / 1,500 on the current level, and the slot rule D' (d_parts, as in
walls.py and the B12 notebook). A game hands its slot back at every drain (every D generated tokens; base 58k, the size of
one drain) and when it finishes; the free slot goes to the highest priority among the waiting games and the game that
handed it back. Games that have not started wait in the same queue. Every game's clock starts at minute 0.
Variants share their random numbers (the same worlds, level waits and walls), so differences are paired.
Optional: the V11 level-1 guard (a game on level 1 with 0 < tokens < 150k on it gets priority 5e8), for comparison only;
it is not part of the sealed V11 test."""
import math, statistics, sys
import numpy as np
from v5_budget_prediction import FB, load

NL = [g['number_of_levels'] for g in load(FB[0])]
GID = [g['game_id'][:4] for g in load(FB[0])]
MU, SD_LEVEL, S_RUN, Q = math.log(22.1e3), 0.80, 0.49, 0.06
R1 = 4.45e6 / (121 * 10)          # tokens per minute per held slot
TPA, SLOTS, DT = 1500.0, 10, 0.25  # tokens per action, slots, minutes per step

def d_parts(level, a, t, pace, n):   # copied from walls.py (B12 notebook, ours_form_priority.py)
    nc = min(10, max(6, n)); norm = 55 / (nc * (nc + 1) / 2)
    A = (1.0 + 0.5 * (level - 1)) * norm * (300.0 / (300.0 + a)) ** 2.5
    if pace > 0: A *= min(4.0, max(0.25, (30000.0 / max(1.0, pace)) ** 0.4))
    T = 225000.0 * min(2.0, max(0.5, (pace if pace > 0 else 30000.0) / 30000.0)) ** 0.5
    C = 0.1 * max(0.1, 1 - a / 115.0) + 0.9 * max(0.1, 1 - t / T)
    B = (0.0, 10.0, 14.0, 16.0)[min(max(0, n - level), 3)]
    return A * C, B, norm

def run(need, budget, endgame=True, guard=False, D=58e3, switch=None, mark=121.0, info=None):
    """need[i][k]: tokens to crack level k of game i (inf = wall). Returns (total, levels per game, start minute per game,
    total at minute `mark`). switch: before this minute the end-of-run rule runs on a `switch`-minute budget, after it on
    the full budget (a 121-minute run continued as if it had been a 242-minute run, the way the sealed V5 calculation
    continued the control runs' end states)."""
    G = len(NL); lvl = [0] * G; t = [0.0] * G; ctx = [0.0] * G; spent = [0.0] * G
    fin = [False] * G; start = [None] * G; hold = set(); queue = set(range(G))
    def phi(m):
        bud = switch if (switch is not None and m < switch) else budget
        if not endgame or m < 0.6 * bud: return 1.0
        return max(0.0, (bud - m) / (0.4 * bud))
    def prio(i, m):
        if guard and lvl[i] == 0 and 0 < t[i] < 150e3: return 5e8
        pace = spent[i] / lvl[i] if lvl[i] else 0.0
        u, b, _ = d_parts(lvl[i] + 1, t[i] / TPA, t[i], pace, NL[i])
        return u + b * phi(m)
    def fill(m, extra=None):
        cand = set(queue) | ({extra} if extra is not None else set())
        while len(hold) < SLOTS and cand:
            j = max(cand, key=lambda i: (prio(i, m), -i)); cand.discard(j); queue.discard(j); hold.add(j)
            if start[j] is None: start[j] = m
        queue.update(cand)
    fill(0.0)
    steps = int(budget / DT); at_mark = None
    for s in range(steps):
        m = s * DT
        if at_mark is None and m >= mark:
            at_mark = sum(lvl)
            if info is not None: info['mark'] = (lvl[:], t[:], spent[:], fin[:])
        for i in sorted(hold):
            if i not in hold: continue
            dtok = R1 * DT; t[i] += dtok; ctx[i] += dtok
            if t[i] >= need[i][lvl[i]]:
                spent[i] += need[i][lvl[i]]; lvl[i] += 1; t[i] = 0.0
                if lvl[i] >= NL[i]:
                    fin[i] = True; hold.discard(i); fill(m); continue
            if ctx[i] >= D:
                ctx[i] -= D; hold.discard(i); fill(m, extra=i)
    if info is not None: info['end'] = (lvl[:], t[:], spent[:], fin[:])   # optional state for sched_sealedcalc_check.py
    return sum(lvl), lvl, start, (at_mark if at_mark is not None else sum(lvl))

def worlds(nw, nr, seed=11):
    rng = np.random.default_rng(seed)
    for w in range(nw):
        med = [[MU + SD_LEVEL * rng.standard_normal() for _ in range(n)] for n in NL]
        wall = [[rng.random() < Q for _ in range(n)] for n in NL]
        for r in range(nr):
            yield w, [[math.inf if wall[i][k] else math.exp(med[i][k] + S_RUN * rng.standard_normal()) for k in range(n)]
                      for i, n in enumerate(NL)]

if __name__ == '__main__':
    NW, NR = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (20, 10)
    V = {'121 on': (121, True, False), '121 off': (121, False, False), '242 on': (242, True, False),
         '242 off': (242, False, False), '121 on, continued to 242': (242, True, False, 121),
         '121 on + L1 guard': (121, True, True)}
    res = {k: [] for k in V}; zeros = {k: [] for k in V}; last = {k: [] for k in V}; never = {k: [] for k in V}
    mk = {k: [] for k in V}
    for w, need in worlds(NW, NR):
        for k, (b, eg, gd, *sw) in V.items():
            tot, lv, st, m121 = run(need, b, eg, gd, switch=(sw[0] if sw else None)); mk[k].append(m121)
            res[k].append(tot); zeros[k].append(sum(1 for x, s0 in zip(lv, st) if s0 is not None and x == 0))
            last[k].append(max(s0 for s0 in st if s0 is not None)); never[k].append(sum(s0 is None for s0 in st))
    n = len(res['121 on'])
    print(f'Scheduler simulation: {NW} H2 worlds x {NR} runs = {n} paired runs per variant; R1 {R1:.0f} tokens/min per slot, '
          f'drain every 58k tokens, 1,500 tokens per action')
    print('observed: controls at 121 min 102.75 levels (95-112 over 7 runs, 2.43 zero-level games per run, last start minute '
          '69-108, none unstarted); V5 at 242 min 141 (105 by minute 121)')
    for k in V:
        a = np.array(res[k])
        print(f'  {k:18s} levels mean {a.mean():6.1f} (run SD {a.std(ddof=1):4.1f}); zero-level games {np.mean(zeros[k]):4.2f}; '
              f'last start minute {np.mean(last[k]):5.1f} (10-90% {np.quantile(last[k], .1):5.1f}-{np.quantile(last[k], .9):5.1f}); '
              f'never started {np.mean(never[k]):4.2f}; levels at minute 121 {np.mean(mk[k]):6.1f}')
    def paired(a, b):
        d = np.array(res[a]) - np.array(res[b]); se = d.std(ddof=1) / math.sqrt(len(d))
        return d.mean(), se
    for a, b in (('121 on', '121 off'), ('242 on', '242 off'), ('242 on', '121 on'), ('242 off', '121 off'),
                 ('242 on', '121 on, continued to 242'), ('121 on + L1 guard', '121 on')):
        m_, se = paired(a, b)
        print(f'  {a} minus {b}: {m_:+.2f} levels (SE {se:.2f})')
    g_on = np.array(res['242 on']) - np.array(res['121 on']); g_off = np.array(res['242 off']) - np.array(res['121 off'])
    print(f'  gain from doubling the budget: with the end-of-run rule {g_on.mean():.1f}, without {g_off.mean():.1f} '
          f'(difference {np.mean(g_on - g_off):+.2f}, SE {np.std(g_on - g_off, ddof=1) / math.sqrt(n):.2f})')
