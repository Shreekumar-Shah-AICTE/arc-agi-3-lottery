#!/usr/bin/env python3
"""A first look inside the walls from the action logs (6-7 Nov 2026, exploratory, CPU; D103). Written after outside
review round 2 asked what the walls are. Reads only the benchmark files: per level, the human baseline action count h
that the scorer uses (base_actions_per_level) and the agent's actions on the level (actions_per_level; for the open level
of a game, the actions it spent before the run ended). The 9 likely walls are the levels that walls.py calls walls with
probability 0.5 or more (reached but never cleared in 8 runs). No reasoning text is read: the agent's own notes are not in
the repo. Question: are the walls the levels that need the most actions from people?"""
import gzip, json, statistics as st
import numpy as np
from common import HERE, CTL, SPEC2_VALID
ONEX = CTL + SPEC2_VALID + ['t1c_fb25_ctl']            # seven valid same-agent runs at 121 min (as walls.py)
RUNS = ONEX + ['v5_fb50']
WALLS = [('dc22', 5), ('ka59', 6), ('sp80', 5), ('s5i5', 4), ('g50t', 5), ('bp35', 4), ('cn04', 6), ('ls20', 6), ('lf52', 5)]
DEV = {'ar25', 'ft09', 'lp85', 'r11l', 're86', 'sb26', 'sc25', 'tr87', 'tu93', 'vc33'}
H, cleared, spent, tok, clr = {}, set(), {}, {}, []
for r in RUNS:
    for g in json.load(gzip.open(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz', 'rt'))['game_runs']:
        gm = g['game_id'][:4]; H[gm] = g['base_actions_per_level']; c = g['levels_completed'] or 0
        for l in range(1, c + 1): cleared.add((gm, l))
        apl = g.get('actions_per_level') or []; hist = g.get('history') or []
        cum = np.cumsum([0] + list(apl))
        for l in range(1, c + 1):          # cleared levels: actions a, tokens, a/h
            seg = hist[cum[l - 1]:cum[l]]
            if seg and H[gm][l - 1]: clr.append((gm, l, apl[l - 1], sum(x.get('generated_tokens') or 0 for x in seg), apl[l - 1] / H[gm][l - 1]))
        if c < g['number_of_levels'] and len(apl) > c and hist:
            spent.setdefault((gm, c + 1), []).append(apl[c])
            seg = hist[cum[c]:]
            tok.setdefault((gm, c + 1), []).append((len(seg), sum(x.get('generated_tokens') or 0 for x in seg)))
def rel(gm, l):                      # h of this level over the median h of the game's earlier levels
    prev = H[gm][:l - 1]; return H[gm][l - 1] / st.median(prev) if prev else float('nan')
print(f'{len(RUNS)} runs; levels cleared at least once: {len(cleared)}')
print('  game level  human-h  h/median(earlier h)  rank of h in game  agent actions on the level (runs reaching)')
ratios_w = []
for gm, l in WALLS:
    h = H[gm][l - 1]; rk = sorted(H[gm], reverse=True).index(h) + 1; ratios_w.append(rel(gm, l))
    print(f'  {gm} L{l}   {h:5d}   {rel(gm, l):5.2f}   {rk} of {len(H[gm])}   {sorted(spent.get((gm, l), []))}')
ho_cleared = [(gm, l) for gm, l in cleared if gm not in DEV and l > 1]
ratios_c = [rel(gm, l) for gm, l in ho_cleared]
print(f'h relative to the earlier levels of the game: walls median {st.median(ratios_w):.2f} (range {min(ratios_w):.2f} to {max(ratios_w):.2f}); '
      f'held-out levels cleared at least once (level 2 or later, n={len(ratios_c)}): median {st.median(ratios_c):.2f}')
above = sum(r > max(ratios_c) for r in ratios_w); top = sum(sorted(H[gm], reverse=True).index(H[gm][l - 1]) == 0 for gm, l in WALLS)
print(f'walls whose h is the largest so far in their game: {sum(H[gm][l - 1] > max(H[gm][:l - 1]) for gm, l in WALLS)} of {len(WALLS)}; '
      f'walls with the largest h of the whole game: {top} of {len(WALLS)}')
# rank-based check: among held-out levels reached (cleared or wall), is h/median(earlier) higher for walls? (Mann-Whitney U by hand)
import itertools
u = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(ratios_w, ratios_c))
print(f'P(wall ratio > cleared ratio) = {u / (len(ratios_w) * len(ratios_c)):.2f} (AUC; 0.5 = no difference)')

# tokens per action and the actions a person needs (exploratory)
ho = [x for x in clr if x[0] not in DEV]
tpa_c = st.median(t / a for _, _, a, t, _ in ho if a); ah_c = st.median(r for *_, r in ho)
print(f'cleared held-out levels (n={len(ho)} level clears): median a/h {ah_c:.2f}, median tokens per action {tpa_c:,.0f}')
print('  wall: runs, actions taken (max), tokens on the level (max), tokens per action, actions taken / h (max), tokens to take a/h x h actions at this rate')
short = 0
for gm, l in WALLS:
    rows = tok.get((gm, l), []); h = H[gm][l - 1]
    if not rows: continue
    A = max(a for a, _ in rows); T = max(t for _, t in rows); tpa = sum(t for _, t in rows) / max(1, sum(a for a, _ in rows))
    need = ah_c * h * tpa; short += A < ah_c * h
    print(f'  {gm} L{l}: {len(rows)} runs, {A} actions, {T / 1e3:.0f}k tokens, {tpa:,.0f} per action, {A / h:.2f} of h, {need / 1e3:.0f}k tokens needed')
print(f'walls where no run took even the median cleared share of h (a/h {ah_c:.2f}): {short} of {len(WALLS)}')
tw = [sum(t for _, t in tok[(gm, l)]) / max(1, sum(a for a, _ in tok[(gm, l)])) for gm, l in WALLS if tok.get((gm, l))]
many = sum(max(a for a, _ in tok[(gm, l)]) >= ah_c * H[gm][l - 1] for gm, l in WALLS if tok.get((gm, l)))
print(f'summary: walls median tokens per action {st.median(tw):,.0f} against {tpa_c:,.0f} on cleared held-out levels; '
      f'walls where some run took at least the median cleared share of h and still did not clear: {many} of {len(WALLS)}')
