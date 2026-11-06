<!-- Appendix C, written 5 Nov 2026 from the B12 notebook (kaggle/b12_fb25/b12_fb25.ipynb: the harness settings and the
     embedded ours_form_priority.py, d_parts) and analysis/s37/out/walls.out. Numbers are registered in build_results.py. -->
**What the agent is.** B12 is dfranzen's public Milestone 2 notebook on Tufa Labs' Duck harness (Jeroen Cottaar) with
two changes (Section 3.1): the solved-level ledger (X6), in which the rule the model wrote for each solved level stays in
the prompt for the whole game, and the slot-priority rule D' from shiiin9's public notebook (Form D1). We credit all
three; the harness and nearly all of the agent are theirs.

**One turn.** Each game runs in its own thread with its own conversation. Each turn the model sees the current frame as
an image with the recent history, thinks, and answers with a tool call: an action, or Python code whose functions persist
for the rest of the game. Settings: a 128k-token context window, at most 12k generated tokens per turn, temperature 0.7.
When the history outgrows the window (or 150 turns), the harness drops the oldest turns in one large cut (about 58k
tokens; a drain) rather than a few at a time, because any cut at the front invalidates the server's cached prefix.

**Slots.** The server runs 10 generation streams, and a gate lets at most 10 games hold one. A game keeps its slot while
its cached prefix is intact. After a drain its next request must be prefilled anyway, so that is when it hands its slot
back and competes again with a priority computed from its current state; the highest priority in the queue gets the
slot, and the game that handed over may win it straight back. Games that have not started wait in the same queue. For a
game on level l of N, with a actions and t generated tokens spent on that level:

```
priority = A x M x C + B x phi
A    = (1 + 0.5 (l - 1)) x norm(N) x (300 / (300 + a))^2.5
norm = 55 / (N' (N' + 1) / 2), N' = N clipped to 6..10
M    = clip((30,000 / pace)^0.4, 0.25, 4); 1 before the first clear
C    = 0.1 max(0.1, 1 - a / 115) + 0.9 max(0.1, 1 - t / T)
T    = 225,000 x clip(pace / 30,000, 0.5, 2)^0.5; pace = 30,000 before the first clear
B    = 16, 14, 10 or 0 with 3 or more, 2, 1 or 0 levels left
phi  = 1, falling linearly to 0 over the last 40% of the run
```

A rewards depth and few actions (norm is the weight of a level in the game's score), M rewards a fast pace (pace: the
game's mean tokens per cleared level), C is the patience on the current level, B is a bonus for levels left, and phi
fades that bonus out at the end (the endgame of Appendix D).

**What the rule does.** A game that has not started has a = t = 0, so its priority is norm(N) + 16 (every public game has
at least 6 levels): games with fewer levels start first, and the longest games start last. A game stuck on its level
loses priority with every token and action. At the median state of the 14 shelved 0-level game runs of Section 4.3 (51
actions and 77k tokens on level 1), A x M x C is 0.44 x norm(N) against norm(N) for a game that has not started, so at
its next drain the stuck game yields its slot to almost any waiting game, and later to games that keep clearing levels.
Over the last 40% of the run the headroom bonus fades out and the order is A x M x C alone, which favours deep levels
and fast games.

**Long games start last.** The slot rule ranks the games still waiting for their first turn by the value of their first
level, which is highest in games with few levels (above). Ten games start at once and the others roughly in order
of their number of levels, so lf52 (10 levels) starts at minute 89.3 of 121 on average. The 9 games that start after
minute 30 clear 0.48 of their levels against 0.65 for the rest, but the gap is mostly the games themselves: within a
game the start minute varies little between runs (SD 9.1 minutes), and a later start made no detectable difference
(-0.02 levels per 10 minutes, permutation p = 0.767). Late starts matter when serving slows down or slots are held
longer: then the last games never start (T1 and V11, Section 4.4).
