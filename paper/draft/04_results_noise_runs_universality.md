<!-- Draft v0, 4 Nov 2026 ~12:10 UTC. Results 4.3, 4.6, 4.7 and 4.8 (no V5 dependence). 4.1, 4.2, 4.4 and 4.5 follow.
     Sources: lever3_noise.out, PLAN results log (V2), lever3d/3_stuck, lever4/4b, lever5/5b, arc2_plateau_lottery.out,
     lever2_external.out, lever2b_franzen_4pass.out, S37_HANDOFF.md section 8 (ARC-AGI-2 subgroup draws). -->

### 4.3 The noise is luck near the budget edge
Six FB25 runs of one agent (four controls and two with a serving-only change) cleared 95 to 107 levels (mean 102.5,
SD 5.3). Three findings locate this noise.

It is per game (Fig. 3). Across the 13 valid runs of the B12 family, the per-game score variances, summed over games, predict an SD of the
run mean of 3.79 against 3.23 observed. Per-game luck therefore accounts for 138% of the run-level variance (bootstrap
90% interval 88% to 260%): most or all of it (the interval reaches down to 88%), with games slightly negatively coupled. Per-game scores are uncorrelated within a
run (mean pairwise correlation -0.007), and across the 13 valid runs of this agent family 22 of the 25 games change their level count between runs, by up to 7 levels (Fig. 3 shows six of these runs).

It is not throughput. Total tokens per run vary little (4.31M to 4.60M, a CV of about 2.5%) and do not track levels
(correlation -0.24). We re-read each run's logs at a fixed token budget per game instead of the fixed clock. That halves
the per-game variance early in the run (ratio 0.47 at a quarter of the budget) but not at the end (0.97 at the full
budget; Fig. 4). At half the budget, run totals are steadier than independent games would make them (permutation test,
p = 0.021, but one of four budget points tested, so it does not survive a Bonferroni correction), as expected when
games share one server and trade tokens.

It sits on the hard levels, and they stay hard. A model that redraws every level's difficulty and walls in every run
(H0) gives a run-to-run SD of 12.3 levels. Fixing each level's difficulty but redrawing walls (H1) gives 11.1. Fixing
both (H2) gives 5.6 (10-90% range 3.9 to 7.4) against 5.3 observed. With T1c, a seventh run of this agent made later (Appendix D), the observed SD is 6.0, still inside that range. The noise needs persistent difficulty, not walls: without walls, a
lognormal wait whose level difficulty persists across runs (centred on the no-wall fit below) gives 6.0 (10-90% 4.9 to
7.1), and one that redraws difficulty in every run gives 11.1 (analysis/s37/noise_no_walls_7nov.py; exploratory). At the end of a run about 17 attempts are still open
(69 across the four controls); the four-run fit calls 42% of them walls and the rest slow cracks.

The rest of this section is exploratory, written after the doubled run (V5, Section 4.4) and the token-supply runs.

**Fewer walls than the four-run fit says.** Refitted on all eight runs of this agent (1,003 attempts, including V5), the
wall share falls from 6% to 4% of levels, and the refit calls 28% of the controls' open attempts walls. It has seen V5,
so this is a consistency check, not a test: it expects 101.2 levels at 121 minutes and 140.2 at twice the budget
(observed: 102.75 on average and 141). Fourteen levels were reached but never cleared in any of the eight runs; the
refit makes 9 of them walls with probability 0.5 or more, the clearest being level 5 of dc22, left open after 880k
tokens in total. If those nine are fixed walls, unlimited play caps this agent at 154 of the 183 public levels (Appendix
E).

**Walls or a long tail?** The outside reviews asked whether a long-tailed wait with no walls fits as well. We fitted four
waiting-time families, each with q fixed at 0 and with q free, to the sealed fit's attempts with its censoring
(analysis/s37/rivals_no_walls_7nov.py). In sample, walls win, but weakly: without them the lognormal loses 3.6 to 4.7
log-likelihood units on the four controls, the seven 121-minute runs and those plus V5 (AIC 5.3 to 7.5 worse, BIC only
1.1 to 2.7 worse). Log-logistic waits fit worse (AIC 10.5 to 20.3), Weibull and power-law (Lomax) waits much worse (28.2
to 159.1). Out of sample the order flips. Left out one game at a time, the lognormal without walls predicts the left-out
game 3.4 units better than the crack model; fitted on the seven 121-minute runs, it scores V5's attempts 0.8 units
better; and the sealed V5 calculation redone without walls gives 145.7 levels (log-logistic: 142.0) against 141
observed, where the crack model gives 128.9. Past 150k tokens V5 cleared 6 of the 12 attempts that got there, between
the crack model's 3.8 and the no-wall lognormal's 10.6. Up to twice the play time, then, the data do not separate
permanent walls from a long lognormal tail, and the unlimited-play caps of Section 5 hold only if walls are permanent.
The two readings part at four times the play time: 141.2 levels with walls (5-95% 124 to 158) against 172.3 without
(162 to 181) of the 183, a test we have not run.

**Do the walls settle with more play?** A wall share is identified only if attempts are followed until survival levels
off [MallerZhou96], and ours has not. With every attempt of the eight runs censored at a token cap and the model refitted
(analysis/s37/walls_followup_7nov.py), q on the held-out games falls from 0.126 with a 200k-token cap to 0.080 with no
cap (all 25 games: 0.052 to 0.038), and the Kaplan-Meier survival of attempts is still falling at the longest waits,
though only 10 attempts are still open past 200k tokens. A bootstrap over games, which counts a wall that recurs in
every run once, gives the eight-run fits wide 90% intervals: q 0.017 to 0.138 on the held-out games (cap 48% to 91%)
and 0.003 to 0.075 on all 25 (cap 67% to 98%). In 10% of the held-out resamples and 34% of the pooled ones, q is at or below the 3.1% that
85% needs (Appendix E).

**The walls are in the games we did not develop on.** Our early development runs played 10 of the 25 public games (Fig.
1, right); the other 15 were a holdout set, played only to check candidate changes (Appendix B), until the 121-minute
runs used all 25. Fitted separately on the four controls, the 10 dev games
show no walls (q 0.000, median wait 16.9k tokens, s 1.00; 90% of their levels cleared) and the 15 held-out games all of
them (q 0.110, median 35.7k, s 1.05; 34% cleared). On all eight runs the held-out games give q 0.070 and a median of
40.3k. The pooled three numbers therefore average two different populations, and the held-out games, on which we fitted nothing and ran only two early design checks (X2 and X6, 1 and 2 October; Appendix B), are probably the better guide to the hidden leaderboard games. Section 5 gives the unlimited-play caps for both
game sets.

**A first look inside the walls.** An exploratory check that uses only the action logs (Appendix E) compares the actions
the agent took on each of the 9 likely walls with the human baseline h that the scorer uses. On the held-out levels it
clears, the agent needs a median of 0.72 h actions. In 6 of the 9 walls, some run took at least that many and still
found no way through, so these look like failures to find the goal or the mechanic, not a shortage of moves. In the
other 3, no run came close: dc22 level 5 asks people for 324 actions, and at the agent's 1,950 tokens per action there,
reaching 0.72 h would take about 456k tokens, roughly twice the 218k it got in the doubled run.

**Half of the per-game noise is the first level.** Levels are played in order, so a game that never clears level 1
scores nothing. Over the seven 121-minute runs, the split between 0 levels and at least one accounts for 49% of the
summed per-game variance (bootstrap over runs, 90% interval 39% to 57%), almost all of it in four games (sc25, cd82,
dc22 and g50t). These zeros are not walls: 14 of the 17 game runs that ended with 0 levels were shelved by the harness's
slot rule, which at each context trim puts a game behind games of higher priority, and a game's priority falls with the tokens and actions spent on its current level (Appendix C), and in
V5 all 6 of these games cleared level 1. A game's first crack therefore races the slot rule's patience as well as the
clock (Appendix E). A sealed one-run test of a guard that protects games on level 1 from shelving was inconclusive (V11, Section 4.4): by
the letter it passed, with 1 of the 23 games that started ending with 0 levels, but 2 games never started. In the
harness simulator of Appendix D (exploratory), the slot rule moves this noise more than it makes it. Over 400 worlds of
the fitted model, seven runs each and the same random numbers for every scheduler, sharing the tokens equally with no
queue gives a run-to-run SD of 5.5 levels and the slot rule 5.8, so the crack waits alone carry 90% of the run-to-run
variance (bootstrap over worlds, 85% to 95%). What the rule changes is where the noise sits: the split between 0 levels
and at least one carries 7% of the summed per-game variance with equal sharing and 39% with the rule (observed 49%), and
zero-level games rise from 1.7 to 3.1 per run (observed 2.4). The rule's priorities still pay, 2.2 levels per run over
taking turns, while the V11 guard costs 1.4 levels per run (Appendix E).

### 4.6 How many runs it takes to see a real change
On the leaderboard one submission has an SD of 2.87 points. To detect a true difference with power 0.8 (two-sided test
at 0.05) takes 15 submissions per arm for a gain of 3 points and 6 for a gain of 5; one run per arm resolves only 11.4
points. The SD comes from five submissions, so the 15 could be anywhere from 7 to 81. Resubmission alone buys points:
the best of 10 identical submissions beats their mean by 4.42. In development runs, crack cost b needs several times
fewer runs than raw level totals: 4.5 against 22.6 per arm on the 6 valid same-agent runs, and at least about 2.8 times
fewer when any one run is left out. On the five true controls alone (Section 4.5) the factor is 15.0. A CPU notebook (paper/notebook/how_many_runs.ipynb in our repository, also public on Kaggle at
kaggle.com/code/shreekumarshah/how-many-runs-do-you-need-arc-agi-3) computes these run counts and fits b for two arms from a team's own run logs; the repository copy also turns a fit of the three numbers into the cap that unlimited play would reach if walls are permanent, with a bootstrap over games. Appendix E gives the t-test counts and the full best-of-n
series.

### 4.7 Methods the model rules out
We simulated schedulers under the fitted model before spending GPU time on them. Giving the 10 streams to the games with
the highest current crack hazard gains +0.0% over equal sharing when games are alike and -3.6% when their difficulty
differs, while an oracle that knows which games are walled and skips them gains +16.9% and +13.4%. A scheduler is
therefore worth building only with a wall detector, and a detector trained on run logs was close to chance
(leave-one-game-out AUC 0.491 to 0.582; Appendix E), so we did not build it.

### 4.8 The same lottery elsewhere
ARC-AGI-2. On the final public ARC-AGI-2 board (3,627 teams), 2,133 teams (59%) sit on a plateau between 27 and 35. If
these teams mostly resubmit near-identical public notebooks, each team's score is roughly the best of its n draws.
Without fitting anything to the board, the best of n draws from seven scored draws of one public notebook lineage (mean
29.66, SD 0.81; Table 2) predicts a plateau mean of 29.65 for single-entry teams rising to 31.67 at 91 entries; observed
means rise from 29.41 (282 teams with one entry) to 32.28 (64 teams with 64 or more) (Fig. 6). From 1 to 91 entries, run luck alone, at the pooled SD of our unchanged resubmissions (0.61), gives 1.50 of the observed 2.87 points (52%). The SD of all seven draws gives 2.01 (70%), but it pools two notebooks (A0 three times, PP four times; their means differ by 1.11 points), so that share includes a code difference. The rest of the rise mixes small code differences with teams improving their code between entries. The A0 parent's public score, 33.89 (Table 2), fits this reading: on the board of 9 October only 3 teams sat at 33.89, while the public perfpatch family clustered from 30.14 to 32.22, so we read 33.89 as a lucky draw of that family rather than its level (inference). Run luck is thus
about half of the effect, with wide bounds from only seven draws (Appendix E). Section 4.4 gives a sealed prediction of
how this plateau reorders on the private board.

Other teams' harnesses. Fitted separately to six public runs by other teams, the crack model reproduces each run's level
total within 1.3 levels. Four forks of the Duck harness show our median crack wait (20.2k to 28.1k tokens) but more
walls (q from 0.22 to 0.37); two runs show no walls (q = 0.00).

Another benchmark (exploratory, added 7 Nov). Bjarnason et al. ran each of three models in two scaffolds, at two
temperatures, ten times on SWE-Bench-Verified and published the runs [Bjarnason26]. We applied the check of Section 4.3
to all 12 configurations (120 runs over 500 tasks): independent per-task luck, computed from each task's success rate
across a configuration's runs, predicts 86% of the observed run-to-run variance of the resolution rate when pooled,
inside the 81% to 127% that pure per-task luck would give (near its low end), and 51% to 200% per configuration. Between
144 and 309 of the 500 tasks are solved in some runs and not in others. On this coding benchmark too, the run noise is consistent with independent per-task luck. The check tests only that independence, not the crack model: whether the noise also sits near a budget edge would need a fit to their token logs (Appendix E).

One caveat on independence. dfranzen's public notebook plays four passes over the 25 games, and their level totals (102,
106, 104 and 105) vary far less than independent games would make them (p = 0.020), which fewer level-1 zeros do not
explain (Appendix E). Some harnesses therefore couple passes, and for them our independent-game noise is an upper bound.
We have not found the mechanism.
