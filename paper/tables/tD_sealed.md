<!-- Appendix D, 7 Nov 2026: the full text of Section 4.4 as of 6 Nov (moved verbatim from paper/draft/04_results_sealed.md).
     Numbers are registered in build_results.py and checked by check_numbers.py. -->
**The private ARC-AGI-3 leaderboard (sealed 2 November 2026).** Before Kaggle published the private board, we wrote down
what the lottery model (H) predicts and what a low-noise alternative (S) predicts. In H, a run's score on each half of the
hidden games is the agent's skill plus independent run noise with an SD of about 2.9 points, and a team's public score is
its best run, so teams with more entries hold more tickets. In S, the noise is small (SD 1.0) and skill grows by 0.5 points
per doubling of entries. On the public board alone, S and H with a noise SD of 2.3 fit about equally well (H with our
measured 2.87 fits somewhat worse), so only the private board can decide. The predictions concern the teams with final
public scores from 20 to just under 40 (the crowd C; 2,259 teams when we fitted the model):
- P1: rank correlation of public and private scores within C: H 0.39 (0.28 to 0.57), S 0.90; H fails above 0.70.
- P2: share of C's public top 10% that stays in C's private top 10%: H 30% (20% to 42%), S 71%; chance is 10%; H fails
  above 55%.
- P3: private-score gap between teams with 64 or more entries and teams with one (5.42 points on the public board):
  H 0.9 (0.0 to 1.7), S 3.6; H fails above 2.5.
- P4: how much more C's public top 10% falls from public to private than C as a whole: H 2.7 points (1.9 to 3.3), S 0.70;
  H fails below 1.3.
- P5: rank correlation among teams scoring 40 or more: H 0.89 (0.78 to 0.96); H fails below 0.60. Noise reorders the
  crowd, not the top.
- P6: our own runs: the private SD of our 8 B12-family submissions (the five B12 runs, B28 and both B30 runs in Table 1)
  is 1.7 to 4.2; our best public draw (33.37) falls back to about
  28 (22.5 to 34.5) after the board-wide shift.

The decision rule was fixed in the same file: H passes if none of P1 to P4 crosses its line and at least three fall in
their ranges; H fails if two or more cross. Two outcomes are covered by neither clause: exactly one crossing, and no crossing with fewer
than three in range. A dated note kept apart from the sealed file (6 November) says both will be reported as undecided,
with every value shown. The file's SHA-256 (2c5b49a9...) was posted on Kaggle (discussion 755123) and
time-stamped in Bitcoin block 974295. Outcome: pending. The hosts had not published the private
leaderboard when this version was written; score_sealed.py applies the rule above when they do, whatever the result.
Anyone can run it from the repo with three CSV files (the final public board, the private board and our submissions
list): `python3 analysis/s37/score_sealed.py PUBLIC.csv PRIVATE.csv OURS.csv`.

**Twice the play time (V5, sealed 4 November 2026).** The crack model predicts that doubling the play time per game
(242 instead of 121 minutes, everything else byte-identical) raises the level total from 102.75 (the mean of the four
1x controls) to 127 (90% interval 110 to 145), with 13 games completed (10 to 16) instead of 7 to 8. The sealed rule: a
total from 110 to 145 passes the model; 109 or less fails it toward "stuck" (the games that hit the 121-minute limit were
mostly at walls); 146 or more fails it toward "out of time" (progress kept coming at close to the first-half rate).
The run was valid: all 25 games played, no tracebacks, 8.49M generated tokens. It cleared 141 levels, inside the sealed
interval, so by the sealed rule the crack model passes (Fig. 5). The three secondary checks also landed inside their
ranges: 11 games completed (predicted 13, 10 to 16), 105 levels by minute 121 (about 103, 93 to 113), and a rank
correlation of 0.48 between predicted and actual per-game gains over the 19 open games (sealed median 0.22, 90% -0.23 to
0.54). The pass is not comfortable. The total is 14 levels above the point prediction, five below the "out of time"
line, and close to the log-linear alternative stated in the same file (about 140); saturation (about 103) and linear
growth (150 to 175) are ruled out. The sealed file was pushed to our GitHub repository at 10:18 UTC on 4 November, ten
minutes before the run started (10:28 UTC); its public OpenTimestamps proof is anchored in Bitcoin block 974519 (10:52
UTC), 24 minutes into the run and 3.6 hours before it finished (14:31 UTC).

Afterwards we checked where the extra levels came from (exploratory). The harness's slot-priority rule has an endgame:
over the last 40% of each game's budget it fades out a bonus for games with levels left, which moves streams to games
close to finishing. At 121 minutes per game the endgame starts at minute 72.6; at 242 minutes, at minute 145.2. The
sealed calculation continued the control runs' end states, which already include their endgame: 7 to 8 games finished
and the rest on old attempts (mean 77k to 100k tokens). V5 reached minute 121 before its endgame, with 3 games finished
and younger attempts (mean 53k). From V5's own state the sealed calculation gives 42.9 further levels instead of 24.4 to
27.1, and with the tokens each game actually generated afterwards the model expects 38.5, against 36 observed. Between
minutes 72.6 and 121 the six games stuck at 1x (at most 1.5 levels on average in the controls) received 0.74M tokens in
V5 and went from 6 to 13 levels; in the control runs' endgame they received 0.21M to 0.52M and gained 1 to 3. A second
opinion committed after the seal but before the result, a simulator in which finished games hand their streams to the
others, had predicted 135 to 139; it has no endgame either. The one sign that walls are less permanent than the model
assumes is in the long tail: attempts that passed 150k tokens cracked 6 times against 2.9 expected from the fitted
hazard (Poisson p = 0.073).

A later simulation sizes the endgame's share (exploratory; written after the V5 result and committed before the V11
readout). It plays 100 worlds of the crack model with persistent level difficulty and walls (H2, Section 3.4) through
the harness's slot rule (Appendix C) on 10 slots, with and without the endgame. It comes close on both budgets:
106.5 levels at 121 minutes (observed 102.75) and 139.2 at 242 (V5: 141). The endgame barely changes what doubling
buys (32.7 levels with it, 31.9 without), and a 121-minute run continued to 242 minutes ends at the same total as a
242-minute run. What it changes is the sealed calculation's input. Applied to 4 simulated control runs per world, the
sealed calculation predicts a gain of 24.5 levels (from the real controls: 24.4 to 27.1), 8.3 short of the simulated
gain; from control runs simulated without the endgame it falls 1.5 short. The difference, 6.8 levels (SE 0.8; 5.9 and
10.0 if a context trim, which hands the slot back, comes every 80k or 40k generated tokens instead of 58k), is about
half of the 14 between 127 and 141. V5 ran another 5.7 above 127 plus the simulated shortfall, about one run-to-run SD. The simulator misses one check: from its 242-minute runs'
state at minute 121 the sealed calculation falls 5.0 short, while from V5's state it gave 42.9 against 36 observed.

**Less token supply (V10, sealed 4 November 2026).** If the agent is limited by tokens, a run that generates fewer
tokens on the same clock should play like a control run stopped early. We sealed this as a table before launch
(committed at 16:42 UTC; Bitcoin block 974554): with speculative decoding off (T1: the same notebook, games and
121-minute budget), the level total should follow the controls' mean clock curve at minute r × 121, where r is T1's
generated tokens over the controls' mean. A control run on the same day (T1c) checked for drift (92.7 to 112.8 levels,
tokens within 5%). The file's validity rule asks for all 25 games played, and an invalid run is rerun once. Both T1 runs
were invalid. They generated 0.72 and 0.73 of the control tokens, and games were left that never started (2 in the first
run, 1 in the rerun). The harness plays at most 10 games at a time, and every game's budget counts from the start of the
run, so a game that waits for a turn loses time: in the four controls and T1c the last game started 69 to 99 minutes in.
With slower serving, the last games never got a turn. By the sealed rule the supply prediction is untested.
Exploratory only: the two totals, 86 and 82 levels, lie inside the sealed bands at their measured supply (86.7, 77.6 to
96.0; and 87.2, 78.1 to 96.4). T1c was valid and showed no drift: 112 levels with 0.997 of the control tokens. It is the
highest of the seven same-agent runs and raises their SD from 5.3 to 6.0 (the simulator gives 5.6).

**A first-level guard (V11, sealed 5 November 2026).** If most zero-level games are shelved rather than walled
(Section 4.3), keeping games on level 1 from being shelved should remove most zeros. We sealed this before launch
(committed at 14:32 UTC; kernel pushed at 14:40 UTC; Bitcoin block 974681, 15:34 UTC). V11 is the control notebook
with one change: a game that has started on level 1 gets top slot priority until it clears the level or has spent
150k tokens on it.
The sealed rule counts the games that start and end with 0 levels: PASS if at most 1, FAIL if 2 or more. The seven
reference runs had 1 to 4 (mean 2.43), so with no effect a pass has about a 1 in 7 chance. V11 passed with 1 of the 23
games that started: tn36, which had spent 161k tokens on level 1, past the guard's limit. All six games with zeros in
the reference runs cleared level 1, sk48 (a zero in 6 of the 7) included. The run was partial, though: 2 games never
started (lf52 and su15), which the sealed file reports instead of rerunning; counting them as zeros gives 3. The total,
103 levels, matches the reference mean (103.9, SD 6.0), with 0.86 of the control tokens. One run is a weak test. In the
scheduler simulation above, committed before the readout, the guard cut zero-level games from 3.25 to 2.21 per run and
raised the games that never start from 0.20 to 0.63. Rerun after the readout at this run's token supply (exploratory),
the simulation leaves 2 or more games unstarted in 18% of runs without the guard and in 59% with it (9% with the guard
at full supply), so the guard and the lower supply probably both played a part; it also expects fewer levels at that
supply (90.5) than V11 cleared.

**The private ARC-AGI-2 leaderboard (V9, sealed 4 November 2026).** After the ARC-AGI-2 close and before Kaggle
published its private board, we refitted the lottery model to the plateau of Section 4.8 and sealed what it predicts. A
team's public score is its skill plus the best of its noisy draws; a fitted share of its entries (30%) count as draws, and
skill may rise with the number of entries (fitted: 0.3 points per doubling). The public board's own fit is best at a
noise SD of 1.0, but that fit leans on the assumed shapes of skill and entry counts, so the primary prediction uses the
SD measured on our own reruns: 0.6 points per half-board, the pooled SD of our same-code draws. The predictions concern
the 2,133 teams with public scores from 27 to just under 35:
- P1: rank correlation of public and private scores: 0.82 (0.65 to 0.91).
- P2: share of the plateau's public top 10% that stays in its private top 10%: 60% (43% to 73%); chance is 10%.
- P3: private-score gap between teams with 64 or more entries and teams with one (2.87 points on the public board): 2.0
  (1.0 to 2.3).
- P4: how much more the public top 10% falls from public to private than the plateau as a whole: 0.48 points (0.29 to
  0.88).
- P5: rank correlation among the 106 teams scoring 40 or more: about 0.99; the model fails below 0.90.

Three named alternatives carry their own values for P1 to P4: a noise SD of 0.81 (all seven of our draws) gives 0.69,
48%, 1.6 and 0.72; an SD of 1.0 (the public board's best fit) gives 0.52, 37%, 1.2 and 1.08; low noise (SD 0.3, with the
entry-count gap mostly skill) gives 0.95, 79%, 2.4 and 0.17. The primary passes if at least three of P1 to P4 fall in
their ranges and fails if two or more fall outside on the same side. A second test does not depend on the chosen SD:
each of P1 to P4 implies an SD, and the model's structure passes if the largest implied SD is at most 1.5 times the
smallest. The file's SHA-256 (1fffeeef...) was posted on Kaggle as a comment in discussion 755123 and time-stamped with
OpenTimestamps (Bitcoin block 974554, 17:27 UTC). Outcome: pending, as for
ARC-AGI-3; score_v9.py applies the rule above when the private board is published
(`python3 analysis/s37/score_v9.py PUBLIC.csv PRIVATE.csv`).
