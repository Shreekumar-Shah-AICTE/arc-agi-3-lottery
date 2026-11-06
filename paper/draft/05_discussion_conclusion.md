<!-- Draft v0, 4 Nov 2026 ~12:15 UTC. Sections 5-7. [V5] filled 4 Nov ~14:50 UTC. -->

## 5. Discussion: what this means for progress
**What can raise a score.** In the model, a change raises the score only if it raises cracks per GPU-second, and one run
shows the gain only if it is larger than the noise floor. Two routes qualify. One is more tokens for the games that need them: faster serving, fewer wasted tokens, fewer
games stalled at the end, and more patience for slow first levels (about half of the per-game noise is whether a game
clears level 1 before the harness's slot rule shelves it, Section 4.3; a guard against that shelving passed a sealed
one-run test, but 2 games never started, Section 4.4). Frontier results point the same way: keeping the model's reasoning and compacting the history
tripled GPT-5.6 Sol's ARC-AGI-3 public-set score (13.3% to 38.3%) with 6 times fewer output tokens [OpenAI26]. Doubling the play time bought 38 more levels (141 against 102.75 at 1x), more than the 24 the sealed
calculation predicted; in simulation the harness's endgame barely changes what doubling buys, but it shaped the control
runs that the calculation continued, which explains about half of the gap (Section 4.4 and Appendix D). The other is fewer walls. At the end of a control run 28% to 42% of the open attempts are walls in the model (refitted
with the doubled run, and fitted on the four controls; Section 4.3). A first look at the action logs splits them in two
(Section 4.3): on 6 of the 9 likely walls the agent took as many moves as it usually needs and still missed the goal,
while on long levels such as dc22 level 5 it thinks too long per move to get there, so cheaper moves are a route too. A
scheduler that knew which games were
walled would gain 13% to 17% in simulation, while one that only chases the highest crack hazard gains nothing (and loses 3.6%
when the games differ in difficulty), and our attempt to predict walls from the logs scored close to chance (AUCs 0.49 to 0.58;
Section 4.7). Changes
that only reshape how the agent talks or remembers, without changing either route, should be expected to fail, and ours
did.

**How to tell a real gain from luck.** One leaderboard draw per arm resolves only differences of 11.4 points or more. A
team's best of 100 identical entries sits about 7 points above its mean. Inside a crowded band, rank therefore depends
heavily on the number of entries and on luck, not only on the code. For agent comparisons in development, comparing crack cost on shared levels
needed several times fewer runs than comparing level totals (5.1 times on our six valid runs, at least 2.8). We suggest that papers and leaderboards report the SD and the
number of runs, and that leaderboards show the number of entries next to each best score.

**Why this matters for the 85% goal.** A field that reads leaderboards learns from differences. If many of those
differences between similar agents are luck, the field learns slowly. A model that predicts the size of the luck tells a
team when a result is worth acting on, and which kinds of change could ever beat it. In the model's terms the distance
is set by walls, not by time. In ARC's vocabulary, run luck sits on the compute side and walls on the skill side:
extra tokens buy the cracks near the budget edge, but a wall does not crack within any budget we can afford. In this
model, then, the gap to 85% closes through skill acquisition (fewer walls), not through more play alone. Frontier results suggest what that skill is.
The hosts report that once a frontier model understands a game's mechanics, it plays within the range of human action
efficiency [ARCAstra26], and in our logs most walls are levels where the agent made its usual number of moves and still
missed the goal (Section 4.3): failures to understand rather than to execute. A harness that keeps the model's reasoning
state between requests lifted one frontier model from 62.7% to 99.9% [ARCAstra26]; our X6 ledger, which keeps each solved
level's rule in the prompt, is a small open-model version of the same idea. At the wall rate fitted on the four controls (6% of levels), even unlimited play caps this agent at
about 73% on the 25 public games, and at the 4% refitted with the doubled run, at about 81%. On the 15 held-out games,
the better guide to the hidden games (Section 4.3), the caps are lower: about 56% at their four-control wall rate
(q 0.110) and 69% at their eight-run rate (q 0.070). On either game set, 85% needs a wall rate of
3.1% or less, and without walls it would take about 1.6 (four-run fit) to 1.9 (refit) times the current budget. The 100% line of the ARC-AGI-3 grand prize needs no walls and move counts that match humans on aggregate
(matching them on every level is enough but not required; an extrapolation of the fitted model, not a measurement).

## 6. Limitations
- One agent family (the Duck harness through dfranzen's notebook and our B12), one language model, one GPU type.
- Development runs use the 25 public games. The hidden games are harder (B12's four control runs average 47.3 on the public
  games in development, against 28.85 on average on the leaderboard), so the fitted numbers describe the public games. The 10 games we developed on show no walls and the 15 we held out
show all of them (Section 4.3), so the pooled fit flatters the agent on new games.
- The harness schedules games differently over the last 40% of each game's budget, so the end state of a run is not the
  state of a longer run at the same minute. The sealed V5 calculation continued such end states; the simulator that sizes
  this (Appendix D) is fitted to the same runs and misses one of our checks.
- The model treats walls as permanent. In the doubled run, attempts past 150k tokens cracked twice as often as the fitted
  hazard expects (6 against 2.9), so some walls may be very slow cracks. Refitted with that run, the wall share falls from 6% to 4% (Section 4.3).
- Most analyses are exploratory: they were written after seeing data. Only the V5, V10 and V11 predictions and the
  private-leaderboard predictions were sealed before their data existed; V10 went unscored, and V11 is one partial run
  (Section 4.4).
- Games wait for a turn: the harness plays at most 10 games at a time and counts every game's budget from the start of
  the run, so the last games to start get less time (in the four controls and T1c the last one started 69 to 99 minutes
  into its 121, and in the slower T1 runs and in V11 some never started). Start order follows the number of levels, so
  the longest games start last (Appendix C).
- Some harnesses couple their passes (Section 4.8), so our noise estimates may not transfer to them.
- The private leaderboard is still withheld, so the sealed private-board predictions are not yet scored.

## 7. Conclusion
Identical code on the ARC-AGI-3 leaderboard spans 1,723 places because each level waits for a crack whose timing is
luck, and the luck falls on the hard levels near the edge of the budget; for a game stuck on its first level that edge
is the harness's slot rule, which can shelve it early in the run (a guard against this passed a sealed test in one partial run). A three-number model fitted on four runs explains
the size of that luck, our negative results, and the plateau of the ARC-AGI-2 board; on 120 public SWE-Bench-Verified
runs, the same per-task luck accounts for most of the run noise (exploratory). A prediction sealed before a run
at twice the play time held: 141 levels against 127 (90% interval 110 to 145), near the top; in simulation about half
of the gap traces to a harness endgame rule that shaped the runs the sealed calculation continued. The model also says what to do: buy tokens
for the games that need them, attack walls, and count runs before believing a gain.
