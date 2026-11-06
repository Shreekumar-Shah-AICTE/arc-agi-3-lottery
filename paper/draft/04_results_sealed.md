<!-- Draft v0, 4 Nov 2026 ~12:20 UTC. Results 4.4. Source: analysis/s37/sealed/ (read-only), score_sealed.py,
     v5_readout.py. [V5] filled 4 Nov ~14:50 UTC; the two private outcomes say pending (6 Nov); replace them when the boards exist.
     7 Nov 2026: the full rules and readouts moved verbatim to paper/tables/tD_sealed.md (Appendix D); one paragraph per test stays here. -->

### 4.4 Sealed predictions
The five sealed tests at a glance (times UTC; one paragraph per test below, and every rule and readout in full in Appendix D):

| Test | Sealed | Public time stamp | Outcome |
|---|---|---|---|
| Private ARC-AGI-3 board, lottery model H against low noise S | 2 Nov | hash posted on Kaggle (discussion 755123); Bitcoin block 974295 | pending |
| V5: twice the play time | 4 Nov, pushed to GitHub at 10:18; run started 10:28 | Bitcoin block 974519 at 10:52, 24 minutes into the run | pass: 141 levels, inside 110 to 145 but near the top |
| V10: less token supply | 4 Nov, committed at 16:42 | Bitcoin block 974554 | unscored: both runs left games unplayed |
| V11: first-level guard | 5 Nov, committed at 14:32; kernel pushed 14:40 | Bitcoin block 974681 at 15:34 | pass in one partial run: 1 of 23 started games at 0 levels; 2 never started |
| Private ARC-AGI-2 board (V9) | 4 Nov | hash in a comment on the same Kaggle post; Bitcoin block 974554 at 17:27 | pending |

Until its Bitcoin anchor, a sealed file is dated only by its GitHub push.

**Private ARC-AGI-3 board (sealed 2 November; pending).** Before Kaggle published the private board, we wrote down what
the lottery model (H) and a low-noise alternative (S) predict for the crowd of teams with final public scores from 20 to
just under 40 (2,259 teams when we fitted the model). The two disagree sharply: for example, the rank correlation of
public and private scores within the crowd is 0.39 under H and 0.90 under S, and H fails above 0.70. Four such tests and
a decision rule are fixed in the file; score_sealed.py applies the rule when the hosts publish the board.

**Twice the play time (V5, sealed 4 November; pass).** The crack model predicted that doubling the play time per game,
everything else byte-identical, would raise the level total from 102.75 to 127 (90% interval 110 to 145). The run
cleared 141 levels (Fig. 5), and the three secondary checks also landed inside their ranges. The pass is not
comfortable: 141 is 14 above the point prediction and five below the "out of time" line. A later simulation of the
harness's slot rule traces about half of the gap to an endgame rule that shaped the control runs the sealed calculation
continued (Appendix D).

**Less token supply (V10, sealed 4 November; unscored).** With speculative decoding off, a run that generates fewer tokens
on the same clock should play like a control run stopped early. Both runs broke the file's validity rule, because games
were left that never started (2 in the first run, 1 in the rerun), so the prediction is untested. Exploratory only: their
totals, 86 and 82 levels, lie inside the sealed bands at their measured supply.

**A first-level guard (V11, sealed 5 November; pass in one partial run).** A game on level 1 gets top slot priority until
it clears the level or has spent 150k tokens on it. The sealed rule passes the guard if at most 1 of the games that start
ends with 0 levels; the seven reference runs had 1 to 4, so with no effect a pass has about a 1 in 7 chance. V11 had 1 of
the 23 games that started, but 2 games never started; counting them as zeros gives 3.

**Private ARC-AGI-2 board (V9, sealed 4 November; pending).** The lottery model, refitted to the plateau of Section 4.8,
predicts how its 2,133 teams reorder on the private board: for example, a rank correlation of public and private scores
of 0.82 (0.65 to 0.91). Three named alternatives carry their own values, and score_v9.py applies the rule when the board
is published.

Appendix D gives each file's full predictions, decision rule, time stamps and readout, with the analyses made after the
V5 and V11 results.
