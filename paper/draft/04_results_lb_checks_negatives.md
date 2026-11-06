<!-- Draft v0, 4 Nov 2026 ~12:20 UTC. Results 4.1, 4.2 and 4.5 (no V5 dependence). 4.4 waits for the V5 readout.
     Sources: tables/t1_arc3_submissions.md (table_submissions.py), PLAN results log (budget retrodiction),
     v5_budget_prediction.out (half-time check), fig_survival_fit.out, lever4b_two_numbers.out, PLAN C3 line. -->

## 4. Results

### 4.1 Leaderboard
Table 1 lists all 17 of our ARC-AGI-3 submissions. The linked notebook, arc3-b12-x6dprime (public), is B12. Its first
submission scored 33.37, our best, which places 197th of 5,073 on the final public board. Four byte-identical
resubmissions scored 29.69, 28.24, 26.76 and 26.21 (mean of the five 28.85). The seven later submissions of the B12 harness (six
variants, B30 twice), whether they changed the agent or only its start-up handling, scored 23.82 to 29.88 (mean 25.89); none beat B12. Nor can we show that B12 beats
the notebook it started from. dfranzen's Milestone 2 notebook (B1), submitted twice, scored 28.30 and 26.72 (mean 27.51).
B12's five draws are 1.34 points higher on average, with a standard error of 2.40 at the per-submission SD of 2.87
(z = 0.56, p = 0.58); detecting a true gain of that size with power 0.8 would take about 72 submissions per arm. Our own method (Section
3.5; exploratory) agrees: on the levels of the 10 games that B1's two development runs played, the seven valid 121-minute
B12 runs need the same tokens per crack as B1 (b = +0.013, a factor of 1.01; bootstrap over runs, 90% interval -0.069 to
+0.092). B12's changes moved tokens between games; they did not make cracks cheaper. A fork of
a different public notebook (A1) scored 11.05. The private scores are not yet published; Section 4.4 gives our sealed
prediction for them.

### 4.2 Does the model describe the runs?
Two checks use data that the fit did not see. First, a fit on the first half of each control run predicts the full-run
level totals: 101.1 on average against 102.75 observed. Second, the same B12 notebook at one third of the budget (runs b12d25
and b12d25_r2 in Appendix B) cleared 43 and 45 levels; cutting the four full-budget runs at their token count per game gives 43, 36, 41 and 38 (mean
39.5). The token budget alone thus accounts for about 90% of the short runs' level count. Both short runs started only 17 of the 25 games; the cut uses
the short runs' mean tokens per game, so a game that neither short run started counts as zero on both sides. Two more checks are in-sample and
show only that the model's form fits. Fitted separately to each of six other teams' public runs, the model reproduces
that run's level total within 1.3 levels (Section 4.8). The fitted curve matches the survival of open attempts in token
time: at 62k tokens the observed share of attempts still open is 0.220 against 0.213 from the model (Fig. 2).

### 4.5 Our 32 negative results, sorted by what each change did
We sorted the negative results after the fact by what each change did to the run. The sorting describes them; it does
not test the model.
- Fewer tokens for the levels that count. Rolling memory summaries cut prefix-cache hits from 94% to 69% and agent turns
  by 29%, so the same budget bought fewer attempts. ThinkingCap, a variant of our language model tuned to think less,
  generated 3.52M tokens in the run instead of about 4.4M.
- Higher crack cost. Low reasoning effort halved the level count.
- No change to cracking. More concurrent streams and shallower speculative decoding (one or two draft steps instead of three; both in Appendix A) changed serving, not
  what the agent does; a scene-digest tool was never called (0 of 541 calls).
On the 14 valid FB25 runs with token logs, crack costs b range from -0.150 to +0.060, against an A/A SD of 0.051. None
of the 6 variants with valid runs passes a multiple-comparison line (Bonferroni over the 6 variants, |z| of 2.64 or more); ThinkingCap's cheaper
cracks come closest and reach the line only after rounding (one run). Next is the ht harness (mean b -0.095 over two runs, z 2.2), the one
candidate worth a third run; two-step speculative
decoding (spec2, mean b -0.048 over its two valid runs) is 1.1 standard errors from the control. An A/A check made after the outside
review asks how firm that line is. Left out in turn and tested like a variant (analysis/s37/aa_calibration.py), five of the
six same-agent runs sit within 0.4 standard errors of the control, but b32_spec2_r3 sets most of the A/A SD: without it the
other five spread only 0.015, and 5 of the 6 variants would cross the line. A level bootstrap of the same fit (400 refits,
analysis/s37/aa_bootstrap_b.py) sides with the six-run SD: it puts the standard error of a one-run variant's b at 0.057 to
0.072, b32_spec2_r3 at z -1.97, and none of the 6 variants past the line (largest |z| 2.09 for ThinkingCap, then 2.01 for ht).
The tight cluster of the other five runs is the unusual part, so 'none passes' stands, but with six A/A runs the line is soft. Two runs do not fit the two numbers. ThinkingCap's cost per level fell
(b -0.150) but it cleared 13.1 levels fewer than predicted, which in the model means more walls. B15C generated 23% more
tokens (5.45M) and still cleared 6.9 levels fewer than predicted: more supply did not turn into more cracks.
