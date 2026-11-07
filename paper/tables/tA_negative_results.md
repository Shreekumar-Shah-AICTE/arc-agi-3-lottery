| # | Ledger id | Date | Track | Kind | Hypothesis (short) | Verdict |
|---|---|---|---|---|---|---|
| 1 | E1b-qwen | 2026-10-01 | ARC-AGI-2 eval 120 | ARC-AGI-2 | Qwen3.8-27B FP8 L4x4 K=4 16k | KILLED |
| 2 | B3v2-X2-rep | 2026-10-01 | ARC-AGI-3 | rerun | X2 identical replicate | NOISE |
| 3 | B5-X2-HO-res | 2026-10-01 | ARC-AGI-3 | GPU change | X2 on 15 holdout games | KILLED |
| 4 | A1-LB | 2026-10-02 | ARC-AGI-2 | ARC-AGI-2 | A0+dyncap | NOISE |
| 5 | B1-rerun-LB | 2026-10-02 | ARC-AGI-3 | rerun | Franzen fork rerun | NOISE |
| 6 | A2-LB | 2026-10-03 | ARC-AGI-2 | ARC-AGI-2 | pass2 ensemble | NOISE |
| 7 | E3T | 2026-10-03 | ARC-AGI-2 eval 120+60 | ARC-AGI-2 | TPU v5e-8 bf16 vllm-tpu 0.28: phase A low16k 120 tasks (9000 s) + phase B xhigh48k 60 tasks (12600 s | FAIL (infra) |
| 8 | B8-X7 | 2026-10-03 | ARC-AGI-3 | GPU change | X7 playbook summary compaction on X6 (interval 16k < drain 58k; stop drain at summaries; hide follow | FAIL (killed) |
| 9 | B9-X8 | 2026-10-03 | ARC-AGI-3 | GPU change | X8 prediction-mismatch halt as prompt (existing ARC3_STEP_VERIFICATION_HINT=1) on X6 | FAIL gate (neutral) |
| 10 | E2.1 | 2026-10-03 | ARC-AGI-2 eval 60 | ARC-AGI-2 | model-first 27B GPTQ xhigh 48k cap, 2 attempts, forced answer on truncation | FAIL accuracy (T2 mechanics pass) |
| 11 | B10-X9 | 2026-10-03 | ARC-AGI-3 | GPU change | X9 reasoning_effort=low on every agent turn (template default xhigh); compute-bound evidence says mo | FAIL (killed) |
| 12 | R4-FS2-G0 | 2026-10-03 | ARC-AGI-3 CPU | CPU only | G0: own FX explorer clears >= 25% of frontier level starts at 40k actions | FAIL G0 |
| 13 | P1B-DOTS3 | 2026-10-04 | ARC-AGI-2 | ARC-AGI-2 | Dots3-Note IQ2_XXS L4x4 probe with forced 30k thinking budget; ub512 vs ub4; 24 tasks pass@1 (P1 v1  | FAIL (infra: build) |
| 14 | B13-X6D16-res | 2026-10-05 | ARC-AGI-3 serving | GPU change | 16 streams + mamba 80 + HiCache (D38) | FAIL (D38 prefill criterion; no job-level gain) |
| 15 | K1-WASTE | 2026-10-06 | ARC-AGI-3 harness | CPU only | Recovering wasted decode (answer forcing at the token cap, filling idle slots) is worth >= 10% more  | KILLED (CPU) |
| 16 | K2-DRAIN | 2026-10-06 | ARC-AGI-3 harness | CPU only | History drains hurt: keeping more history (drain floor 58K -> 81.5K) raises levels | KILLED/PARKED (CPU) |
| 17 | K5-EFFIC | 2026-10-07 | ARC-AGI-3 harness | CPU only | The action-efficiency term (s_i = min(1.15,(h/a)^2)) costs a large share of our score | KILLED (CPU) |
| 18 | B15C-FB25-res | 2026-10-07 | ARC-AGI-3 serving | GPU change | vLLM mixed NVFP4-FP8 at 12 streams beats B12 at full budget | KILL |
| 19 | K-SURPRISE | 2026-10-09 | ARC-AGI-3 harness | CPU only | An automatic surprise-halt (stop a batch after an action whose interior diff is > max(20, 3x that ac | KILLED (CPU) |
| 20 | HT-FB25 | 2026-10-10 | ARC-AGI-3 harness | GPU change | Fresh-eyes analysts at every SDT fire (2 fresh-context diagnoses through goal / controls lenses appe | PARK by D67 levels (108; 104-111) but HT_VALID FAIL: analysts answered 0 of 28 calls (D70) |
| 21 | SDT-FB25-res | 2026-10-10 | ARC-AGI-3 harness | GPU change | SDT pair readout (D58/D60) | KILL (D64) |
| 22 | HT-FB25-res | 2026-10-11 | ARC-AGI-3 harness | readout | HT readout (D67 gates, n = 1) | PARK (levels) / HT INERT -> not replicated as built (D70) |
| 23 | TC27-FB25-res | 2026-10-11 | ARC-AGI-3 engine | GPU change | TC27 readout (D65) | KILL (D66) |
| 24 | D27-FB25-res | 2026-10-11 | ARC-AGI-3 engine | GPU change | D27 readout (D65) | KILL (D66) |
| 25 | G4-D25 | 2026-10-11 | ARC-AGI-3 engine | GPU change | Gemma-4-31B-it (Google QAT W4A16 + QAT assistant MTP) on the B12 harness vs the B12-D25 pair, 25 gam | KILL (D69) |
| 26 | G4-D25-res | 2026-10-11 | ARC-AGI-3 engine | readout | G4 readout (D68 gates) | KILL (invalid: 12/25 fed < 15; levels 5 < 40) (D69) |
| 27 | HT2-FB25 | 2026-10-11 | ARC-AGI-3 stall | GPU change | Fresh-eyes analysts that answer directly (no thinking) add levels on SDT stalls (D71) | PARK (D74) |
| 28 | X7B-FB25-run | 2026-10-11 | ARC-AGI-3 memory | GPU change | Drain-aligned rolling summaries add levels at no prefill cost (D72; build D50) | KILL (D74) |
| 29 | B28-TC-FB25-res | 2026-10-28 | ARC-AGI-3 model | GPU change | ThinkingCap W4A16 view (D78) | KILL (D79) |
| 30 | B29-OCC-FB25-res | 2026-10-29 | ARC-AGI-3 occupancy | GPU change | fill client-time gaps with 12 active streams, server maxreq stays 10 (D80) | KILL (D84) |
| 31 | B31-SPEC2-FB25-res | 2026-10-29 | ARC-AGI-3 engine | GPU change | fewer draft tokens raise aggregate tok/s at batch 9-10 | INVALID (24/25 played: lf52 never admitted) + mechanism KILL |
| 32 | B31-SPEC1-FB25-res | 2026-10-29 | ARC-AGI-3 engine | GPU change | fewer draft tokens raise aggregate tok/s at batch 9-10 | INVALID (23/25 played: lf52 + su15 never admitted) + mechanism KILL |

Row 30 also played only 24 of 25 games (lf52 never admitted, D84). Like rows 31 and 32 it is left out of run-level comparisons; its played attempts stay in the level-attempt fit (Appendix B, "partial").

Kind (added 7 Nov 2026): GPU change = a distinct change to the ARC-AGI-3 agent or its serving, run on the GPU (17 rows); ARC-AGI-2 (6); CPU only = a simulation or log analysis without a new GPU run (5); rerun = identical code, a noise sample (2); readout = a second row for a change listed above (2).
