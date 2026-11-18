# E20 — full-system comparison

Question: with the same training data and the same tuning budget, does our system (R1·P6: the st_c1m recognizer plus
the v6 post-correction) beat the best recipe from the literature, and where does the difference come from
(recognizer or decoder)? A VLM is compared both as a corrector of a recognizer (R·PE) and as a recognizer (V0).
Plan and dated log: vault note `integracao/E20_sistemas_completos_plan.md` (not in git).

## Systems

| id | recognizer | adaptation |
|---|---|---|
| R0 | PP-OCRv6 medium (Zenodo 10.5281/zenodo.21788410) | none |
| R1 | st_c1m (ours) | medium fine-tuned on 2,425 Gemini-labelled lines of 78 training pages (E16) |
| R2 | TraPrInq (published) | none |
| R3 | McCATMuS (published) | none |
| R4 | TraPrInq | fine-tuned with the st_c1m lines and recipe (`train/selection.json`) |
| R5 | McCATMuS | idem |

| id | decoder |
|---|---|
| P0 | CTC greedy (+ text rules) |
| PA | pyctcdecode + KenLM, period corpus (E17 baseline A) |
| PB | CTC Word Beam Search, E17 configuration (NGrams, smoothing 0.001, corpus w4b8, beam 256) |
| P6 | our post-correction (v6), with the E16 sweep per recognizer |
| PE | VLM correction (Qwen2.5-VL-7B NF4, prompt `paper_pt_period`, frozen 2026-11-07) of the recognizer's greedy text |
| V0 | VLM transcription from the line crop alone (prompt `tr_pt`, frozen 2026-11-13, `v0/`) |

All recognizers read the same pre-processed image and the same blla segmentation of v6 (rule 3); the "as published"
row (`p0raw`: raw image, Kraken's default segmentation and line padding 16, `configs/e20/raw_*.json`) is context only.

## Rules (fixed before running)

1. Same training lines for R1/R4/R5 (2,425 lines, no page 5–16 nor test page); learning rate of R4/R5 chosen from
   {1e-3 (Kraken default), 1e-4, 3e-4} by validation accuracy on 243 of those lines (`train/selection.json`).
2. Selection on the 9 tuning pages only (5, 6, 8, 9, 11, 13, 14, 15, 16), same budget per recognizer: PA 61 random
   + 31 local configurations (baseline_search A/A2, seed 0); PB fixed; P6 the 16 E16 variants, accepted if they lower
   the Gemini-reference WER by ≥ 0.3 without raising the official WER, then combined greedily (keep ≥ 0.15).
3. Character map (`tools/charmap.py`): characters outside the st_c1m training alphabet become their NFD components
   inside it, or are deleted; applied to every system's output before scoring (for P6 after the post-correction, a
   deviation from the plan, which said before).
4. Frozen before any evaluation page was decoded: `select/frozen.json`, commit 9ee0af3.
5. Held-out (7, 10, 12) against the official GT and the Gemini reference; test10 against its Gemini reference;
   normalized WER/CER; block bootstrap (10 reference words, 10k resamples, seed 42) of R1·P6 against each system.

## Selection on the tuning pages (official GT, WER %)

| recognizer | P0 | PA | PB | P6 | P6 variants accepted |
|---|---|---|---|---|---|
| R0 medium | 58.18 | 40.01 | 45.43 | 28.84 | minlen3 + fold |
| R1 st_c1m | 31.04 | 20.54 | 23.08 | 19.64 | none (v6 as is) |
| R2 TraPrInq | 75.00 | 63.15 | 73.53 | 58.47 | beta15 |
| R3 McCATMuS | 91.70 | 83.35 | 91.70 | 77.26 | beta15 + fold |
| R4 TraPrInq-ft | 27.82 | 20.71 | 24.60 | 20.71 | none |
| R5 McCATMuS-ft | 34.09 | 22.86 | 28.16 | 22.35 | none |

Files: `select/<rec>/` (A/A2 scored configurations, PB result, P6 sweep runs and final config), `select/vm_*`
(environment, machine, job summary). Compute: PA of R0–R2 and PB of R1/R2 on free Colab CPU VMs (2 vCPU, commit
033f083, 2026-11-13); the rest on three Colab pro high-RAM CPU VMs (8 vCPU, commit 1119c08, 2026-11-17: PA in 4
shards, P6 runs split over pages; the split decode was checked byte-identical to the single-process one on R1 base),
25–35 min per recognizer, ~1.2 compute units in all. The free-VM job lost the work done after its watcher stopped
(2026-11-14 ~02:00); everything not pulled by then was recomputed.

## Evaluation (once)

Posteriors of held-out and test10 exported on a Colab pro T4 (commit 82607a6; the kraken device must be given as
`cuda:0`, the first launch with `cuda` failed before decoding anything, log kept as `job_failed_cuda.log`). R1's
export reproduces the committed v6 page hashes (`tools/check_v6_reproduction.py`: REPRO v6 ok).

Results: `eval/grid.md`, `eval/paired.json` (tools/e20_report.py). V0 and R1·PE were decoded before E20 and are
rescored here with the character map: R1·PE held-out official 26.30 → 24.72, test 18.05 → 15.97 (the VLM writes í, â
and other characters outside the convention); V0 test 54.40 → 53.78.

R1·P6 here is v6 decoded from the posteriors (held-out 18.74 / CER 4.35, test 3.49), as in every comparison of
`records/postcorr`; the published v6 numbers (CER 4.43, test 3.41) are the older GPU run that differs on 3 pages
(`records/postcorr/README.md`, point 2).

### Main systems (WER / CER %, normalized; full grid in `eval/grid.md`)

| system | held-out official | held-out Gemini | test10 Gemini |
|---|---|---|---|
| R1·P6 (ours) | **18.74** / 4.35 | 6.33 / 2.26 | **3.49** / 1.50 |
| R4·P6 (TraPrInq-ft + our post-correction) | **18.74** / 4.11 | **5.42** / 1.92 | 4.60 / 1.78 |
| R4·PA (best literature recipe on tuning) | 19.21 / 4.32 | 6.63 / 2.31 | 5.84 / 1.99 |
| R5·PA (McCATMuS-ft + pyctcdecode) | 19.37 / 5.12 | 7.98 / 3.34 | 6.02 / 2.21 |
| R1·PA | 19.84 / 4.88 | 10.39 / 3.26 | 5.44 / 1.93 |
| R4·PE (VLM corrects R4) | 24.72 / 5.97 | 14.01 / 4.23 | 17.91 / 6.64 |
| R1·PE (VLM corrects R1) | 24.72 / 6.54 | 16.57 / 5.18 | 15.97 / 5.19 |
| R0·P6 (medium, not adapted) | 26.61 / 10.32 | 27.11 / 10.02 | 25.87 / 8.88 |
| R2·P6 / R3·P6 (published, not adapted) | 56.22 / 76.22 | 54.67 / 76.05 | 51.92 / 69.39 |
| V0 (VLM reads the crop) | 49.29 / 21.82 | 50.30 | 53.78 / 23.58 |

Greedy (P0) for reference: R1 29.45 / 18.22 / 11.50; R4 23.78 / 12.20 / 9.29.

"As published" context row (`p0raw`: raw image, Kraken default segmentation, padding 16, greedy; commit 6a2872d on a
Colab pro high-RAM CPU VM, `eval/vm_e20raw/`): R0 60.79 / 63.10 / 57.28, R2 79.84 / 78.31 / 73.73, R3 91.65 / 89.91 /
86.82 WER. Against greedy on our pre-processing and segmentation (R0 58.58 / 60.84 / 55.20, R2 74.49 / 73.19 / 69.44,
R3 90.55 / 89.76 / 86.47) our image chain alone gains 0.15–5.35 WER points (least on McCATMuS, whose held-out CER is lower on the raw image: 44.15 vs 46.01); the rest of the distance to R1·P6 is adaptation and
decoding. The first launch of this job never started (the shipped archive had no `outputs/` for its log); relaunched
after 1 h 20 min idle, then setup 803 s and eval 1,054 s.

### Paired comparisons (`eval/paired.json`; dWER = R1·P6 minus the other, CI95, pages won/tied/lost)

| R1·P6 vs | held-out official | held-out Gemini | test10 Gemini |
|---|---|---|---|
| R4·PA | −0.47 [−2.07, 1.13] 1/0/2 | −0.30 [−2.26, 1.65] 2/0/1 | **−2.34 [−3.42, −1.28]** 8/0/2 |
| R4·P6 | 0.00 [−1.40, 1.40] 1/0/2 | +0.90 [−0.60, 2.41] 0/1/2 | **−1.11 [−1.99, −0.22]** 7/1/2 |
| R5·PA | −0.63 [−2.67, 1.41] | −1.66 [−3.81, 0.45] | **−2.52 [−3.53, −1.51]** 10/0/0 |
| R1·PA | −1.10 [−2.98, 0.64] | **−4.07** | **−1.95** 9/0/1 |
| R4·PE, R1·PE, V0, R0·P6 | all CIs exclude 0 in favour of R1·P6 on every set | | |

Decomposition:
- decoder, same recognizer: R4·P6 − R4·PA = −0.47 [−1.28, 0.32] / −1.20 [−2.41, −0.15] / **−1.24 [−1.92, −0.58]**;
- recognizer, greedy: R1·P0 − R4·P0 = **+5.67** / **+6.02** / **+2.21**, so the fine-tuned TraPrInq is the better
  recognizer before decoding;
- recognizer, both under PA: R1·PA − R4·PA = +0.63 / **+3.77** / −0.40.

Reading by the rule fixed in the plan ("beats" only if the test CI excludes 0 and held-out is not unfavourable):
R1·P6 beats R4·PA (test −2.34, held-out −0.47 / −0.30, CIs across 0). Against R4·P6, the same post-correction on the
better recognizer, the test favours R1·P6 (−1.11) but held-out Gemini leans the other way (+0.90, CI across 0): a tie
on held-out, an advantage on test only. The gain over the literature recipe comes from the decoder: P6 beats PA on
both recognizers, and R4 is the better recognizer at greedy. Caveat: v6 was developed on R1 over E01–E16; R4 got the
same 16-variant sweep but not the design history, which favours R1·P6.

### Error types (`eval/error_types_*.json`; raw = greedy errors; fixed / broken per category)

test10 Gemini, R1 (raw errors: non_word 178, real_word 32, short 18, deleted 19, numeral 9, split_merge 4):

| decoder | non_word | real_word | short | deleted | split_merge |
|---|---|---|---|---|---|
| PA | 136 / 4 | 18 / 14 | 9 / 6 | 2 / 2 | 2 / 2 |
| PB | 131 / 11 | 9 / 1 | 5 / 0 | 3 / 0 | 2 / 1 |
| P6 | **162 / 0** | 17 / 9 | 11 / 1 | 1 / 0 | 0 / 0 |
| PE | 82 / 31 | 8 / **112** | 8 / 32 | 3 / 3 | 0 / 2 |

R4 shows the same pattern (P6 non_word 93/0 of 117; PE breaks 144 real words). P6's edge is non-words fixed
without breaking any; the VLM rewrites valid words (modernized spelling), which is where PE loses.

### Cost

Evaluation VM (Colab pro T4 high-RAM, ~1.3 units/h): setup 208 s (first launch), export 933 s, R4·PE 1,262 s (in
parallel with the decodes), 48 CPU decodes 5,597 s on 7 workers; the VM was up 2 h 19 min, ~3 units. Bootstrap and
report on the Mac, 2 s.

### Page-level robustness check (added after the evaluation, 2026-11-17)

The word-block bootstrap ignores the correlation of errors within a page, so the main pairs were re-tested with the
page as the unit (`tools/e20_page_test.py` → `eval/paired_pages.json`): exact paired sign-flip permutation over the
10 test pages, and a page bootstrap (10k, seed 42). Test10, Gemini reference:

| pair | dWER | page-bootstrap CI95 | sign-flip p | W/T/L |
|---|---|---|---|---|
| R1·P6 − R4·PA | −2.34 | [−3.64, −1.16] | 0.008 | 8/0/2 |
| R1·P6 − R4·P6 | −1.11 | [−2.02, −0.18] | **0.066** | 7/1/2 |
| R1·P6 − R1·PA | −1.95 | [−2.68, −1.20] | 0.004 | 9/0/1 |
| R4·P6 − R4·PA | −1.24 | [−2.03, −0.50] | 0.014 | 9/0/1 |
| R1·P0 − R4·P0 | +2.21 | [0.94, 3.73] | 0.016 | 0/3/7 |

The verdict against R4·PA and the decoder effect hold with pages as the unit; R1·P6's test advantage over R4·P6 does
not reach p < 0.05 by permutation, so it is reported as fragile. With 3 held-out pages the sign-flip p cannot go below
0.25. The "2,34 vs R4·PA" splits into 1.24 from the decoder (R4 fixed) and 1.11 from R1·P6 vs R4·P6.

### Raw row vs. the paper's baseline (added 2026-11-17)

The paper's starting point (held-out official 64.57 / 21.71, `README.md` progression table) has no raw log in
`records/`. The E20 raw row measures the same condition (published medium, raw image, default segmentation,
greedy) at 60.79 / 21.15, but it is scored with the text rules. Rescored with `--no-rules`
(`eval/score/heldout/r0_medium/p0raw_norules/oficial`): 62.99 / 21.42. The rules explain 2.2 of the 3.8 points;
the remaining 1.6 (Kraken version, padding, segmentation defaults at the time) is not isolated.
