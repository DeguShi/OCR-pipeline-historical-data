# Post-correction study: protocol and records

This study asks one question: **how much does our post-correction add, and does it beat standard LM-fused CTC
decoders given the same recognizer and the same text resources?** This file fixes the rules. A claim enters the paper
only if it follows them.

## Rules

1. **Page roles are fixed.**
   - Tuning: the 9 pages {5, 6, 8, 9, 11, 13, 14, 15, 16}. Every hyperparameter of every method is chosen here.
   - Held-out: {7, 10, 12}, scored against the official reference.
   - Test: the 10 pages of `data/extra/teste`, scored against the Gemini reference.

   Held-out and test are measurement only. A method's configuration is frozen before they are decoded, and each frozen
   configuration is measured on them once.

2. **Same inputs.** Every method decodes the same exported recognizer posteriors (`tools/export_posteriors.py`,
   st_c1m, local CPU). They are scored with the same text rules and metric (`tools/score_lines.py`,
   `ocr_pipeline/metrics.py`, normalized policy).
   - The published v6 numbers (`records/final_scores/final_scores_v6.json`) came from a Colab GPU run. They differ on
     3 pages (p12 CER, p22 and p55 WER), so comparisons use the local posteriors, and the paper states this.
3. **Same resources.** The baselines get the period corpus our model uses: unweighted, and weighted (laws ×4,
   book ×8) where the toolkit accepts it. The corpus text comes from `tools/make_lm_corpus.py` and its SHA-256 values
   are recorded below.
4. **Fair tuning.** Each baseline is searched until its tuning WER stops improving. The search space covers every
   parameter the toolkit exposes that matters for decoding, not only the LM weights. Searches are seeded and
   resumable (`tools/baseline_search.py`). Their full output is kept here.
5. **Statistics.**
   - Held-out and test only; the tuning column describes fit, not evidence.
   - Paired comparison on blocks of reference words (`tools/block_bootstrap.py`): block bootstrap CI95 and sign-flip
     p-value at block sizes 10 and 30.
   - Page win counts.
   - Page-level bootstrap over 3 pages is not reported as evidence.
6. **Reproducibility.**
   - Every number in the paper names its producing command and file.
   - Every remote run uses the pinned environment `tools/requirements-decoders.txt`, takes code from a commit
     (`git archive`), and is checked for byte-identical output against a local decode of the same configuration.
7. **Known biases are stated, not hidden.** The test reference is Gemini's. Gemini also labeled the recognizer's
   training lines and the book text in the corpus.

## Environment

- Decoders: Python 3.12.12; `tools/requirements-decoders.txt`, with KenLM @ 4cb443e6 and word-beam-search @ 6ae8c87.
  Locally the environment is `~/.cache/ocr-postcorr/venv-dec`; on Colab it is `/content/venv`, built with `uv`.
  - Colab sets `UV_SYSTEM_PYTHON=true` and a `PYTHONPATH`. Remove both from the environment and run Python with `-I`.
- KenLM models: `lmplz -o 3` on the corpus text, then `build_binary`. `kn3_w4b8` needs `--discount_fallback`
  (duplicated text breaks modified Kneser-Ney discount estimation).

| file                               | SHA-256                                                          |
| ---------------------------------- | ---------------------------------------------------------------- |
| `outputs/lm/period_w1.txt`         | 81cf1f5366ff8354e93b336207fc2bf8f213e5c96bd9d529ef29e93136b7777e |
| `outputs/lm/period_w4_livro78.txt` | 9f87ee1bdc32c4c127d72e4dc9292ba19bca653dc5e3715921a059a96c79a125 |
| `outputs/lm/kn3_w1.bin`            | 7612318f361a387b9df26354008f744fcd2b5b441732d271160138085a913959 |
| `outputs/lm/kn3_w4b8.bin`          | 782260bbcec940dfb5193cc97b76dbd9686ed3f76c51d6d0b329d7b19c8b080b |

## Verification log

- 2026-11-05: the v6 cache reproduces the published 19.40 (all 12 pages).
- 2026-11-05: exported greedy text plus `score_lines` reproduces ladder L0/L1 exactly.
- 2026-11-05: the regenerated corpus matches the manifest token count (24,670,416).
- 2026-11-06: baseline A configuration 0, decoded on Colab CPU, is byte-identical to the local decode
  (285 of 285 tuning lines).
- 2026-11-06: baseline B configuration 0, decoded on Colab CPU (x86-64), matches the local decode (arm64) on 282 of
  285 tuning lines; tuning WER 23.70 vs 23.65 (one word). The three lines differ at near-ties between rare symbols
  (`&`/`⟦`, `،`/`✳`) and one word (`convidar`/`comida2`). Two local runs agree with each other, so the difference is
  across platforms, not between runs. Consequence: B is selected on its Colab search output, and its frozen
  configuration is decoded on held-out and test on the same Colab CPU; a local decode of the same configuration is
  reported alongside as the cross-platform check.
- 2026-11-06: baseline A stage 1 (121 configurations, `records/postcorr/search_A_scored.jsonl`): configuration 0
  rescored from the search output gives tuning WER 21.56, as before. Best is id 112 (20.48). The best improved at
  id 112 of 120, so a local stage A2 (60 configurations around id 112, beam up to 256) runs before A is frozen.
- 2026-11-06: baseline A frozen at A2 id 11 (kn3_w1, alpha 0.773, beta -0.243, unk -10.492, beam 256, prune -16.713,
  token_min -7.338; tuning WER 20.09; the 8 best A2 configurations lie within 0.35 points, so the search has
  plateaued). The local decode of all 12 pages matches the Colab search output on 285 of 285 tuning lines.
  Held-out and test were decoded once, after freezing. v6 vs frozen A (`records/postcorr/blocks/`):
  held-out 18.74 vs 20.00, dWER -1.26, CI95 [-3.01, 0.16], p 0.20 (block 10) / 0.19 (block 30), 3 of 3 pages;
  test 3.49 vs 4.33, dWER -0.84, CI95 [-1.41, -0.26], p 0.007 / 0.019, pages 6 won, 2 tied, 2 lost.
  Against the default-tuned A the test gap was -1.68: half of v6's apparent advantage was tuning asymmetry.

- 2026-11-07: baseline B frozen at B2 id 1 (NGrams, smoothing 0.001, corpus w4b8, beam 256; tuning WER 23.08; beam
  128 gave 23.25, so the beam axis has flattened). The NGramsForecast setting (stage B id 12) is excluded: on 2
  Colab CPUs it ran over 8 h without finishing the 9 tuning pages, and the session was reclaimed. The frozen
  configuration was decoded on all 12 GT pages and on test10 on one Colab CPU; the 9 tuning pages are identical to
  the search output (9 of 9 pages). v6 vs frozen B: held-out 18.74 vs 20.31, dWER -1.57, CI95 [-3.15, 0.00],
  p 0.09 (block 10) / 0.13 (block 30), 3 of 3 pages; test 3.49 vs 5.75, dWER -2.26, CI95 [-3.17, -1.35],
  p 0.0001 / 0.0002, 10 of 10 pages.
- 2026-11-07: `tools/postcorr_from_posteriors.py` (our post-correction run from the posterior export, for the cost
  benchmark) reproduces the pipeline's v6 page 7 exactly once the configuration's lexicon edit step
  (`postprocess.correct`) is included. `score_lines` gives external baselines the text rules only, not that step.
- 2026-11-07: correction analysis (`tools/correction_analysis.py`, reference words vs raw OCR):
  test, fixed / broken: v6 191 / 10, A 168 / 5, B 150 / 13 (of 260 raw errors, 2,001 raw-correct words);
  tuning: v6 229 / 15, A 210 / 14, VLM paper_pt 195 / 136. The VLM breaks about 10% of the raw-correct words; at
  least 52% of those are modernized spellings or accents (`records/postcorr/vlm_broken_words_tuning.json`).
- 2026-11-07: cost benchmark run 1 (one Colab CPU VM: 1 vCPU, Xeon @ 2.20 GHz, 13 GB; one process each, OMP threads
  1, `/usr/bin/time -v`; the 12 GT pages from the posterior export). v6 from posteriors reproduces the pipeline on
  12 of 12 pages (all12 WER 19.40), and A's output is byte-identical to its frozen decode. v6: 1128 s user CPU
  (about 94 s per page), 19 min wall, peak RSS 0.40 GB, models 17 MB (lexicon 2.4 MB + n-grams 14.7 MB).
  A: 34 s user CPU, peak RSS 0.57 GB, model 324 MB (KenLM binary) + 107 MB unigram corpus. B was cut off by a
  status-check bug (v6's own log line contains "ALL DONE"); fixed, and the whole benchmark is rerun on a fresh VM.
- 2026-11-07: profiling v6 (page 7) showed about 90% of its time in the trigram Viterbi's `logp`, recomputing the
  bigram probability and trigram history count for every transition. Memoizing both per line (`c159005`, pure
  functions) leaves the output unchanged: 12 of 12 GT pages byte-identical to the unmemoized run, 10 of 10 test pages
  identical to the pipeline (test WER 3.49). Local decode time per page fell about 5x (Mac M1, single process:
  gt12 181 s, test10 154 s). Cost tables report both versions; run 1 above is the unmemoized one.
- 2026-11-07: cost benchmark run 2 (fresh Colab CPU VM, same setup as run 1; unmemoized v6, then A, then B, one
  process each). Outputs verified: v6 12 of 12 pages identical to run 1; A 12 of 12 identical; B 11 of 12 identical to
  its frozen decode, the 12th (page 12) differs in one character of punctuation (`(` vs `;`), which normalizes to the
  same words, so WER is unchanged. `/usr/bin/time -v`: v6 988.7 s user CPU, 16:38 wall, peak RSS 0.40 GB;
  A 36.5 s, 0:37, 0.57 GB; B 4,845 s, 1:21:34, 2.88 GB (CTC word beam search, NGrams mode, beam 256, over the 144 MB
  period corpus `period_w4_livro78.txt` as its language model). Raw files in `outputs/bench/run2` (sha256 of the tarball 44c6f40b…).
- 2026-11-07: cost benchmark run 3 (fresh Colab CPU VM, same setup; memoized v6 only). Output 12 of 12 pages
  identical to runs 1 and 2. 416.8 s user CPU (about 35 s per page), 7:02 wall, peak RSS 0.41 GB; on this Xeon the
  memo gives 2.4x (989 -> 417 s), less than the 5x measured on the Mac. Raw files in `outputs/bench/run3_memo`
  (sha256 of the tarball 3cb582fa…).

Cost per page (CPU methods: the 12 GT pages, one Colab vCPU, Xeon @ 2.20 GHz, model load included (v6 0.7 s). VLM: the
9 tuning pages on one T4 GPU, `seconds` per line from `generations.jsonl`, minus the first line, which includes the
model load (87 s for paper_pt); peak GPU memory was not recorded):

| method | compute per page | peak memory | model files | kind |
|---|---|---|---|---|
| v6 (memoized) | 35 s CPU | 0.41 GB RAM | 17 MB | lexicon + trigram heuristics |
| v6 (unmemoized) | 82 s CPU | 0.40 GB RAM | 17 MB | same output |
| A pyctcdecode + KenLM | 3 s CPU | 0.57 GB RAM | 431 MB | beam search + 3-gram LM |
| B CTC word beam search | 404 s CPU | 2.88 GB RAM | 144 MB corpus | beam search + n-gram LM |
| E Qwen2.5-VL-7B 4-bit | 33 s T4 GPU (paper_pt; paper_en 34 s) | not recorded | 16.6 GB checkpoint, NF4 at load | 7B-parameter VLM |
- 2026-11-07: baseline E (VLM) frozen at prompt paper_pt, chosen on the tuning pages only (WER 30.93 vs paper_en
  34.03). The third prompt, paper_pt_period, was never run: the free Colab tier refused a T4 on 18 tries
  (02:00-08:17), so it stays untested and is reported as such. The final held-out and test decodes of paper_pt are
  queued for the next free T4.
- 2026-11-07: baseline E (VLM, frozen at paper_pt) decoded on held-out (7, 10, 12) and test10 on one Colab T4 (pro
  account, by the user's decision, after the free tier refused T4s all night). Scored like every external baseline
  (`score_lines`, text rules only; lines rebuilt from `generations.jsonl`, every line present: 99 + 333).
  Held-out: WER 33.23, CER 13.04 (raw greedy 31.5; v6 18.74). v6 vs E: dWER -14.49, CI95 [-25.15, -7.57]
  (block 10) / [-26.81, -6.57] (block 30), p 0.0001, 3 of 3 pages.
  Test: WER 21.10, CER 7.31 (raw greedy 11.54; v6 3.49). v6 vs E: dWER -17.60, CI95 [-19.28, -15.94] / [-19.51,
  -15.78], p 0.0001, 10 of 10 pages.
  Correction analysis, fixed / broken (break rate): held-out v6 78 / 8 (1.7%), A 70 / 8, B 66 / 4, E 63 / 48
  (10.2%); test v6 191 / 10 (0.5%), A 168 / 5, B 150 / 13, E 85 / 268 (13.4%). E rewrites 93% of the non-empty lines
  (unchanged 9 of 99 and 24 of 330); only 1 of the 429 replies was chatty, so the loss is over-correction
  (modernized spelling and accents, plausible wrong words), not formatting. Output reported as produced: no filter
  was added on evaluation pages. GPU time excluding the first line (model load): held-out 109 s, test 374 s, about
  1.1 s per line, consistent with the tuning pages.
  Setup incident, for the record: the first three launches died at 4-bit loading because the code tarball shipped a
  stale `tools/requirements-vlm.txt` pinning bitsandbytes 0.45.5 (no CUDA 13 binary; the committed file pins
  0.50.2). The run used the committed `baseline_vlm.py` and `requirements-vlm.txt` (sha256 prefixes b17dab52…,
  ebc7e6fd…, checked on the VM). Failed logs kept as `job_failed_bnb*.log`.
- 2026-11-07: VLM prompt paper_pt_period run on the tuning pages (free-tier T4, available again): WER 29.97, CER
  10.33 (paper_pt 30.93 / 11.85, paper_en 34.03 / 13.64, raw greedy 31.04 / 8.38). It is the best prompt on the
  tuning pages, so it replaces paper_pt as the frozen VLM, as the protocol states; held-out and test are decoded
  with it, and the paper_pt results above stay as a reported second prompt. First measurement of peak GPU memory
  (`baseline_vlm.py` now records it): 6.87 GB allocated, 7.06 GB reserved, on a T4 with 4-bit NF4 weights. GPU
  seconds (808 s for 285 lines) are inflated by a CPU job sharing the VM (the masked-LM slot dump), so the per-line
  time stays the paper_pt measurement.
- 2026-11-07: recognizer-quality curve. The same v6 configuration (tuned with st_c1m only; nothing retuned) is run on
  the posteriors of two weaker recognizers: `medium` (the published Kraken model, Zenodo 10.5281/zenodo.21788410)
  and `st_v3` (an earlier fine-tune of ours). Configs `configs/curve_<model>.json` differ from final_v6 only in
  `rec.model`; posteriors exported with `tools/export_posteriors.py`, then `tools/run_curve.sh <model>` (greedy and v6
  from the same export, `score_lines --no-rules`). Paired block bootstrap (block 10) v6 vs raw greedy,
  `records/postcorr/curve/`:
  held-out: medium 62.36 -> 28.03 (dWER -34.33, CI95 [-38.81, -29.89]); st_v3 37.32 -> 22.68 (-14.65, [-17.68,
  -11.74]); st_c1m 31.50 -> 18.74 (-12.76, [-16.05, -9.61]).
  test: medium 55.59 -> 25.61 (-29.99, [-32.07, -27.93]); st_v3 27.16 -> 19.24 (-7.92, [-9.17, -6.63]); st_c1m
  11.54 -> 3.49 (-8.05, [-9.21, -6.89]). p 0.0001 and every page won in all six.
  The post-correction never hurts and helps most, in absolute points, on the weakest recognizer (about half of
  medium's errors removed); the relative reduction on test is largest for the best recognizer (70%).
- 2026-11-07: frozen VLM (paper_pt_period) on held-out and test (free-tier T4). Held-out WER 26.30, CER 6.83 (raw
  31.5); test WER 18.05, CER 5.60 (raw 11.54). v6 vs E: held-out dWER -7.56, CI95 [-10.47, -4.69] (block 10) /
  [-10.59, -4.76] (block 30), p 0.0001 / 0.0002, 3 of 3 pages; test -14.55, [-16.01, -13.12] / [-16.12, -12.98],
  p 0.0001, 10 of 10 pages. Fixed / broken: held-out 65 / 40 (break rate 8.5%), test 96 / 223 (11.1%); of the
  test breaks, 156 are real-word substitutions (`records/postcorr/error_types_test.json`). The period sentence
  lowers the damage (paper_pt broke 268 on test) but the VLM stays worse than raw OCR on test. On held-out it is
  better than raw; part of that is the official GT itself modernizing, which matches some of the VLM's changes
  (the test reference keeps the period spelling). Peak GPU 6.87 GB allocated; the test run used the GPU alone:
  474.8 s for 333 lines (1.43 s per line incl. the 4-bit load). A first test attempt died with CUDA out of
  memory because the masked-LM scoring (baseline D) shared the GPU; its log is kept as job_oom.log, and the rerun
  had the GPU alone.
- 2026-11-07: baseline D, masked-LM rescoring of our own candidates (`tools/postcorr_mlm.py`) with PortOldBERT
  (LIACC/PortOldBERT @ 9641bba8, EACL 2026, historical-Portuguese DeBERTa, 100M parameters). It adds lambda * PLL to
  each candidate's slot score inside v6's Viterbi (top 10 candidates per slot; context = v6's choice for the other
  words plus the previous and next lines). Checks: the slot dump reproduces v6 (12 of 12 pages) and lambda = 0
  reproduces v6 exactly (12 of 12). lambda on the 9 tuning pages only (corpus WER, `records/postcorr/mlm/`):
  0 19.64, 0.1 19.53, 0.25 19.47, 0.5 19.36, 1 19.64, 2 21.05; frozen at 0.5.
  Held-out 18.58 vs v6 18.74 (dWER -0.16, CI95 [-0.48, 0.00], p 1.0; pages won/tied/lost 1/2/0); test 3.27 vs 3.49
  (-0.22, [-0.49, 0.04], p 0.18 / 0.24 for blocks 10 / 30; 4/6/0). A small, consistent, non-significant gain: the
  masked LM never makes a page worse, and v6's trigram already captures most of what it adds. Cost: about 44 s
  of T4 per page for the scoring (530 s for the 12 GT pages, 21,016 candidates), on top of v6, plus a 100M model.
- 2026-11-07: baseline C, ByT5-small corrector (`tools/baseline_byt5.py`, google/byt5-small) trained on synthetic
  recognizer noise, no evaluation page involved. The noise model holds per-character substitution, deletion and
  insertion statistics of st_c1m's raw greedy output against the reference on the 9 tuning pages (NFC, CER 9.8%,
  `records/postcorr/byt5/noise_tuning.json`). It is applied to clean lines from the 78 training-page transcriptions
  (pages 17-104, no overlap with tuning, held-out or test; x10, fresh noise each time) and from ALESP laws
  1889-1910 cut to 45-character lines. 200k pairs were generated (sha256 86090751…); the first 80k were used (78k
  train, 2k synthetic validation), 1 epoch, batch 16 x 2 accumulation, lr 5e-4, max 128 bytes, seed 42, fp32, free
  T4: 3,257 s, best validation loss 0.077 (`records/postcorr/byt5/`). No hyperparameter was tuned (the first run,
  batch 32, ran out of GPU memory at step 81; its log is kept). Prediction: beam 4, greedy text of each line of
  the posterior export, about 2 s of T4 per page. Scored like A, B and E (text rules only).
  Tuning 22.91, held-out 21.26, test 6.86 (raw 31.04 / 31.5 / 11.54). v6 vs C: held-out dWER -2.52, CI95 [-4.54,
  -0.63] (block 10) / [-4.86, -0.33] (block 30), p 0.02 / 0.06, 3 of 3 pages; test -3.36, [-4.41, -2.34] /
  [-4.55, -2.18], p 0.0001, 10 of 10 pages. Fixed / broken: held-out 63 / 7, test 132 / 26. It is the only
  method that fixes numerals (3 of 9 on test) and most split/merge errors (3 of 4), which v6 cannot do by
  construction (one candidate word per recognized word); v6 fixes many more non-word errors (161 vs 107 of 177).
- 2026-11-07: reproduction check. `tools/check_v6_reproduction.py` (and `OCR_SLOW=1 pytest tests/`) decodes v6
  from the committed st_c1m posteriors (`records/postcorr/posteriors/`, 3.6 MB: held-out 7, 10, 12 and the 10
  test pages, exported by `tools/export_posteriors.py`) and compares a sha256 per page with
  `records/postcorr/repro/v6_expected.json`. No kraken, images or GPU needed; 3 CPU min. Verified: the decoded
  lines score to exactly 18.74 (held-out) and 3.49 (test) WER, identical per page to the committed runs
  (bootstrap dWER 0.00), and a corrupted expected hash makes the check exit 1.
- 2026-11-07: pinned the C/D GPU environment (`tools/requirements-postcorr-gpu.txt`: transformers 4.51.3,
  accelerate 1.6.0; both ran in the session that had the VLM pins installed) and D's tokenizer revision
  (d15e086, unchanged since 2025-06-23). Colab's default transformers (5.22.1 on 2026-11-07) breaks the ByT5
  trainer (`warmup_ratio` removed), caught by a smoke test before relaunching C.
- 2026-11-07: v6 + C hybrid (`tools/hybrid_v6_byt5.py`), a negative result. v6's text is kept, and ByT5's
  version of a differing word block is adopted only for the error kinds v6 cannot fix (numerals, split/merge).
  Four fixed rules were compared on the 9 tuning pages (v6 19.64): split_strict 19.53 (1 block), split_loose 19.92,
  numeral 19.47 (22 blocks; pages won/lost 3/2, p 0.74), numeral_split 19.86. numeral was frozen and run once on
  held-out: 19.06 vs 18.74 (dWER +0.31 worse, CI95 [0, 0.97], 0 pages better / 1 worse). On test: 3.27 vs 3.49
  (dWER -0.22, CI95 [-0.53, 0], p 0.25, 3 pages better / 0 worse). ByT5's numeral edits are wrong about as
  often as they are right, so there is no gain to deploy. Uses the C predictions recorded above (first model).
  Records are in `records/postcorr/hybrid/` and `blocks/blk_*_v6_vs_hybrid_numeral_*`.
