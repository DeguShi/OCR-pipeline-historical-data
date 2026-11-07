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
