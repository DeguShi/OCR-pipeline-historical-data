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
