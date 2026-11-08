# E18 — adaptive preprocessing for new books (records)

Question: for a book that looks different from ours, can the preprocessing chain (segmentation image, recognition
image, upscale) be chosen from ONE sample page without labels, and which label-free objective should choose it?

## Setup (2026-11-08)

- Synthetic "other scans" of our 12 GT pages (`tools/synth_shift.py`, deterministic): uneven_light,
  low_contrast_cast, faded_ink, bleed_through, jpeg_noise, low_res_blur (70%, blur 0.6). Page split unchanged.
- One sample page (tuning page 6, `random.Random(42)`); the search (`tools/adapt_preprocess.py`) never sees
  held-out 7, 10, 12. Coordinate-wise: 8 segmentation candidates (upscale 1/2/3 × 6 seg_ops chains), then 13
  recognition chains on each objective's stage-1 winner; greedy decode for the search.
- Objectives: `conf` (mean top-1 CTC posterior on non-blank frames), `lex` (number of greedy words in the period
  lexicon, top 100k), `conflex` (conf × lex), `stat` (distance of image statistics to 8 training pages of our book:
  "bring it closer to our current best"), and `oracle` (greedy CER on the sample page, using its GT: the bound of
  one-page selection).
- Each chosen chain runs the full v6 pipeline once on held-out 7, 10, 12 (`ocr_pipeline.run`); clean default 18.74
  reproduced on the VM.
- Compute: one free Colab CPU VM (2 vCPU, 12 GB; free T4 refused, one session per free account), code `afd3083`,
  driver `tools/run_e18.sh`; about 1.8 h per shift. Raw outputs (no caches) in `outputs/e18_colab/`.

## Results (6 shifts; `e18_report/e18_report.md`)

Held-out WER (%), clean default 18.74; in brackets the share of the shift's damage recovered:

| shift             | default | conf                 | lex (= conflex)      | stat          | oracle |
| ----------------- | ------- | -------------------- | -------------------- | ------------- | ------ |
| uneven_light      | 21.57   | 19.21 (83%)          | **19.06 (89%)**      | 24.41 (worse) | 19.06  |
| low_contrast_cast | 19.84   | **18.74 (100%)**     | 19.53 (29%)          | 21.57 (worse) | 20.16  |
| faded_ink         | 20.16   | 19.69 (33%)          | **19.06 (78%)**      | 20.00         | 19.37  |
| bleed_through     | 19.21   | 19.21 (kept default) | 19.21 (kept default) | 20.47 (worse) | 19.53  |
| jpeg_noise        | 90.08   | **68.66 (30%)**      | 70.08 (28%)          | 90.08         | 77.95  |
| low_res_blur      | 28.98   | 27.72 (12%)          | **23.31 (56%)**      | 25.35 (35%)   | 28.03  |

Spearman(objective, greedy CER) over the 47–60 candidates on the sample page: lex 0.94–0.97 on every shift,
conflex 0.94–0.98, conf 0.69–0.97, stat 0.56–0.76.

Against the plan's criteria: Q2 (Spearman ≥ 0.6 on most shifts) is met by lex, conflex and conf, and only partly by
stat. Q1 (on shifts that cost ≥ 2 points — uneven_light, jpeg_noise, low_res_blur — ≥ 50% recovery on average,
never > 1 point worse): **lex meets it** (89%, 28%, 56%; mean 58%); conf does not (83%, 30%, 12%; mean 42%).
Neither is ever worse than the default on any shift. stat fails
(worse on 3 of 6 shifts: it often prefers upscale 1, which segments worse). The one-page oracle is not better than lex:
the best chain on one page transfers imperfectly. On jpeg_noise most of the damage stays (70 vs 18.7):
preprocessing cannot undo heavy compression noise for this recognizer.

Reading: the user's idea works when "closer to our current best" means "where our recognizer reads the most
period-lexicon words". It does not work when it means matching image statistics. Single-page selection by `lex`
(or `conflex`) is the candidate for a deployable `adapt` step.

## Pending

- Controls `tools/run_e18_control.sh` (identity = adaptation on our own book; a fixed
  clahe segmentation chain on every shift, to separate per-book adaptation from a better static config).
- Real books (Q3): TraPrInq validation set (Zenodo 13986218, CC BY 4.0), case files with ≥ 4 pages.
- Figure panels done: `e18_figures/` (sample page 6, a band of lines: shifted input, default seg image, chosen
  seg and rec images of the lex objective).
