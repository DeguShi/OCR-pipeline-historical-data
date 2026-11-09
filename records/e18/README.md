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
| low_res_blur      | 28.98   | 27.72 (12%)          | **23.31 (55%)**      | 25.35 (35%)   | 28.03  |

Spearman(objective, greedy CER) over the 47–60 candidates on the sample page: lex 0.94–0.97 on every shift,
conflex 0.94–0.98, conf 0.69–0.97, stat 0.56–0.76.

Against the plan's criteria: Q2 (Spearman ≥ 0.6 on most shifts) is met by lex, conflex and conf, and only partly by
stat. Q1 (on shifts that cost ≥ 2 points — uneven_light, jpeg_noise, low_res_blur — ≥ 50% recovery on average,
never > 1 point worse): **lex meets it** (89%, 28%, 55%; mean 57%); conf does not (83%, 30%, 12%; mean 42%).
Neither is ever worse than the default on any shift. stat fails
(worse on 3 of 6 shifts: it often prefers upscale 1, which segments worse). The one-page oracle is not better than lex:
the best chain on one page transfers imperfectly. On jpeg_noise most of the damage stays (70 vs 18.7):
preprocessing cannot undo heavy compression noise for this recognizer.

Reading: the user's idea works when "closer to our current best" means "where our recognizer reads the most
period-lexicon words". It does not work when it means matching image statistics. Single-page selection by `lex`
(or `conflex`) is the candidate for a deployable `adapt` step.

## Controls (2026-11-08; `controls/`, `tools/run_e18_control.sh`, code `41a2bb0`, about 2.6 h free CPU)

**Identity** (the whole search on our own book, JPEG re-saved at q95): lex (= conflex) keeps the default chain,
18.74 (no change). conf picks illumination+contrast for segmentation and clahe for recognition, giving 20.16
(+1.42, worse than the 1-point limit). stat gives 19.21. The one-page oracle gives 19.69: it overfits page 6.

**Static chain** `fixed_clahe` (seg_ops = clahe, remove_rules, the chain lex and the oracle chose most often),
same everywhere:

| shift             | default | fixed_clahe | lex       |
| ----------------- | ------- | ----------- | --------- |
| clean             | 18.74   | 18.43       | 18.74     |
| uneven_light      | 21.57   | 20.00 (55%) | **19.06** |
| low_contrast_cast | 19.84   | 20.16       | **19.53** |
| faded_ink         | 20.16   | 19.37       | **19.06** |
| bleed_through     | 19.21   | 19.53       | **19.21** |
| jpeg_noise        | 90.08   | 89.45 (1%)  | **70.08** |
| low_res_blur      | 28.98   | 26.30 (26%) | **23.31** |

lex is at least as good as the static chain on every shift and better on all six. On the three shifts that cost
2 points or more, the static chain recovers 28% of the damage on average and lex 57%. So most of lex's gain comes
from choosing a chain per book, not from CLAHE being better in general. On clean pages the static chain is 0.31
better (3 pages, within noise), so it is not adopted as the new default.

Incident: the first control run failed after 108 min on the per-shift loop. macOS bsdtar had packed `._*` xattr
files into the state tarball (`tar t` on the Mac does not list them; GNU tar on the VM extracts them), and
`discover()` parsed `._16…` as a page number. The fix was to delete them on the VM and resume only the missing
runs (`control_part1.log` + `control.log`). From now on, tarballs are built with `COPYFILE_DISABLE=1`.

## Real books (Q3, 2026-11-09; `books/`, `tools/traprinq_prepare.py`, `tools/run_e18_books.sh`, code `bad826a`)

Data: TraPrInq validation set (Zenodo 13986218, CC BY 4.0; Portuguese Inquisition case files, 16th–19th c.,
Transkribus PAGE XML with diplomatic line text). 4 case files with 4–8 pages, drawn with seed 42. Each case file is
treated as a "new book": one sample page (seeded) for the search with lex, conf and oracle, then full v6 on the
other 18 pages. The recognizer was never trained on these hands. Compute: free Colab CPU VM, 4.5 h. A first attempt
on a free T4 was reclaimed after 2.5 h with nothing pulled; the watcher now pulls every 30 min.

Eval-page WER / CER (%), normalized:

| case file (eval pages)  | default      | lex              | conf         | oracle       | lex chain (seg / rec / upscale) |
| ----------------------- | ------------ | ---------------- | ------------ | ------------ | ------------------------------- |
| IL-028-CX1622-16125 (3) | 62.87 / 25.9 | **62.00 / 26.1** | 63.05 / 26.7 | 71.45 / 32.7 | none / gamma 0.7 / 2            |
| IL-28-12955 (5)         | 81.84 / 46.6 | **81.84 / 43.3** | = lex        | = lex        | contrast+rules / gamma 0.7 / 1  |
| IL-28-7123 (7)          | 73.04 / 38.2 | **67.76 / 32.9** | 69.76 / 32.2 | 68.34 / 31.5 | clahe+rules / channel min / 2   |
| IL-28-7196 (3)          | 74.66 / 36.5 | **72.48 / 34.2** | 73.99 / 33.5 | = conf       | clahe+rules / channel min / 2   |

Against the criterion (lex ≤ default on ≥ 3 of 4 books, never > 1 point worse): **met**. lex is at least as good
as the default on all 4 books in WER (−0.87, 0.00, −5.28, −2.18) and reduces CER on 3 of 4 (+0.1 on the first).
conf also helps on 3 of 4 but less in WER. The one-page oracle is again not better than lex and is much worse on the
first book (71.45), because a chain tuned to one page's GT overfits that page.

Caveats. Absolute error is high (62–82% WER): these are other hands and centuries, the recognizer has never
seen them, and the references are diplomatic (abbreviations kept, no expansion), unlike our v6 output. The
comparison is within-book (same pages, same metric), so the reference style affects every config the same way.
There are 4 books and 18 pages, so this is directional evidence, not a CI-backed claim.

Decision: an `adapt` step (one sample page → lex search → per-book config) is worth adding to the deployed
pipeline as an opt-in mode. It never hurt in WER on any synthetic or real book with lex (identity included), and it
gains 1–5 points where the scan differs from ours.

## Pending

- (done) Real books (Q3): TraPrInq validation set (Zenodo 13986218, CC BY 4.0), case files with ≥ 4 pages.
- Figure panels done: `e18_figures/` (sample page 6, a band of lines: shifted input, default seg image, chosen
  seg and rec images of the lex objective).

## Segmentation check (2026-11-09; `seg/`, `tools/seg_eval.py`, code `3c80fba`)

Question: do the real-book gains come from better line finding? For each TraPrInq case file, every one of the 8
segmentation candidates was run on the 18 eval pages and its baselines matched to the PAGE XML baselines (after the
same upscale + deskew; match = x-coverage ≥ 0.5 and mean |dy| ≤ 0.5 × median GT line spacing, one-to-one greedy).
Free Colab CPU, 1.7 h. P = matched / predicted lines, R = matched / GT lines (summed over the eval pages):

| case file (GT lines) | default (up2 contrast) P / R / F1 | lex chain P / R / F1     | F1 range over the 8 |
| -------------------- | --------------------------------- | ------------------------ | ------------------- |
| 16125 (61)           | 73 / 100 / 84                     | none: 74 / 100 / 85      | 80–85               |
| 12955 (168)          | 84 / 99 / 91                      | up1: 84 / 99 / 91        | 86–92               |
| 7123 (199)           | 70 / 96 / 81                      | clahe: 59 / 93 / 72      | 72–84               |
| 7196 (80)            | 68 / 98 / 80                      | clahe: 71 / 99 / 82      | 75–90               |

Recall is 93–100% for every candidate on every book: the segmenter finds nearly all GT lines whatever the image
chain. Candidates differ in precision (59–86%), i.e. in extra lines; the overlays show these are mostly marginal notes with no
baseline in the PAGE XML, plus main-text lines broken into two pieces (one piece matches, the other counts as extra). lex's segmentation choice is within ±2 F1 of the
default on three books and 9 F1 worse on 7123, which is the book where lex gains most in WER (−5.28). So the real-book
gains do not come from line finding; they come from the image given to the recognizer (gamma 0.7 / channel min),
and the lex score of a segmentation candidate reflects how well its lines are read, not how well they are found.
Overlays for the default and lex chains (first eval page) are in `seg/overlays/`.
