# E19 — a new book of the same archive, no training

## 2026-11-11: does the v6 pipeline transfer to the Livro de Atas da Câmara, 1894–1902?

**Question.** Can the v6 pipeline (st_c1m recognizer + v6 post-correction, `configs/final_v6.json`) read an unseen
book with **no training**? Only the preprocessing chain may adapt, chosen from 3 sample pages. The book is the
Livro de Atas da Câmara Municipal de São Carlos 1894–1902: Pró-Memória photos, a different clerk ("Souza Franco"),
council minutes and not laws. Two further questions:

- does the automatic page crop (`ocr_pipeline/auto_crop.py`, `page_crop.py --box auto`) cut any text?
- does it change book-1 results compared with the hand crops?

**Data.**

- 50 photos (folios 36–65, 1895–1896) with Gemini transcripts supplied by the user (`vault/integracao/E19_livro2/`).
- Line-end `-` hyphens are joined in the references (443 of 453).
- Split (`vault/integracao/E19_split.json`, drawn before any OCR):
  - sample pages 41, 8, 2 (photos 118, 82, 76);
  - 20 eval pages 3 6 7 9 14 15 16 18 21 23 28 33 35 37 38 43 45 47 48 50;
  - 27 pages untouched.
- No page was used both to choose and to score.

Caveats on the references:

- They are Gemini transcripts, not a human GT.
- st_c1m was trained on Gemini labels of book 1, so absolute numbers are optimistic.
- The pages are consecutive folios of one clerk.

**Setup.**

- Free Colab CPU, 2 VMs (2 vCPU, 12 GB).
- Code `exp/e19-new-book`: 32eaf79 for run 1; 988f491 for run 2. Tool SHAs were checked on the VM before each job.
- Run 1 lost both VMs mid-job (2026-11-10 ~17:36 BRT; client offline). Kept from it: the crop part, book-1 runs and
  1 of 8 segmentation candidates.
- Run 2 was relaunched with the search objectives restricted to `lex` (the label-free method) and `oracle`
  (greedy CER on the 3 samples, an upper bound on what sample labels buy). `conf`, `conflex` and `stat` were **not run**.
- Outputs:
  - `outputs/e19_colab_e19{a,b}_run1/` and `outputs/e19_colab_e19{a,b}/` (git-ignored);
  - merged view `outputs/e19_merged/`;
  - digest `records/e19/e19_report.json` (`tools/e19_report.py`);
  - paired block bootstraps `records/e19/bootstrap_*.json` (`tools/block_bootstrap.py`, block 10, 10k resamples, seed 42).

**Chain search** (3 sample pages, greedy decoding, no post-correction):

| objective                            | chosen chain                                     | default chain (v6)                               |
| ------------------------------------ | ------------------------------------------------ | ------------------------------------------------ |
| lex (label-free)                     | upscale 2, seg `clahe+remove_rules`, rec none    | upscale 2, seg `contrast+remove_rules`, rec none |
| oracle (uses the sample transcripts) | upscale 3, seg `contrast+remove_rules`, rec none |                                                  |

The lex score orders the chains much like greedy CER on the samples: the worst lex chains are also the worst CER
chains. Among the top chains, however, the lex gap is small (335 for the chosen chain vs 323 for the default).

**Full v6 on the 20 eval pages** (Gemini references, normalized WER/CER, corpus rates):

| config             | WER       | CER  | strict WER | vs default, paired block bootstrap (dWER, CI95, p) | pages won/tied/lost vs default |
| ------------------ | --------- | ---- | ---------- | -------------------------------------------------- | ------------------------------ |
| default (final_v6) | 22.50     | 7.26 | 32.1       | —                                                  | —                              |
| lex                | **21.14** | 7.02 | 30.8       | −1.36 points, [−2.38, −0.36], p = 0.011            | 12 / 4 / 4                     |
| oracle             | 21.39     | 6.98 | 31.3       | −1.11 points, [−2.18, −0.05], p = 0.051            | 10 / 3 / 7                     |

- Pre-registered criterion: lex ≤ default, and never worse by more than 1 point. **Met.** Lex is better, and the CI
  excludes 0.
- The label-free choice did as well as the label-based one. Oracle is not better than lex, because selecting on the
  greedy CER of 3 pages is noisy.
- Per-page WER for default and lex is in `e19_report.json`. Page 43 (photo 123) is the hardest: 48 / 48.

**Segmentation against the transcript lines** (weak GT, all 8 seg candidates × 20 eval pages, 773 transcript lines):

| seg chain                                 | lines found (d ≤ 0.4) | extra | split | missed | greedy CER |
| ----------------------------------------- | --------------------- | ----- | ----- | ------ | ---------- |
| up2 clahe+remove_rules (**lex choice**)   | 730 (94.4%)           | 16    | 3     | 5      | **13.23**  |
| up3 contrast+remove_rules (oracle choice) | 727 (94.0%)           | 13    | 8     | 5      | 13.35      |
| up2 contrast+remove_rules (v6 default)    | 727 (94.0%)           | 13    | 5     | 4      | 13.47      |
| up2 illumination+remove_rules             | 724 (93.7%)           | 17    | 6     | 6      | 14.02      |
| up2 illumination+contrast+remove_rules    | 732 (94.7%)           | 26    | 2     | 5      | 14.20      |
| up2 none                                  | 695 (89.9%)           | 77    | 15    | 6      | 14.83      |
| up1 contrast+remove_rules                 | 714 (92.4%)           | 16    | 6     | 5      | 16.51      |
| up2 denoise+contrast+remove_rules         | 713 (92.2%)           | 21    | 4     | 4      | 17.07      |

- The lex choice is also the best segmentation by greedy CER on the eval pages.
- Line recall is close (94.0–94.7%) across the top chains. `remove_rules` matters most: without it the segmentation
  has 77 extra lines.
- The differences among the top 3 are small and have no CI (this table is descriptive).

**Automatic crop** (part `crop`, run 1, crop of commit 32eaf79):

- **Book 1:**
  - Kraken lines on the painted full photo: 375/375 inside the auto crop vs 367/375 inside the hand crop. The hand
    crops cut the folio rubric.
  - IoU with the hand crops is 0.84–0.90 (page 5, a partial page: 0.43). It is descriptive only, since the hand crops
    cut the rubric.
  - v6, official GT: hand 19.40 / 5.26, auto 19.94 / 5.33 (WER / CER). Paired bootstrap: auto − hand = +0.54 points,
    CI95 [−0.49, 1.57], p = 0.34.
  - v6, Gemini references (which include the rubric): hand 8.61 / 3.37, auto 8.33 / 2.86. Paired bootstrap: auto −
    hand = −0.28 points, CI95 [−1.08, 0.51], p = 0.56.
  - Neither difference is significant. Against the official GT the extra rubric line turns into insertions.
- **Book 2:** 1977/1986 Kraken lines inside. All 8 photos with a line outside were inspected:
  - 81 and 107: real writing cut (rubric "Souza Franco" and folio number on the right/top paper edge);
  - 79, 109 and 116: the crop already reaches the paper edge; the segmented box runs past it;
  - 80, 86 and 88: false Kraken lines on the painted watermark (versos; no writing there).
  - **The "no line cut" criterion failed on 2/50 photos.** Of the E19 eval pages, only page 7 (photo 81) is affected:
    3 header tokens ("Souza Franco 39"). The crop is the same for every config, so the default vs lex comparison stands.
- **Fix 1, header on the edge** (`exp/e19-crop-header` 070e327; 8249aa6 before a local rebase, same code). A
  component that touches the paper border is kept when it is a block of writing, not a strip
  (`CropParams.edge_min_side` / `edge_min_area`).
  - Rerun of the boxes only: 3/50 book-2 boxes change (81 and 107 fixed; 105 by +2 px). 1/106 book-1 photos changes
    (photo 46, top −4 px; not an eval page).
  - `edge_min_side=0` reproduces the old boxes exactly (0 mismatches against `crop_book2.json`).
- **TraPrInq validation set with the first crop** (504 pages, another archive): 13072/13497 PAGE-XML lines inside
  (96.9%). Inspection of the 39 pages with a line outside showed one main cause: on dark, stained or densely written
  paper, Otsu on lightness split paper from ink rather than from the background. The longest run of "paper" rows then
  broke, and the paper box kept a band of the page (e.g. 412 px of 2090).
- **Fix 2, paper box** (30a7e59; 69ecc89 before the rebase, same code). The paper box is the union of the plain box
  and a box on the lightness closed with a square of 1/40 of the side (ink removed), with 3% gaps bridged
  (`CropParams.paper_closing`). Alone, the closed box collapsed on 2 of 44 pages tested locally, so only the union is used.
- **Crop v2 check** (part `crop2`, `tools/crop_compare.py`, boxes only: first crop vs crop v2). Free Colab CPU, 1 VM,
  1154 s. Results in `records/e19/crop2_{book1,book2,traprinq}.json`.
  - TraPrInq: lines inside go from 13072 to **13476 of 13497 (99.84%)**. No page loses a line. 21 lines remain
    outside, on 7 pages:
    - 16809_m0014b: 13 lines; both paper boxes keep only the top half;
    - 11735_m0109: 3 header lines above a torn, dark edge;
    - 5 single lines: folio numbers or headers past the paper edge, a box running past the edge.
  - Book 2: 9/50 boxes change and every new box contains the old one; area ×1.06 at most.
  - Book 1 raw photos: 100/106 boxes change, area ×1.13 at most. 3 new boxes (photos 43, 51, 57) do not contain the
    old one. The ink threshold is computed over the paper region, which changed, so a few faint pixels at the edge
    can fall below it. The recession is 2–29 px, and the overlays show blank margin there, with no text cut.
  - TraPrInq: 6 pages where the new box does not contain the old one recede 1–47 px, and keep every PAGE-XML line.
  - Watermark detector: it still fires on 7 TraPrInq pages (another archive's stamps; max score 0.59, threshold
    0.55). It only paints the matched box, and no line is lost on those pages (`crop2_traprinq.json`, `watermark`).
  - Neither crop v2 nor fix 1 was run through Kraken or v6. The E19 numbers above use the first crop.

**Cost.** Free Colab CPU only.

- Run 1: VM A 6456 s (crop) + ~1 h of seg; VM B ~3 h. Both lost.
- Run 2: VM A 3299 s (default) + 16062 s (seg); VM B 17814 s (search + lex and oracle runs).
- Crop v2 check: 1 VM, 1154 s.

**Not done / open.**

- Objectives conf, conflex and stat were not run.
- Crop v2: no Kraken or v6 re-run. TraPrInq page 16809_m0014b still loses its bottom half.
- Watermark false positives on another archive: harmless here, not fixed.
- One book only, 20 pages; Gemini references.
- Shipped as an opt-in path in the repository README (crop, lex search, batch); `entrega_ocr` not rebuilt (built from the main code; these branches are not merged).
