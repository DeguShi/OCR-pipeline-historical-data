# Baseline: published PP-OCRv6 medium on the raw image (run of 2026-10-25)

Recovered from `~/hackathon_ocr_historico/outputs/runs/raw` (the original output dir; it holds the run over all 12
GT pages, `pages_subset: all`, 63.52 / 22.83). The held-out run of the same config wrote to the same dir and was
overwritten. The held-out numbers in the paper are recomputed from the per-page counts of this file:

    jq '[.pages[] | select(.page==7 or .page==10 or .page==12)]
        | {wer: ((map(.norm_we)|add)/(map(.norm_rw)|add)), cer: ((map(.norm_ce)|add)/(map(.norm_rc)|add))}' result.json
    -> wer 0.64567, cer 0.21712   (paper: 64,57 / 21,71)

Config (in `result.json`): no image ops, default blla segmentation, medium, padding 16, greedy, no text rules.
E20's raw row (`records/e20/eval/score/heldout/r0_medium/p0raw_norules`) has the same config plus the E20 character
map and a newer Kraken; it gives 62.99 / 21.42. The 1.6-point gap is not isolated.
