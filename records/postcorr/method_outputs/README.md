# Frozen outputs of every post-correction method (E17)

Each file is the scored `result.json` of the run behind one row of the comparison table (`pages[].text` = the
page text, one line per segmented line, aligned with the line crops of `tools/export_line_crops.py`). `*_gt12`
hold all 12 GT pages (the table's held-out numbers use pages 7, 10, 12; tuning pages are 5, 6, 8, 9, 11, 13-16);
`E_vlm_heldout` holds only the 3 held-out pages; `*_test10` the 10 test pages (Gemini reference).

| file prefix | method | source run |
|---|---|---|
| v6 | ours | outputs/runs/v6_all12_cache, v6_test10_cache |
| A_pyctc | pyctcdecode + KenLM 3-gram (frozen A2 id 11) | outputs/runs/base_A2_pyctc_* |
| B_wbs | CTC word beam search, NGrams (frozen B2 id 1) | outputs/runs/base_B2_wbs_* |
| C_byt5 | ByT5-small, synthetic noise (paper run) | outputs/baselines/byt5/{gt12,test10}/score |
| D_mlm | v6 + PortOldBERT PLL, lambda 0.5 | outputs/baselines/mlm/*_lam0.5/score |
| E_vlm | Qwen2.5-VL-7B 4-bit, prompt paper_pt_period | outputs/baselines/search/vlm/final_*_paper_pt_period/score |
| hybrid | v6 + ByT5 on numerals (negative result) | outputs/baselines/hybrid/*_numeral/score |

Copied 2026-11-08 for the paper's qualitative examples (`tools/qualitative_examples.py`).

`raw_greedy_*_lines.json`: the recognizer's greedy best path before any post-correction ("sem pós-correção"),
as {page: [line, ...]} (outputs/baselines/greedy_*.json).
