# Qualitative examples (E17), generated 2026-11-08

`tools/qualitative_examples.py --per-category 4` on the frozen outputs in `../method_outputs` and the line crops
(run on the Colab VM e18, code `2acb41c`). Of 429 lines (test10 + held-out 7, 10, 12), 380 pass the alignment
sanity check; lines per category: v6_fixes_others_miss 40, vlm_modernizes 55, byt5_better 14, v6_breaks 6.
`examples.tex` is a pt-BR LaTeX fragment (crop + reference + every method, wrong words in `\erro{}`), the crops are
in `crops/`, all texts and per-method word errors in `examples.json`.

Notes for the paper: p10 l18 (held-out) is listed as "v6_breaks" only because the official GT modernizes `mezes` to
`meses` (see "O gabarito oficial"); p22 l29 (test) is a real v6 failure (`denominados` read as `consummados`).
