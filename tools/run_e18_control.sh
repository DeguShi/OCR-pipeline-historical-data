#!/bin/bash
# E18 controls, run after tools/run_e18.sh into the same OUT (reuses its reference stats, shifts and caches).
# (a) identity: the whole adaptation run on our own book (JPEG re-saved at quality 95); it should keep the default
#     chain or at least not hurt.
# (b) fixed_clahe: one static chain (segmentation image clahe+remove_rules, the chain the lexicon objective and the
#     oracle chose most often in the main run) on the clean pages and on every shift, to tell per-book adaptation
#     apart from a better static config.
# usage: tools/run_e18_control.sh OUT DEVICE [SHIFTS...]
set -euo pipefail
OUT=$1; DEV=$2; shift 2
SHIFTS=("$@"); [ ${#SHIFTS[@]} -eq 0 ] && SHIFTS=(uneven_light low_contrast_cast faded_ink bleed_through jpeg_noise low_res_blur)
PY=${PY:-python}; CFG=configs/final_v6.json; FIX=(--set seg_ops=clahe,remove_rules)
tools/run_e18.sh "$OUT" "$DEV" identity
$PY -m ocr_pipeline.run --config $CFG "${FIX[@]}" --pages heldout --out "$OUT/clean/runs" --name fixed_clahe \
    --cache "$OUT/clean/cache" --device "$DEV"
for s in "${SHIFTS[@]}"; do
  d="$OUT/shifts/$s"; [ -d "$d" ] || { echo "skip $s (no shift dir)"; continue; }
  $PY -m ocr_pipeline.run --config $CFG "${FIX[@]}" --images "$d/images" --references "$d/references" \
      --pages heldout --out "$OUT/$s/runs" --name fixed_clahe --cache "$OUT/$s/cache" --device "$DEV"
  echo "FIXED $s done"
done
echo "E18 CONTROL DONE"
