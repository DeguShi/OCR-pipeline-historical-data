#!/bin/bash
# E18 driver: synthetic shifts -> one-page preprocessing search -> held-out evaluation of each chosen chain.
# usage: tools/run_e18.sh OUT DEVICE [SHIFTS...]     (env SMOKE=1: 2 reference pages, 3 candidates per stage)
# Sample page: one tuning page drawn with seed 42. Held-out 7, 10, 12 are only evaluated, never searched on.
set -euo pipefail
OUT=$1; DEV=$2; shift 2
SHIFTS=("$@"); [ ${#SHIFTS[@]} -eq 0 ] && SHIFTS=(low_contrast_cast uneven_light low_res_blur faded_ink jpeg_noise bleed_through)
PY=${PY:-python}; CFG=configs/final_v6.json
if [ "${SMOKE:-0}" = 1 ]; then NREF=2; LIMIT=3; else NREF=8; LIMIT=0; fi
SAMPLE=$($PY -c "import random; print(random.Random(42).choice([5, 6, 8, 9, 11, 13, 14, 15, 16]))")
mkdir -p "$OUT"; echo "sample page $SAMPLE, shifts ${SHIFTS[*]}, smoke ${SMOKE:-0}" | tee "$OUT/driver.txt"
$PY tools/synth_shift.py --out "$OUT/shifts" --shifts "${SHIFTS[@]}"
[ -f "$OUT/reference_stats.json" ] || $PY tools/adapt_preprocess.py reference --config $CFG \
    --images data/extra/treino_crop --n-pages $NREF --out "$OUT/reference_stats.json" --cache "$OUT/cache_ref" --device "$DEV"
[ -f "$OUT/clean/runs/default/result.json" ] || $PY -m ocr_pipeline.run --config $CFG --pages heldout \
    --out "$OUT/clean/runs" --name default --cache "$OUT/clean/cache" --device "$DEV"
for s in "${SHIFTS[@]}"; do
  d="$OUT/shifts/$s"; imgs=("$d"/images/"$SAMPLE".*.jpg); img="${imgs[0]}"; ref="$d/references/$(basename "${img%.jpg}").txt"
  $PY tools/adapt_preprocess.py search --config $CFG --sample "$img" --reference "$ref" --stats "$OUT/reference_stats.json" \
      --limit $LIMIT --out "$OUT/$s/adapt" --cache "$OUT/$s/cache" --device "$DEV"
  declare -A seen=()
  for name in default conf lex conflex stat oracle; do
    cfg=$CFG; [ $name = default ] || cfg="$OUT/$s/adapt/chosen_$name.json"
    h=$(md5sum "$cfg" | cut -c1-12); [ $name = default ] && h=default
    if [ -n "${seen[$h]:-}" ]; then echo "$s/$name = ${seen[$h]}" >> "$OUT/$s/same_as.txt"; continue; fi
    seen[$h]=$name
    $PY -m ocr_pipeline.run --config "$cfg" --images "$d/images" --references "$d/references" --pages heldout \
        --out "$OUT/$s/runs" --name $name --cache "$OUT/$s/cache" --device "$DEV"
  done
  unset seen; echo "SHIFT $s done"
done
echo "E18 DONE"
