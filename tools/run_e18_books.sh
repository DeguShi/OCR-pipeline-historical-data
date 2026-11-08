#!/bin/bash
# E18 Q3 driver: per real book (case folder from tools/traprinq_prepare.py), search the chain on its sample page with
# the lex and conf objectives (the two that worked on the synthetic shifts), then run the full v6 pipeline on the
# case's other pages with the default config and with each chosen config.
# usage: tools/run_e18_books.sh BOOKS_DIR OUT DEVICE STATS_JSON
set -euo pipefail
BOOKS=$1; OUT=$2; DEV=$3; STATS=$4
PY=${PY:-python}; CFG=configs/final_v6.json
cases=$($PY -c "import json,sys; [print(c['case'], c['sample'], ','.join(map(str, c['eval']))) for c in json.load(open('$BOOKS/manifest.json'))['cases']]")
while read -r name sample evalp; do
  d="$BOOKS/$name"; imgs=("$d"/images/"$sample".*.jpg); img="${imgs[0]}"; ref="$d/references/$(basename "${img%.jpg}").txt"
  $PY tools/adapt_preprocess.py search --config $CFG --sample "$img" --reference "$ref" --stats "$STATS" \
      --objectives lex conf oracle --out "$OUT/$name/adapt" --cache "$OUT/$name/cache" --device "$DEV"
  for cfgname in default lex conf oracle; do
    cfg=$CFG; [ $cfgname = default ] || cfg="$OUT/$name/adapt/chosen_$cfgname.json"
    $PY -m ocr_pipeline.run --config "$cfg" --images "$d/images" --references "$d/references" --pages "$evalp" \
        --out "$OUT/$name/runs" --name $cfgname --cache "$OUT/$name/cache" --device "$DEV"
  done
  echo "BOOK $name done"
done <<< "$cases"
echo "E18 BOOKS DONE"
