#!/bin/bash
# E18 Q3 driver: per real book (case folder from tools/traprinq_prepare.py), search the chain on its sample page with
# the lex and conf objectives (the two that worked on the synthetic shifts) and the one-page oracle, then run the full
# v6 pipeline on the case's other pages with the default config and with each distinct chosen config (identical
# configs, compared as canonical JSON, run once and are listed in same_as.txt).
# usage: tools/run_e18_books.sh BOOKS_DIR OUT DEVICE STATS_JSON
set -euo pipefail
BOOKS=$1; OUT=$2; DEV=$3; STATS=$4
PY=${PY:-python}; CFG=configs/final_v6.json
canon() { $PY -c "import json,sys,hashlib; print(hashlib.md5(json.dumps(json.load(open(sys.argv[1])), sort_keys=True).encode()).hexdigest()[:12])" "$1"; }
cases=$($PY -c "import json; [print(c['case'], c['sample'], ','.join(map(str, c['eval']))) for c in json.load(open('$BOOKS/manifest.json'))['cases']]")
while read -r name sample evalp; do
  d="$BOOKS/$name"; imgs=("$d"/images/"$sample".*.jpg); img="${imgs[0]}"; ref="$d/references/$(basename "${img%.jpg}").txt"
  [ -f "$OUT/$name/adapt/chosen_lex.json" ] || $PY tools/adapt_preprocess.py search --config $CFG --sample "$img" \
      --reference "$ref" --stats "$STATS" --objectives lex conf oracle --out "$OUT/$name/adapt" \
      --cache "$OUT/$name/cache" --device "$DEV"
  declare -A seen=()
  for cfgname in default lex conf oracle; do
    cfg=$CFG; [ $cfgname = default ] || cfg="$OUT/$name/adapt/chosen_$cfgname.json"
    h=$(canon "$cfg")
    if [ -n "${seen[$h]:-}" ]; then echo "$name/$cfgname = ${seen[$h]}" >> "$OUT/$name/same_as.txt"; continue; fi
    seen[$h]=$cfgname
    [ -f "$OUT/$name/runs/$cfgname/result.json" ] || $PY -m ocr_pipeline.run --config "$cfg" --images "$d/images" \
        --references "$d/references" --pages "$evalp" --out "$OUT/$name/runs" --name $cfgname \
        --cache "$OUT/$name/cache" --device "$DEV"
  done
  unset seen; echo "BOOK $name done"
done <<< "$cases"
echo "E18 BOOKS DONE"
