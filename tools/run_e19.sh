#!/bin/bash
# E19 driver: a new book of the same archive (Livro de Atas da Câmara de São Carlos 1894-1902), no training.
# usage: tools/run_e19.sh PART DATA OUT [DEVICE]
#   DATA holds imagens/ (raw photos), transcricoes/ (Gemini, one text line per page line) and E19_split.json.
#   PART smoke  book prep + 1 seg candidate on 1 eval page + 1 search candidate on 1 sample page + v6 on 1 eval page
#   PART crop   crop checks (book 2 photos, book 1 hand crops, TraPrInq PAGE XML when TRAPRINQ_IMG/TRAPRINQ_XML are
#               set) and v6 on the 12 book-1 GT pages, auto crop vs hand crop (official references)
#   PART seg    weak-GT segmentation of the 8 seg candidates on the eval pages (tools/seg_weak_eval.py)
#   PART main   chain search on the 3 sample pages (lex conf conflex stat, oracle = greedy CER on the samples),
#               then full v6 on the 20 eval pages with the default config and each distinct chosen config
#               (identical configs, compared as canonical JSON, run once and listed in same_as.txt)
#   PART default  full v6 on the 20 eval pages with the default config only (run on the other VM; then main is
#               called with E19_SKIP_DEFAULT=1, which still dedups chosen configs against the default)
set -euo pipefail
PART=$1; DATA=$2; OUT=$3; DEV=${4:-cpu}
PY=${PY:-python}; CFG=configs/final_v6.json; BOOK=$OUT/book; STATS=records/e19/reference_stats_e18.json
mkdir -p "$OUT"
[ -f "$BOOK/manifest.json" ] || $PY tools/book_prepare.py --images "$DATA/imagens" --transcripts "$DATA/transcricoes" \
    --split "$DATA/E19_split.json" --out "$BOOK"
pages() { $PY -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['cases'][0][sys.argv[2]])))" \
    "$BOOK/manifest.json" "$1"; }
img() { ls "$BOOK"/images/"$1".*.jpg; }
ref() { local i; i=$(img "$1"); echo "$BOOK/references/$(basename "${i%.jpg}").txt"; }
canon() { $PY -c "import json,sys,hashlib; print(hashlib.md5(json.dumps(json.load(open(sys.argv[1])), sort_keys=True).encode()).hexdigest()[:12])" "$1"; }
read -r -a SAMPLE <<< "$(pages sample)"; read -r -a EVAL <<< "$(pages eval)"
EVALP=$(IFS=,; echo "${EVAL[*]}")

search() {  # search OUTDIR LIMIT PAGES...
  local out=$1 limit=$2; shift 2; local s=() r=()
  for n in "$@"; do s+=("$(img "$n")"); r+=("$(ref "$n")"); done
  $PY tools/adapt_preprocess.py search --config $CFG --sample "${s[@]}" --reference "${r[@]}" --stats $STATS \
      --objectives lex conf conflex stat oracle --out "$out" --cache "$OUT/cache" --device "$DEV" --limit "$limit"
}

case $PART in
smoke)
  SECONDS=0
  $PY tools/seg_weak_eval.py --book "$BOOK" --out "$OUT/smoke/seg" --cache "$OUT/cache" --device "$DEV" --limit 1
  search "$OUT/smoke/adapt" 1 "${SAMPLE[0]}"
  $PY -m ocr_pipeline.run --config $CFG --images "$BOOK/images" --references "$BOOK/references" --pages "${EVAL[0]}" \
      --out "$OUT/smoke/runs" --name default --cache "$OUT/cache" --device "$DEV"
  echo "SMOKE ${SECONDS}s";;
crop)
  $PY tools/crop_eval.py book2 --raw "$DATA/imagens" --transcripts "$DATA/transcricoes" --out "$OUT/crop" \
      --cache "$OUT/crop_cache" --device "$DEV"
  $PY tools/crop_eval.py book1 --raw data/extra/ldl1 --hand data/images --out "$OUT/crop" --cache "$OUT/crop_cache" --device "$DEV"
  [ -z "${TRAPRINQ_XML:-}" ] || $PY tools/crop_eval.py pagexml --raw "$TRAPRINQ_IMG" --page-xml "$TRAPRINQ_XML" --out "$OUT/crop"
  $PY -c "
import re, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, '.')
from ocr_pipeline.auto_crop import auto_crop
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
raw = {int(re.match(r'\d+', p.name).group()): p for p in Path('data/extra/ldl1').glob('*.jpg')}
for hand in Path('data/images').glob('*.jpg'):
    auto_crop(Image.open(raw[int(hand.name.split('.')[0])]))[0].save(out / hand.name, quality=95)
print('book-1 auto crops', len(list(out.glob('*.jpg'))))" "$OUT/book1_auto"
  for c in hand auto; do
    imgs=data/images; [ $c = auto ] && imgs=$OUT/book1_auto
    [ -f "$OUT/book1_runs/$c/result.json" ] || $PY -m ocr_pipeline.run --config $CFG --images "$imgs" \
        --references data/ground_truth --pages all --out "$OUT/book1_runs" --name $c --cache "$OUT/cache_book1_$c" --device "$DEV"
  done
  echo "E19 CROP DONE";;
seg)
  $PY tools/seg_weak_eval.py --book "$BOOK" --pages eval --out "$OUT/seg_weak" --cache "$OUT/cache" --device "$DEV"
  echo "E19 SEG DONE";;
main)
  [ -f "$OUT/adapt/chosen_lex.json" ] || search "$OUT/adapt" 0 "${SAMPLE[@]}"
  declare -A seen=()
  for name in default lex oracle conf conflex stat; do
    cfg=$CFG; [ $name = default ] || cfg="$OUT/adapt/chosen_$name.json"
    h=$(canon "$cfg")
    if [ -n "${seen[$h]:-}" ]; then echo "$name = ${seen[$h]}" >> "$OUT/runs_same_as.txt"; continue; fi
    seen[$h]=$name
    [ $name = default ] && [ "${E19_SKIP_DEFAULT:-0}" = 1 ] && continue
    [ -f "$OUT/runs/$name/result.json" ] || $PY -m ocr_pipeline.run --config "$cfg" --images "$BOOK/images" \
        --references "$BOOK/references" --pages "$EVALP" --out "$OUT/runs" --name $name --cache "$OUT/cache" --device "$DEV"
  done
  echo "E19 MAIN DONE";;
default)
  [ -f "$OUT/runs/default/result.json" ] || $PY -m ocr_pipeline.run --config $CFG --images "$BOOK/images" \
      --references "$BOOK/references" --pages "$EVALP" --out "$OUT/runs" --name default --cache "$OUT/cache" --device "$DEV"
  echo "E19 DEFAULT DONE";;
*) echo "unknown part $PART"; exit 2;;
esac
