#!/bin/bash
# E20 driver (full-system comparison, vault/integracao/E20_sistemas_completos_plan.md).
#   bash tools/run_e20.sh PART OUT [DEVICE]
#   PART smoke   R0-R3 (configs/e20/*.json, configs/final_v6.json for R1) on tuning page 6 only: posterior export,
#                P0 (greedy + text rules) and P6 (our post-correction) scored; wall time and peak RSS per step
#                (/usr/bin/time -v) in OUT/smoke/time_*.txt. Checks that every recognizer loads and exports.
#   PART tune_post  the same on the 9 tuning pages for R0, R2, R3 (E20_RECS overrides), in OUT/tune.
#   PART train  R4/R5 fine-tuning on the rebuilt st_c1m training lines (tools/rebuild_training_lines.py), in OUT/train.
# Only tuning pages here; held-out (7, 10, 12) and test are decoded once, after the configurations are frozen.
set -euo pipefail
PART=$1; OUT=$2; DEV=${3:-cpu}
PY=${PY:-python}
SMOKE_PAGES=${E20_SMOKE_PAGES:-6}
declare -A CONF=([r0_medium]=configs/e20/r0_medium.json [r1_st_c1m]=configs/final_v6.json
                 [r2_traprinq]=configs/e20/r2_traprinq.json [r3_mccatmus]=configs/e20/r3_mccatmus.json
                 [r4_traprinq]=configs/e20/r4_traprinq.json [r5_mccatmus]=configs/e20/r5_mccatmus.json)

timed() {  # timed NAME CMD...: run CMD under /usr/bin/time -v, log in $T/time_NAME.txt
  local name=$1; shift
  /usr/bin/time -v "$@" 2> "$T/time_$name.txt" || { tail -20 "$T/time_$name.txt"; return 1; }
  grep -E 'Elapsed|Maximum resident' "$T/time_$name.txt" | sed "s/^/  $name /"
}

case $PART in
smoke)
  T=$OUT/smoke; mkdir -p "$T"
  for r in r0_medium r1_st_c1m r2_traprinq r3_mccatmus; do
    c=${CONF[$r]}
    $PY setup_pipeline.py --skip-install --config "$c"
    timed "export_$r" $PY tools/export_posteriors.py --config "$c" --pages "$SMOKE_PAGES" --device "$DEV" \
        --out "$T/post/$r"
    $PY tools/greedy_lines.py --posteriors "$T/post/$r" --pages "$SMOKE_PAGES" --out "$T/p0/$r"
    $PY tools/score_lines.py --lines "$T/p0/$r/lines.json" --config "$c" --out "$T/runs/p0_$r"
    timed "p6_$r" $PY tools/postcorr_from_posteriors.py --config "$c" --posteriors "$T/post/$r" \
        --pages "$SMOKE_PAGES" --out "$T/p6/$r"
    $PY tools/score_lines.py --lines "$T/p6/$r/lines.json" --config "$c" --no-rules --out "$T/runs/p6_$r"
    echo "SMOKE $r done"
  done
  ;;
tune_post)  # step 4, tuning pages only: posteriors + P0 + P6 of the published recognizers (R1's exist from E17)
  T=$OUT/tune; mkdir -p "$T"; TUNE=5,6,8,9,11,13,14,15,16
  for r in ${E20_RECS:-r0_medium r2_traprinq r3_mccatmus}; do
    c=${CONF[$r]}
    timed "export_$r" $PY tools/export_posteriors.py --config "$c" --pages $TUNE --device "$DEV" --out "$T/post/$r"
    $PY tools/greedy_lines.py --posteriors "$T/post/$r" --pages $TUNE --out "$T/p0/$r"
    $PY tools/score_lines.py --lines "$T/p0/$r/lines.json" --config "$c" --out "$T/runs/p0_$r"
    timed "p6_$r" $PY tools/postcorr_from_posteriors.py --config "$c" --posteriors "$T/post/$r" --pages $TUNE \
        --out "$T/p6/$r"
    $PY tools/score_lines.py --lines "$T/p6/$r/lines.json" --config "$c" --no-rules --out "$T/runs/p6_$r"
    echo "TUNE $r done"
  done
  ;;
train)  # R4/R5: TraPrInq / McCATMuS fine-tuned on the rebuilt st_c1m lines with the st_c1m recipe (E16)
  # E20_TRAIN: jobs NAME:INIT:LR; E20_EPOCHS (20 as st_c1m; 1 for the smoke). Batch 16 like st_c1m, falling back to
  # 8 or 4 with gradient accumulation (same effective batch 16) on out-of-memory. Precision 16-mixed: the T4 has no
  # bf16 (st_c1m was bf16-mixed on an L4), the one stated deviation.
  T=$OUT/train; mkdir -p "$T"; KETOS=${KETOS:-ketos}
  [ -f "$T/lines/select.json" ] || $PY tools/rebuild_training_lines.py --out "$T/lines"
  [ -f "$T/train.arrow" ] || $KETOS --workers 2 compile -f page -o "$T/train.arrow" "$T"/lines/train/*.xml \
      > "$T/compile.log" 2>&1
  sha256sum "$T/train.arrow" | tee "$T/train.arrow.sha256"
  for job in ${E20_TRAIN:-r4_traprinq:traprinq:1e-4 r5_mccatmus:mccatmus:1e-4}; do
    IFS=: read -r name init lr <<< "$job"
    initpath=$($PY -c "from ocr_pipeline.engine import MODELS; print(MODELS['$init'])")
    tag=${name}_lr$lr${E20_TAG:-}; ok=0
    for b in 16 8 4; do
      m=$T/model_$tag; rm -rf "$m"; mkdir -p "$T/tb_$tag"
      if $KETOS -d "$DEV" -s 42 --workers 2 --threads 2 --precision 16-mixed train -f binary -i "$initpath" \
          -o "$m" --logger tensorboard --log-dir "$T/tb_$tag" -B $b --accumulate-grad-batches $((16 / b)) -r "$lr" \
          --augment -q fixed -N "${E20_EPOCHS:-20}" -p 0.9 -u NFD --resize union "$T/train.arrow" \
          > "$T/train_$tag.log" 2>&1; then ok=1; break; fi
      grep -q "out of memory" "$T/train_$tag.log" || { tail -30 "$T/train_$tag.log"; exit 1; }
      mv "$T/train_$tag.log" "$T/train_${tag}_oom_b$b.log"
    done
    [ $ok = 1 ] || { echo "TRAIN $tag out of memory at every batch"; exit 1; }
    best=$(ls "$m"/*best*.safetensors); cp "$best" "$T/$tag.safetensors"
    echo "TRAIN $tag batch $b best $(basename "$best") sha256 $(sha256sum "$T/$tag.safetensors" | cut -c1-16)"
  done
  ;;
select)  # step 5, tuning pages only, the same budget per recognizer (posteriors from PART tune_post in OUT/tune):
  # PA = pyctcdecode+KenLM: baseline_search A (id 0 + 60 random) then A2 (center + 30 local) around the best;
  # PB = Word Beam Search at the E17 frozen configuration (B2 id 1), kept classes from this recognizer's tuning
  # posteriors only; P6 = the E16 v6 sweep (tools/p6_sweep.py). Every output goes through tools/charmap.py before
$1
  # Smoke overrides: E20_SEL_N, E20_SEL_N2, E20_PB_BEAM, E20_P6_VARIANTS (comma list).
  T=$OUT/select; TUNE=5,6,8,9,11,13,14,15,16; DPY=${DPY:-python}
  best() { $PY -c "import json,sys; r=json.loads(open(sys.argv[1]).readline()); r['params'].pop('id'); print(json.dumps(r['params']))" "$1"; }
  for r in ${E20_RECS:-r0_medium r1_st_c1m r2_traprinq r3_mccatmus r4_traprinq r5_mccatmus}; do
    c=${CONF[$r]}; post=$OUT/tune/post/$r; S=$T/$r; mkdir -p "$S"
    [ -d "$post" ] || { echo "no tuning posteriors for $r ($post)"; exit 1; }
    for stage in A A2; do
      if [ $stage = A ]; then extra=(--n "${E20_SEL_N:-60}"); else extra=(--n "${E20_SEL_N2:-30}" --center "$(best "$S/A_scored.jsonl")"); fi
      $PY tools/baseline_search.py --kind $stage "${extra[@]}" --seed 0 --python "$DPY" --root . --posteriors "$post" \
          --out "$S/$stage"
      $PY tools/charmap.py --input "$S/$stage/results.jsonl" --out "$S/$stage/results_mapped.jsonl"
      $PY tools/score_search.py --results "$S/$stage/results_mapped.jsonl" --config "$c" --out "$S/${stage}_scored.jsonl"
    done
    if [ ! -f "$S/pb/result.json" ]; then
      rm -rf "$S/pb_raw"
      $DPY -I tools/baseline_wordbeam.py --posteriors "$post" --pages $TUNE --class-sets "$post" \
          --corpus outputs/lm/period_w4_livro78.txt --mode NGrams --smoothing 0.001 --beam "${E20_PB_BEAM:-256}" --workers 2 \
          --out "$S/pb_raw"
      $PY tools/charmap.py --input "$(ls "$S"/pb_raw/*/lines.json)" --out "$S/pb_lines.json"
      $PY tools/score_lines.py --lines "$S/pb_lines.json" --config "$c" --out "$S/pb"
    fi
    $PY tools/p6_sweep.py --config "$c" --posteriors "$post" --out "$S/p6" --workers 2 ${E20_P6_VARIANTS:+--variants $E20_P6_VARIANTS}
    $PY -c "import json,sys; s=sys.argv[1]; a=json.loads(open(s+'/A2_scored.jsonl').readline()); b=json.load(open(s+'/pb/result.json'))['corpus']; p=json.load(open(s+'/p6/final_overrides.json')); print('SELECT', sys.argv[2], 'PA', a['wer'], 'PB', round(100*b['wer_norm'], 2), 'P6', p['result']['oficial']['wer'], p['chosen'])" "$S" "$r"
  done
  ;;
*) echo "unknown part $PART"; exit 1 ;;
esac
