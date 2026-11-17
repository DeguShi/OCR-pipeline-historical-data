#!/bin/bash
# E20 driver (full-system comparison, vault/integracao/E20_sistemas_completos_plan.md).
#   bash tools/run_e20.sh PART OUT [DEVICE]
#   PART smoke   R0-R3 (configs/e20/*.json, configs/final_v6.json for R1) on tuning page 6 only: posterior export,
#                P0 (greedy + text rules) and P6 (our post-correction) scored; wall time and peak RSS per step
#                (/usr/bin/time -v) in OUT/smoke/time_*.txt. Checks that every recognizer loads and exports.
#   PART tune_post  the same on the 9 tuning pages for R0, R2, R3 (E20_RECS overrides), in OUT/tune.
#   PART train  R4/R5 fine-tuning on the rebuilt st_c1m training lines (tools/rebuild_training_lines.py), in OUT/train.
#   PART choose  step 5, selection of PA/PB/P6 per recognizer on the tuning pages, in OUT/select.
#   PART eval_export / eval_pe / eval_decode  step 6, once, after the selection is frozen (records/e20/select/frozen.json):
#                posteriors of held-out and test10 (+ R1 v6 reproduction check), R4·PE (VPY = VLM env), and the
#                P0/PA/PB/P6 decodes, all scored in OUT/eval/score/SET/REC/DECODER/REF.
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

set_args() {  # page set -> images, references and pages for the export tools
  case $1 in
    heldout) echo "--pages 7,10,12" ;;
    test10) echo "--pages all --images data/extra/teste/imagens --references data/extra/teste/referencias" ;;
  esac
}

score_set() {  # score_set LINES CONFIG SET OUT [--no-rules]: held-out against both references, test10 against its own
  local lines=$1 cfg=$2 s=$3 out=$4; shift 4
  if [ "$s" = heldout ]; then
    $PY tools/score_lines.py --lines "$lines" --config "$cfg" "$@" --references data/ground_truth --out "$out/oficial"
    $PY tools/score_lines.py --lines "$lines" --config "$cfg" "$@" --references data/ground_truth_gemini --out "$out/gemini"
  else
    $PY tools/score_lines.py --lines "$lines" --config "$cfg" "$@" --images data/extra/teste/imagens \
        --references data/extra/teste/referencias --out "$out/gemini"
  fi
}

decode_one() {  # decode_one REC SET DECODER: one frozen decode of the exported posteriors, character map, score
  local r=$1 s=$2 d=$3 P=$T/post/$2/$1 W=$T/raw/$2/$1/$3 L=$T/lines/$2/$1
  mkdir -p "$L"; rm -rf "$W"
  case $d in
    p0) $PY tools/greedy_lines.py --posteriors "$P" --pages all --out "$W" ;;
    pa) read -r lm corpus a b unk beam bp tm < <($PY -c "import json,sys; p=json.load(open(sys.argv[1]))[sys.argv[2]]['PA']['params']; m={'kn3_w1':'outputs/lm/kn3_w1.bin outputs/lm/period_w1.txt','kn3_w4b8':'outputs/lm/kn3_w4b8.bin outputs/lm/period_w4_livro78.txt'}; print(m[p['lm']], p['alpha'], p['beta'], p['unk'], p['beam'], p['beam_prune_logp'], p['token_min_logp'])" "$F" "$r")
        $DPY -I tools/baseline_pyctcdecode.py --posteriors "$P" --pages all --lm "$lm" --corpus "$corpus" --alpha "$a" \
            --beta "$b" --unk "$unk" --beam "$beam" --beam-prune-logp "$bp" --token-min-logp "$tm" --workers 2 --out "$W" ;;
    pb) $DPY -I tools/baseline_wordbeam.py --posteriors "$P" --pages all --class-sets "$OUT/tune/post/$r" \
            --corpus outputs/lm/period_w4_livro78.txt --mode NGrams --smoothing 0.001 --beam 256 --workers 2 --out "$W" ;;
    p6) $PY tools/postcorr_from_posteriors.py --config "records/e20/select/$r/p6/config.json" --posteriors "$P" \
            --pages all --out "$W" ;;
  esac
  $PY tools/charmap.py --input "$(ls "$W"/lines.json "$W"/*/lines.json 2>/dev/null | head -1)" --out "$L/$d.json"
  if [ "$d" = p6 ]; then score_set "$L/$d.json" "records/e20/select/$r/p6/config.json" "$s" "$T/score/$s/$r/$d" --no-rules
  else score_set "$L/$d.json" "${CONF[$r]}" "$s" "$T/score/$s/$r/$d"; fi
  echo "DECODED $r $s $d"
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
choose)  # step 5, tuning pages only, the same budget per recognizer (posteriors from PART tune_post in OUT/tune):
  # PA = pyctcdecode+KenLM: baseline_search A (id 0 + 60 random) then A2 (center + 30 local) around the best;
  # PB = Word Beam Search at the E17 frozen configuration (B2 id 1), kept classes from this recognizer's tuning
  # posteriors only; P6 = the E16 v6 sweep (tools/p6_sweep.py). Every output goes through tools/charmap.py before
  # Smoke overrides: E20_SEL_N, E20_SEL_N2, E20_PB_BEAM, E20_P6_VARIANTS (comma list).
  T=$OUT/select; TUNE=5,6,8,9,11,13,14,15,16; DPY=${DPY:-python}
  best() { $PY -c "import json,sys; r=json.loads(open(sys.argv[1]).readline()); r['params'].pop('id'); print(json.dumps(r['params']))" "$1"; }
  for r in ${E20_RECS:-r0_medium r1_st_c1m r2_traprinq r3_mccatmus r4_traprinq r5_mccatmus}; do
    c=${CONF[$r]}; post=$OUT/tune/post/$r; S=$T/$r; mkdir -p "$S"
    [ -d "$post" ] || { echo "no tuning posteriors for $r ($post)"; exit 1; }
    for stage in A A2; do
      [ -f "$S/${stage}_scored.jsonl" ] && continue  # stage finished in an earlier run
      if [ $stage = A ]; then extra=(--n "${E20_SEL_N:-60}"); else extra=(--n "${E20_SEL_N2:-30}" --center "$(best "$S/A_scored.jsonl")"); fi
      # E20_SHARDS processes, each a disjoint slice of the same configuration list (--shard i/k), in S/stage/s<i>;
      # every configuration is decoded on its own, so the merged results equal one sequential run.
      K=${E20_SHARDS:-1}; pids=()
      for i in $(seq 0 $((K - 1))); do
        $PY tools/baseline_search.py --kind $stage "${extra[@]}" --seed 0 --shard "$i/$K" --python "$DPY" --root . \
            --posteriors "$post" --out "$S/$stage/s$i" > "$S/${stage}_s$i.log" 2>&1 & pids+=($!)
      done
      for p in "${pids[@]}"; do wait "$p"; done
      cat "$S/$stage"/s*/results.jsonl > "$S/$stage/results.jsonl"
      $PY tools/charmap.py --input "$S/$stage/results.jsonl" --out "$S/$stage/results_mapped.jsonl"
      $PY tools/score_search.py --results "$S/$stage/results_mapped.jsonl" --config "$c" --out "$S/${stage}_scored.jsonl"
    done
    if [ ! -f "$S/pb/result.json" ]; then
      rm -rf "$S/pb_raw"
      $DPY -I tools/baseline_wordbeam.py --posteriors "$post" --pages $TUNE --class-sets "$post" \
          --corpus outputs/lm/period_w4_livro78.txt --mode NGrams --smoothing 0.001 --beam "${E20_PB_BEAM:-256}" --workers "${E20_WORKERS:-2}" \
          --out "$S/pb_raw"
      $PY tools/charmap.py --input "$(ls "$S"/pb_raw/*/lines.json)" --out "$S/pb_lines.json"
      $PY tools/score_lines.py --lines "$S/pb_lines.json" --config "$c" --out "$S/pb"
    fi
    $PY tools/p6_sweep.py --config "$c" --posteriors "$post" --out "$S/p6" --workers "${E20_P6_WORKERS:-2}" \
        --cpus "${E20_WORKERS:-2}" ${E20_P6_VARIANTS:+--variants $E20_P6_VARIANTS}
    $PY -c "import json,sys; s=sys.argv[1]; a=json.loads(open(s+'/A2_scored.jsonl').readline()); b=json.load(open(s+'/pb/result.json'))['corpus']; p=json.load(open(s+'/p6/final_overrides.json')); print('SELECT', sys.argv[2], 'PA', a['wer'], 'PB', round(100*b['wer_norm'], 2), 'P6', p['result']['oficial']['wer'], p['chosen'])" "$S" "$r"
  done
  ;;
eval_export)  # step 6a: posteriors of held-out (7, 10, 12) and test10, once, then the v6 reproduction check on R1
  T=$OUT/eval; mkdir -p "$T"
  for r in ${E20_RECS:-r0_medium r1_st_c1m r2_traprinq r3_mccatmus r4_traprinq r5_mccatmus}; do
    for s in heldout test10; do
      [ -f "$T/post/$s/$r/done" ] && continue
      timed "export_${s}_$r" $PY tools/export_posteriors.py --config "${CONF[$r]}" $(set_args $s) --device "$DEV" \
          --out "$T/post/$s/$r"
      touch "$T/post/$s/$r/done"
    done
  done
  mkdir -p "$T/repro"; ln -sfn ../post/heldout/r1_st_c1m "$T/repro/v6_gt12"; ln -sfn ../post/test10/r1_st_c1m "$T/repro/v6_test10"
  $PY tools/check_v6_reproduction.py --posteriors "$T/repro" && echo "REPRO v6 ok"
  ;;
eval_pe)  # step 6b: R·PE (frozen VLM prompt paper_pt_period) on the recognizers in E20_PE_RECS (default R4, the best
  # literature recipe with our data on the tuning pages: R4·PA 20.71); line crops carry that recognizer's greedy text
  T=$OUT/eval
  for r in ${E20_PE_RECS:-r4_traprinq}; do
    for s in heldout test10; do
      $PY tools/export_line_crops.py --config "${CONF[$r]}" $(set_args $s) --posteriors "$T/post/$s/$r" \
          --out "$T/crops/$s/$r"
      ${VPY:-$PY} tools/baseline_vlm.py --manifest "$T/crops/$s/$r/manifest.json" --pages all --prompt paper_pt_period \
          --out "$T/pe/$s/$r"
      $PY tools/charmap.py --input "$T/pe/$s/$r/lines.json" --out "$T/lines/$s/$r/pe.json"
      score_set "$T/lines/$s/$r/pe.json" "${CONF[$r]}" $s "$T/score/$s/$r/pe"
    done
  done
  ;;
eval_raw)  # rule 3 context row "as published": R0/R2/R3 on the raw image with kraken's default blla and line padding
  # (configs/e20/raw_*.json: no geometric, segmentation or recognition image operations, padding 16), greedy only
  T=$OUT/eval
  for r in r0_medium r2_traprinq r3_mccatmus; do
    for s in heldout test10; do
      P=$T/post_raw/$s/$r
      [ -f "$P/done" ] || { timed "export_raw_${s}_$r" $PY tools/export_posteriors.py --config configs/e20/raw_$r.json \
          $(set_args $s) --device "$DEV" --out "$P"; touch "$P/done"; }
      $PY tools/greedy_lines.py --posteriors "$P" --pages all --out "$T/raw/$s/$r/p0raw"
      mkdir -p "$T/lines/$s/$r"
      $PY tools/charmap.py --input "$T/raw/$s/$r/p0raw/lines.json" --out "$T/lines/$s/$r/p0raw.json"
      score_set "$T/lines/$s/$r/p0raw.json" configs/e20/raw_$r.json $s "$T/score/$s/$r/p0raw"
    done
  done
  ;;
eval_decode)  # step 6c: P0/PA/PB/P6 of every recognizer on held-out and test10 with the frozen configurations
  # (records/e20/select/frozen.json), each decode once; E20_WORKERS jobs at a time
  T=$OUT/eval; F=records/e20/select/frozen.json; DPY=${DPY:-python}; jobs=()
  for r in ${E20_RECS:-r0_medium r1_st_c1m r2_traprinq r3_mccatmus r4_traprinq r5_mccatmus}; do
    for s in heldout test10; do for d in p0 pa pb p6; do jobs+=("$r $s $d"); done; done
  done
  for j in "${jobs[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "${E20_WORKERS:-2}" ]; do wait -n; done
    decode_one $j > "$T/log_$(echo $j | tr ' ' _).txt" 2>&1 &
  done
  wait
  $PY -c "
import json,sys,pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob('*/*/*/*/result.json')):
    c=json.loads(p.read_text())['corpus']; print('EVAL', *p.parts[-5:-1], round(100*c['wer_norm'],2), round(100*c['cer_norm'],2))" "$T/score"
  ;;
*) echo "unknown part $PART"; exit 1 ;;
esac
