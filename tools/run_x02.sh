#!/bin/bash
# X02: the PA (pyctcdecode + KenLM) of E20 given the search budget of E17's baseline A (181 configurations instead
# of 92), on the recognizers of the published-parts comparison. Question: does R1·P6's lead over R4·PA / R5·PA on the
# tuning-selected systems survive a PA search as large as E17's? Same tools, seeds and rules as E20 (tools/run_e20.sh
# PART choose / eval_decode); only the budget changes: A = id 0 + 120 random (E20: 60), A2 = center + 60 local
# (E20: 30). The random and local lists are prefixes of each other for a fixed seed, so E20's 92 configurations are
# re-decoded here and must reproduce E20's tuning WER (reproduction check in PART select).
#
#   PART select  tuning pages only (5, 6, 8, 9, 11, 13, 14, 15, 16): A then A2 per recognizer, then freeze
#                OUT/select/frozen_x02.json. Nothing in this part reads a held-out or test page.
#   PART eval    once, after the freeze: decode held-out (7, 10, 12) and test10 with the frozen configuration, character
#                map, score against both references (held-out) and the Gemini reference (test10).
#   PART report  paired comparisons (word-block bootstrap and page-level tests) against the E20 records.
# Inputs (not in git): OUT/tune/post/<rec> and OUT/eval/post/<set>/<rec> (the E20 exports), outputs/lm/ (KenLM).
#   PY=... DPY=... X02_SHARDS=k bash tools/run_x02.sh select OUT
set -euo pipefail
PART=$1; OUT=$2
PY=${PY:-python}; DPY=${DPY:-python}
TUNE=5,6,8,9,11,13,14,15,16
RECS=${X02_RECS:-r1_st_c1m r4_traprinq r5_mccatmus}
declare -A CONF=([r1_st_c1m]=configs/final_v6.json [r4_traprinq]=configs/e20/r4_traprinq.json
                 [r5_mccatmus]=configs/e20/r5_mccatmus.json)
F=$OUT/select/frozen_x02.json

best() { $PY -c "import json,sys; r=json.loads(open(sys.argv[1]).readline()); r['params'].pop('id'); print(json.dumps(r['params']))" "$1"; }

score_set() {  # as in tools/run_e20.sh
  local lines=$1 cfg=$2 s=$3 out=$4
  if [ "$s" = heldout ]; then
    $PY tools/score_lines.py --lines "$lines" --config "$cfg" --references data/ground_truth --out "$out/oficial"
    $PY tools/score_lines.py --lines "$lines" --config "$cfg" --references data/ground_truth_gemini --out "$out/gemini"
  else
    $PY tools/score_lines.py --lines "$lines" --config "$cfg" --images data/extra/teste/imagens \
        --references data/extra/teste/referencias --out "$out/gemini"
  fi
}

# Inputs every part reads from the repository (the 2026-11-18 smoke test died on a missing charmap label file).
for f in records/training/st_c1m/labels_r1/lines.jsonl ${CONF[r1_st_c1m]} ${CONF[r4_traprinq]} ${CONF[r5_mccatmus]} \
         $(for r in $RECS; do echo records/e20/select/$r/A_scored.jsonl records/e20/select/$r/A2_scored.jsonl; done); do
  [ -f "$f" ] || { echo "missing input: $f"; exit 1; }
done

case $PART in
select)
  [ -f "$F" ] && { echo "already frozen: $F"; exit 1; }
  for r in $RECS; do
    c=${CONF[$r]}; post=$OUT/tune/post/$r; S=$OUT/select/$r; mkdir -p "$S"
    [ -d "$post" ] || { echo "no tuning posteriors for $r ($post)"; exit 1; }
    for stage in A A2; do
      [ -f "$S/${stage}_scored.jsonl" ] && continue
      if [ $stage = A ]; then extra=(--n "${X02_N:-120}"); else extra=(--n "${X02_N2:-60}" --center "$(best "$S/A_scored.jsonl")"); fi
      K=${X02_SHARDS:-1}; pids=()
      for i in $(seq 0 $((K - 1))); do
        $PY tools/baseline_search.py --kind $stage "${extra[@]}" --seed 0 --shard "$i/$K" --python "$DPY" --root . \
            --posteriors "$post" --out "$S/$stage/s$i" > "$S/${stage}_s$i.log" 2>&1 & pids+=($!)
      done
      for p in "${pids[@]}"; do wait "$p"; done
      cat "$S/$stage"/s*/results.jsonl > "$S/$stage/results.jsonl"
      $PY tools/charmap.py --input "$S/$stage/results.jsonl" --out "$S/$stage/results_mapped.jsonl"
      $PY tools/score_search.py --results "$S/$stage/results_mapped.jsonl" --config "$c" --out "$S/${stage}_scored.jsonl"
      echo "STAGE $r $stage best $(head -1 "$S/${stage}_scored.jsonl" | cut -c1-200)"
    done
  done
  # Reproduction check: every configuration E20 scored (records/e20/select/<rec>/A*_scored.jsonl; R1 from E20 too)
  # must have the same tuning WER here. Then freeze the best A2 configuration of each recognizer.
  $PY - "$OUT" $RECS <<'EOF'
import json, sys
from pathlib import Path
out, recs = Path(sys.argv[1]), sys.argv[2:]
frozen, bad = {}, 0
for r in recs:
    for stage in ('A', 'A2'):
        old = {json.dumps({k: v for k, v in x['params'].items() if k != 'id'}, sort_keys=True): x['wer']
               for x in map(json.loads, open(f'records/e20/select/{r}/{stage}_scored.jsonl'))}
        new = {json.dumps({k: v for k, v in x['params'].items() if k != 'id'}, sort_keys=True): x['wer']
               for x in map(json.loads, open(out / f'select/{r}/{stage}_scored.jsonl'))}
        shared = old.keys() & new.keys()
        diff = [k for k in shared if abs(old[k] - new[k]) > 1e-9]
        bad += len(diff)
        print('REPRO', r, stage, 'shared', len(shared), 'of E20', len(old), 'differ', len(diff))
    a2 = [json.loads(l) for l in open(out / f'select/{r}/A2_scored.jsonl')]
    a = [json.loads(l) for l in open(out / f'select/{r}/A_scored.jsonl')]
    top = min(a + a2, key=lambda x: (x['wer'], x['cer']))
    params = {k: v for k, v in top['params'].items() if k != 'id'}
    frozen[r] = {'PA181': {'params': params, 'tuning_wer': top['wer'], 'tuning_cer': top['cer'],
                           'n_scored': len(a) + len(a2)}}
    print('FROZEN', r, json.dumps(frozen[r]))
(out / 'select/frozen_x02.json').write_text(json.dumps(frozen, indent=1))
print('REPRO_OK' if bad == 0 else f'REPRO_DIFF {bad}')
EOF
  ;;
eval)
  [ -f "$F" ] || { echo "not frozen: $F"; exit 1; }
  T=$OUT/eval
  for r in $RECS; do
    for s in heldout test10; do
      P=$T/post/$s/$r; W=$T/raw/$s/$r/pa181; L=$T/lines/$s/$r; mkdir -p "$L"; rm -rf "$W"
      read -r lm corpus a b unk beam bp tm < <($PY -c "import json,sys; p=json.load(open(sys.argv[1]))[sys.argv[2]]['PA181']['params']; m={'kn3_w1':'outputs/lm/kn3_w1.bin outputs/lm/period_w1.txt','kn3_w4b8':'outputs/lm/kn3_w4b8.bin outputs/lm/period_w4_livro78.txt'}; print(m[p['lm']], p['alpha'], p['beta'], p['unk'], p['beam'], p['beam_prune_logp'], p['token_min_logp'])" "$F" "$r")
      $DPY -I tools/baseline_pyctcdecode.py --posteriors "$P" --pages all --lm "$lm" --corpus "$corpus" --alpha "$a" \
          --beta "$b" --unk "$unk" --beam "$beam" --beam-prune-logp "$bp" --token-min-logp "$tm" --workers 2 --out "$W"
      $PY tools/charmap.py --input "$(ls "$W"/lines.json "$W"/*/lines.json 2>/dev/null | head -1)" --out "$L/pa181.json"
      score_set "$L/pa181.json" "${CONF[$r]}" "$s" "$T/score/$s/$r/pa181"
      echo "DECODED $r $s pa181"
    done
  done
  ;;
report)
  $PY tools/x02_report.py --e20 records/e20/eval --x02 "$OUT/eval" --out "$OUT/report"
  ;;
*) echo "unknown part $PART"; exit 1 ;;
esac
