#!/bin/bash
# Recognizer-quality curve: per recognizer, raw greedy and v6 (fixed configuration) from that recognizer's posteriors.
set -euo pipefail
cd ~/OCR-pipeline-historical-data
P=~/hackathon_ocr_historico/.venv/bin/python
m=$1  # recognizer: a key of ocr_pipeline/engine.py MODELS with configs/curve_<m>.json (posteriors exported first)
until grep -q "EXPORT DONE $m" outputs/logs/curve_export_$m.log; do sleep 20; done
for s in gt12 test10; do
  X=(); [ $s = test10 ] && X=(--images data/extra/teste/imagens --references data/extra/teste/referencias)
  O=outputs/baselines/curve/${m}_$s; mkdir -p $O/greedy
  $P -c "
import json,sys,numpy as np; from pathlib import Path
d=Path(sys.argv[1]); out={}
for f in sorted(d.glob('page_*.npz')):
    z=np.load(f); out[str(int(f.stem.split('_')[1]))]=[str(z[f'greedy_{i}']) for i in range(int(z['n_lines']))]
Path(sys.argv[2]).write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding='utf-8')" outputs/posteriors/curve_${m}_$s $O/greedy/lines.json
  $P tools/score_lines.py --lines $O/greedy/lines.json --config configs/curve_$m.json --no-rules ${X[@]+"${X[@]}"} --out $O/greedy/score
  OMP_NUM_THREADS=1 $P tools/postcorr_from_posteriors.py --config configs/curve_$m.json --posteriors outputs/posteriors/curve_${m}_$s --pages all --out $O/v6
  $P tools/score_lines.py --lines $O/v6/lines.json --config configs/curve_$m.json --no-rules ${X[@]+"${X[@]}"} --out $O/v6/score
done
echo "CURVE DONE $m"
