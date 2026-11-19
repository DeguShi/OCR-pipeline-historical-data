"""Score every configuration of a baseline search (tools/baseline_search.py results.jsonl) exactly as
tools/score_lines.py scores one: the configuration's text rules, then the same metrics, micro-aggregated over the
pages present (the tuning pages). Writes one JSON line per configuration, sorted by WER, to --out.

Selection rule: the frozen configuration is the one with the lowest tuning WER (ties: lower CER, then lower id).

  python tools/score_search.py --results outputs/baselines/search/search_A_0of1/results.jsonl \
      --config configs/final_v6.json --out records/postcorr/search_A_scored.jsonl
"""
import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_pipeline.config import load_config  # noqa: E402  (path set above)
from ocr_pipeline.data import discover  # noqa: E402
from ocr_pipeline.metrics import corpus, score  # noqa: E402
from ocr_pipeline.postprocess import correct  # noqa: E402

logger = logging.getLogger(__name__)


def score_row(row: dict, pages: dict, post) -> dict:
    scored = []
    for number, lines in sorted(row['lines'].items(), key=lambda kv: int(kv[0])):
        page = pages[int(number)]
        scored.append({'page': page.number, **score(page.reference_text(), '\n'.join(correct(lines, post)))})
    c = corpus(scored)
    return {'id': row['id'], 'params': row['params'], 'pages': [s['page'] for s in scored],
            'wer': round(100 * c['wer_norm'], 3), 'cer': round(100 * c['cer_norm'], 3), 'seconds': row['seconds']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, nargs='+', required=True, help='one or more results.jsonl')
    parser.add_argument('--config', type=Path, required=True, help='source of the text rules')
    parser.add_argument('--images', type=Path, default=Path('data/images'))
    parser.add_argument('--references', type=Path, default=Path('data/ground_truth'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    post = replace(load_config(args.config).post, decoder='none')
    pages = {p.number: p for p in discover(args.images, args.references)}
    rows = {}
    for path in args.results:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row['id']] = row
    scored = sorted((score_row(r, pages, post) for r in rows.values()), key=lambda s: (s['wer'], s['cer'], s['id']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in scored), encoding='utf-8')
    for s in scored[:5]:
        logger.info('id %3d  WER %.2f  CER %.2f  %s', s['id'], s['wer'], s['cer'], json.dumps(s['params']))
    logger.info('%d configurations -> %s', len(scored), args.out)


if __name__ == '__main__':
    main()
