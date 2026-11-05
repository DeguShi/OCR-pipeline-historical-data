"""Score line texts produced by an external decoder exactly like a pipeline run: the configuration's text rules
(conventions, ordinals, dehyphenation; no lexicon pass) are applied, then the same metrics. Writes a result.json
that tools/postcorr_report.py reads.

Input: a JSON file {"<page number>": ["line 1", "line 2", ...], ...}.

  python tools/score_lines.py --lines outputs/baselines/pyctc/lines.json --config configs/final_v6.json \
      --out outputs/runs/base_pyctc
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lines', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='source of the text rules')
    parser.add_argument('--images', type=Path, default=Path('data/images'))
    parser.add_argument('--references', type=Path, default=Path('data/ground_truth'))
    parser.add_argument('--no-rules', action='store_true', help='score the lines as given')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    post = replace(load_config(args.config).post, decoder='none')
    if args.no_rules:
        post = replace(post, conventions=False, ordinals=False, dehyphenate=False)
    pages = {p.number: p for p in discover(args.images, args.references)}
    rows = []
    for number, lines in sorted(json.loads(args.lines.read_text()).items(), key=lambda kv: int(kv[0])):
        page = pages[int(number)]
        text = '\n'.join(correct(lines, post))
        rows.append({'page': page.number, 'lines': len(lines), 'pred_words': len(text.split()), 'text': text,
                     **score(page.reference_text(), text)})
    args.out.mkdir(parents=True, exist_ok=True)
    result = {'source': str(args.lines), 'rules': not args.no_rules, 'corpus': corpus(rows), 'pages': rows}
    (args.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    c = result['corpus']
    logger.info('WER norm %.2f%% | CER norm %.2f%% -> %s', 100 * c['wer_norm'], 100 * c['cer_norm'], args.out)


if __name__ == '__main__':
    main()
