"""E20: the recognizer's greedy (best-path) text from a posterior export (tools/export_posteriors.py), as the
lines.json that tools/score_lines.py reads. This is decoder P0, "as published". Also writes chars.json: the count
of every character in the greedy text, after NFC and in NFD form, so that the character mapping of rule 5 of the
E20 plan can be checked against a recognizer's real output.

  python tools/greedy_lines.py --posteriors outputs/e20/post/r2_traprinq --pages 6 --out outputs/e20/p0/r2_traprinq
"""
import argparse
import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def page_greedy(path: Path) -> list[str]:
    data = np.load(path)
    return [str(data[f'greedy_{i}']) for i in range(int(data['n_lines']))]


def pages_of(posteriors: Path, pages: str) -> list[Path]:
    files = sorted(posteriors.glob('page_*.npz'))
    if pages == 'all':
        return files
    wanted = {int(p) for p in pages.split(',')}
    return [f for f in files if int(f.stem.split('_')[1]) in wanted]


def char_counts(lines: dict[str, list[str]], form: str) -> dict[str, int]:
    counts = Counter(c for page in lines.values() for line in page for c in unicodedata.normalize(form, line))
    return dict(counts.most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--posteriors', type=Path, required=True)
    parser.add_argument('--pages', default='all', help="'all' or a comma list of page numbers")
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    files = pages_of(args.posteriors, args.pages)
    if not files:
        raise SystemExit(f'no page_*.npz for pages {args.pages} in {args.posteriors}')
    lines = {str(int(f.stem.split('_')[1])): page_greedy(f) for f in files}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'lines.json').write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding='utf-8')
    chars = {'nfc': char_counts(lines, 'NFC'), 'nfd': char_counts(lines, 'NFD')}
    (args.out / 'chars.json').write_text(json.dumps(chars, ensure_ascii=False, indent=1), encoding='utf-8')
    logger.info('%d pages, %d lines', len(lines), sum(len(v) for v in lines.values()))


if __name__ == '__main__':
    main()
