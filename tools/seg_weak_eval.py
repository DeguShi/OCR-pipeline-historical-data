"""E19: line segmentation of a book without PAGE XML, judged against its one-text-line-per-page-line transcript.

For every segmentation candidate of the E18/E19 search (adapt_preprocess.seg_candidates) and every eval page, the
page is segmented and read greedily (default recognition chain, no post-correction); the readings are aligned to
the transcript lines (lines/<page>.txt of tools/book_prepare.py) by a monotone DP over
  1:1  one segmented line <-> one transcript line        cost d(seg, ref)
  2:1  two segmented lines <-> one transcript line (split)   cost d(seg1 + seg2, ref) + 0.1
  1:2  one segmented line <-> two transcript lines (merged)  cost d(seg, ref1 + ref2) + 0.1
       (2:1 and 1:2 only where the joined string beats each part alone by more than 0.1)
  skip either side                                       cost 0.7
with d the normalized edit distance of lowercased, punctuation-free strings. A 1:1 pair with d <= 0.4 is a found
line. Reported per page and candidate: transcript lines, segmented lines, found, split, merged, missed transcript
lines, extra segmented lines, d bins of the 1:1 pairs, and the page's greedy WER/CER against references/.
Caveat: a line that is segmented well but read badly (d > 0.4) counts as missed; the d bins show how many.

  python tools/seg_weak_eval.py --book outputs/e19/book --out outputs/e19/seg_weak --cache outputs/e19/cache
"""
import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt_preprocess import apply_candidate, run_candidate, seg_candidates  # noqa: E402  (path set above)
from ocr_pipeline.config import load_config  # noqa: E402
from ocr_pipeline.data import Page  # noqa: E402
from ocr_pipeline.engine import Engine  # noqa: E402
from ocr_pipeline.metrics import score  # noqa: E402

logger = logging.getLogger(__name__)
SKIP, PAIR_EXTRA, FOUND = 0.7, 0.1, 0.4
BINS = (0.2, 0.4, 0.6)


def norm(text: str) -> str:
    text = unicodedata.normalize('NFC', text.lower())
    return ' '.join(re.sub(r'[^\w\s]', ' ', text).split())


def dist(a: str, b: str) -> float:
    return Levenshtein.normalized_distance(a, b) if (a or b) else 0.0


def align(seg: list[str], ref: list[str]) -> list[tuple[str, list[int], list[int], float]]:
    """Monotone alignment of segmented lines to transcript lines: [(kind, seg idx, ref idx, d)]."""
    s, r = [norm(x) for x in seg], [norm(x) for x in ref]
    n, m = len(s), len(r)
    inf = float('inf')
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[Any]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == inf:
                continue
            moves = [(1, 0, 'extra', SKIP), (0, 1, 'missed', SKIP)]
            if i < n and j < m:
                moves.append((1, 1, '1:1', dist(s[i], r[j])))
            if i + 1 < n and j < m:  # only when joining beats either part alone (junk lines stay extra)
                d = dist(f'{s[i]} {s[i + 1]}', r[j])
                if d + PAIR_EXTRA < min(dist(s[i], r[j]), dist(s[i + 1], r[j])):
                    moves.append((2, 1, 'split', d + PAIR_EXTRA))
            if i < n and j + 1 < m:
                d = dist(s[i], f'{r[j]} {r[j + 1]}')
                if d + PAIR_EXTRA < min(dist(s[i], r[j]), dist(s[i], r[j + 1])):
                    moves.append((1, 2, 'merged', d + PAIR_EXTRA))
            for di, dj, kind, step in moves:
                a, b = i + di, j + dj
                if a <= n and b <= m and c + step < cost[a][b]:
                    cost[a][b], back[a][b] = c + step, (i, j, kind, step)
    out, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj, kind, step = back[i][j]
        out.append((kind, list(range(pi, i)), list(range(pj, j)), step - (PAIR_EXTRA if kind in ('split', 'merged') else 0)))
        i, j = pi, pj
    return out[::-1]


def summarize(pairs: list[tuple[str, list[int], list[int], float]], n_seg: int, n_ref: int) -> dict[str, Any]:
    ones = [d for kind, _, _, d in pairs if kind == '1:1']
    count = lambda k: sum(kind == k for kind, *_ in pairs)  # noqa: E731
    return {'n_ref': n_ref, 'n_seg': n_seg, 'found': sum(d <= FOUND for d in ones),
            'split': sum(kind == 'split' and d <= FOUND for kind, _, _, d in pairs),
            'merged': sum(kind == 'merged' and d <= FOUND for kind, _, _, d in pairs),
            'missed': count('missed'), 'extra': count('extra'),
            'd_bins': [sum(lo < d <= hi for d in ones) for lo, hi in zip((-1.0, *BINS), (*BINS, 9.9))]}


def eval_page(engine: Engine, page: Page, lines_file: Path, config: Any) -> dict[str, Any]:
    seg, _, _ = run_candidate(engine, page, config)
    ref_lines = [l for l in lines_file.read_text(encoding='utf-8').splitlines() if l.strip()]
    row = summarize(align(seg, ref_lines), len(seg), len(ref_lines))
    s = score(page.reference_text(), '\n'.join(seg))
    row.update({k: s[k] for k in ('wer_norm', 'cer_norm', 'norm_we', 'norm_rw', 'norm_ce', 'norm_rc')})
    return row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--book', type=Path, required=True, help='output of tools/book_prepare.py')
    parser.add_argument('--config', type=Path, default=Path('configs/final_v6.json'))
    parser.add_argument('--pages', default='eval', help="'eval', 'sample' or a comma list of page numbers")
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('outputs/e19/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='smoke test: first N candidates and first page')
    args = parser.parse_args()
    config = load_config(args.config)
    engine = Engine(args.cache, args.device, args.seed)
    case = json.loads((args.book / 'manifest.json').read_text())['cases'][0]
    numbers = case[args.pages] if args.pages in ('eval', 'sample') else [int(n) for n in args.pages.split(',')]
    cands = seg_candidates(config.params.upscale_factor)
    if args.limit:
        cands, numbers = cands[:args.limit], numbers[:1]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'params.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    rows = []
    for cand in cands:
        cfg = apply_candidate(config, cand)
        for n in numbers:
            image = next((args.book / 'images').glob(f'{n}.*.jpg'))
            page = Page(n, image, args.book / 'references' / f'{image.stem}.txt')
            rows.append({'page': n, 'candidate': cand['name'],
                         **eval_page(engine, page, args.book / 'lines' / f'{image.stem}.txt', cfg)})
            logger.info('%s', rows[-1])
        (args.out / 'seg_weak.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    logger.info('SEG WEAK DONE %d rows', len(rows))


if __name__ == '__main__':
    main()
