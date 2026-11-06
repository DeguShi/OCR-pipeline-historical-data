"""Paired comparison of two runs on reference-word blocks, for page sets too small for a page bootstrap.

Each run's page text is aligned to the reference (normalized words, one optimal Levenshtein path) and every edit is
charged to the block of `--block` consecutive reference words where it falls (an insertion to the block of the
reference position it precedes). Blocks are the same for every run, so the comparison is paired, and the block
errors of a page sum exactly to its word edit count. Reported per page set:
- WER of both runs and the difference;
- block bootstrap CI95 of the WER difference (resampling blocks; blocks of one page are correlated, so this CI is
  somewhat narrower than a page-cluster one);
- sign-flip permutation p-value on block differences;
- pages won / tied / lost.

  python tools/block_bootstrap.py --a outputs/runs/v6_all12_cache --b outputs/runs/base_A_pyctc_gt12 \
      --pages 7,10,12 --references data/ground_truth --images data/images
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_pipeline.data import discover  # noqa: E402  (path set above)
from ocr_pipeline.metrics import normalize  # noqa: E402

logger = logging.getLogger(__name__)


def block_errors(reference: str, hypothesis: str, block: int) -> np.ndarray:
    ref, hyp = normalize(reference, False).split(), normalize(hypothesis, False).split()
    vocab: dict[str, int] = {}
    ref_ids = [vocab.setdefault(w, len(vocab)) for w in ref]
    hyp_ids = [vocab.setdefault(w, len(vocab)) for w in hyp]
    n_blocks = max(1, -(-len(ref) // block))
    errors = np.zeros(n_blocks, np.int64)
    for op in Levenshtein.editops(ref_ids, hyp_ids):
        errors[min(op.src_pos, len(ref) - 1) // block] += 1
    return errors


def block_lengths(reference: str, block: int) -> np.ndarray:
    n = len(normalize(reference, False).split())
    return np.array([min(block, n - i) for i in range(0, n, block)], np.int64)


def page_texts(run: Path) -> dict[int, str]:
    return {p['page']: p['text'] for p in json.loads((run / 'result.json').read_text())['pages']}


def compare(ea: np.ndarray, eb: np.ndarray, length: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    diff = ea - eb
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    boot = 100 * diff[idx].sum(1) / length[idx].sum(1)
    flips = rng.choice([-1, 1], (n_boot, len(diff)))
    observed = abs(diff.sum())
    p = float(((np.abs((flips * diff).sum(1)) >= observed).sum() + 1) / (n_boot + 1))
    return {'wer_a': 100 * ea.sum() / length.sum(), 'wer_b': 100 * eb.sum() / length.sum(),
            'dwer': 100 * diff.sum() / length.sum(), 'ci95': np.percentile(boot, [2.5, 97.5]).round(2).tolist(),
            'p_signflip': p, 'blocks': int(len(diff))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', type=Path, required=True, help='run A (result.json folder)')
    parser.add_argument('--b', type=Path, required=True, help='run B')
    parser.add_argument('--pages', required=True, help='comma list of page numbers')
    parser.add_argument('--images', type=Path, default=Path('data/images'))
    parser.add_argument('--references', type=Path, default=Path('data/ground_truth'))
    parser.add_argument('--block', type=int, default=10)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pages = {p.number: p for p in discover(args.images, args.references)}
    text_a, text_b = page_texts(args.a), page_texts(args.b)
    ea, eb, ln, wins = [], [], [], [0, 0, 0]
    for n in (int(x) for x in args.pages.split(',')):
        ref = pages[n].reference_text()
        a, b = block_errors(ref, text_a[n], args.block), block_errors(ref, text_b[n], args.block)
        ea.append(a), eb.append(b), ln.append(block_lengths(ref, args.block))
        wins[0 if a.sum() < b.sum() else 1 if a.sum() == b.sum() else 2] += 1
    result = compare(np.concatenate(ea), np.concatenate(eb), np.concatenate(ln), args.n_boot, args.seed)
    result.update(pages_won_tied_lost=wins, a=str(args.a), b=str(args.b), block=args.block)
    logger.info('WER A %.2f  B %.2f  dWER %+.2f  CI95 %s  p=%.4f  blocks %d  pages A won/tied/lost %s',
                result['wer_a'], result['wer_b'], result['dwer'], result['ci95'], result['p_signflip'],
                result['blocks'], wins)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
