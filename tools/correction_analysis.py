"""Over- and under-correction of post-correctors against the raw recognizer output, per reference word.

Every run (and the raw run) is aligned to the reference (normalized words, one optimal Levenshtein path). A
reference word is "right" in a run when the path matches it to an identical hypothesis word. For each run, per page
set, every reference word falls in one of four cells against the raw run:
- fixed: wrong in raw, right in the run (useful corrections);
- broken: right in raw, wrong in the run (over-correction);
- kept_wrong: wrong in both (under-correction);
- kept_right: right in both.
Insertions (hypothesis words matched to no reference word) are counted separately, for raw and run. Net word
errors removed = fixed - broken + (raw insertions - run insertions), which equals the WER difference times words.

  python tools/correction_analysis.py --raw outputs/runs/ladder_L0_raw --runs outputs/runs/v6_all12_cache \
      outputs/runs/base_A2_pyctc_gt12 --labels v6 A --pages 7,10,12 --out records/postcorr/correction_heldout.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_pipeline.data import discover  # noqa: E402  (path set above)
from ocr_pipeline.metrics import normalize  # noqa: E402

logger = logging.getLogger(__name__)
CELLS = ('fixed', 'broken', 'kept_wrong', 'kept_right')


def right_words(reference: str, hypothesis: str) -> tuple[list[bool], int]:
    """Per reference word, whether the optimal path matches it to an identical word; plus the insertion count."""
    ref, hyp = normalize(reference, False).split(), normalize(hypothesis, False).split()
    vocab: dict[str, int] = {}
    ref_ids = [vocab.setdefault(w, len(vocab)) for w in ref]
    hyp_ids = [vocab.setdefault(w, len(vocab)) for w in hyp]
    right = [False] * len(ref)
    insertions = 0
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref_ids, hyp_ids).as_list():
        if tag == 'equal':
            right[i1:i2] = [True] * (i2 - i1)
        elif tag == 'insert':
            insertions += j2 - j1
        elif tag == 'replace' and j2 - j1 > i2 - i1:
            insertions += (j2 - j1) - (i2 - i1)
    return right, insertions


def page_texts(run: Path) -> dict[int, str]:
    return {p['page']: p['text'] for p in json.loads((run / 'result.json').read_text())['pages']}


def analyse(raw: dict[int, str], run: dict[int, str], refs: dict[int, str], pages: list[int]) -> dict:
    cells = dict.fromkeys(CELLS, 0)
    ins_raw = ins_run = 0
    for page in pages:
        r_raw, i_raw = right_words(refs[page], raw[page])
        r_run, i_run = right_words(refs[page], run[page])
        ins_raw, ins_run = ins_raw + i_raw, ins_run + i_run
        for a, b in zip(r_raw, r_run):
            cells[('kept_right' if b else 'broken') if a else ('fixed' if b else 'kept_wrong')] += 1
    words = sum(cells.values())
    wrong_raw = cells['fixed'] + cells['kept_wrong']
    return {**cells, 'words': words, 'ins_raw': ins_raw, 'ins_run': ins_run,
            'fix_rate': round(100 * cells['fixed'] / max(1, wrong_raw), 2),
            'break_rate': round(100 * cells['broken'] / max(1, words - wrong_raw), 2),
            'fixed_per_broken': round(cells['fixed'] / max(1, cells['broken']), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, required=True, help='raw recognizer run (result.json folder)')
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--pages', required=True, help='comma list of page numbers')
    parser.add_argument('--images', type=Path, default=Path('data/images'))
    parser.add_argument('--references', type=Path, default=Path('data/ground_truth'))
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    pages = [int(p) for p in args.pages.split(',')]
    refs = {p.number: p.reference.read_text(encoding='utf-8') for p in discover(args.images, args.references)}
    raw = page_texts(args.raw)
    report = {'raw': str(args.raw), 'pages': pages, 'runs': {}}
    for label, run in zip(args.labels, args.runs):
        report['runs'][label] = {'path': str(run), **analyse(raw, page_texts(run), refs, pages)}
        logger.info('%s: %s', label, report['runs'][label])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
