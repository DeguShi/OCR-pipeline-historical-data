"""Correction rate by error type: which raw-recognizer word errors each post-corrector fixes, per category.

Every reference word that the raw run gets wrong (normalized words, one optimal Levenshtein path, as in
tools/correction_analysis.py) gets one category, checked in this order:
- numeral: the reference word contains a digit;
- split_merge: the recognizer split it (two adjacent hypothesis words concatenate to it) or merged it with a
  neighbour (its aligned hypothesis word is it concatenated with the previous or next reference word);
- deleted: no hypothesis word is aligned to it;
- short: reference word of 3 letters or fewer;
- real_word: the raw word aligned to it is itself in the period lexicon (a valid but wrong word);
- non_word: the raw word is not in the lexicon.
Per run and category: raw errors, fixed (right in the run), fix rate. Words the raw run gets right and the run
breaks are reported per category of the run's error, by the same rules (the run's word against the reference).

  python tools/error_types.py --raw outputs/runs/check_greedy_raw_test10 --runs outputs/runs/v6_test10_cache \
      --labels v6 --pages 18,22,23,24,25,31,32,34,44,55 --images data/extra/teste/imagens \
      --references data/extra/teste/referencias --out records/postcorr/error_types_test.json
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from correction_analysis import page_texts  # noqa: E402  (path set above)
from ocr_pipeline.data import discover  # noqa: E402
from ocr_pipeline.metrics import normalize  # noqa: E402

logger = logging.getLogger(__name__)
CATS = ('numeral', 'split_merge', 'deleted', 'short', 'real_word', 'non_word')


def aligned(reference: str, hypothesis: str) -> list[tuple[str, str | None, bool]]:
    """Per reference word: (word, aligned hypothesis word or None, split or merged by the hypothesis)."""
    ref, hyp = normalize(reference, False).split(), normalize(hypothesis, False).split()
    vocab: dict[str, int] = {}
    r_ids, h_ids = [vocab.setdefault(w, len(vocab)) for w in ref], [vocab.setdefault(w, len(vocab)) for w in hyp]
    pos: list[int | None] = [None] * len(ref)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(r_ids, h_ids).as_list():
        if tag in ('equal', 'replace'):
            for k, i in enumerate(range(i1, i2)):
                pos[i] = j1 + k if j1 + k < j2 else None
    return [(w, hyp[pos[i]] if pos[i] is not None else None, boundary_error(ref, hyp, pos, i))
            for i, w in enumerate(ref)]


def boundary_error(ref: list[str], hyp: list[str], pos: list[int | None], i: int) -> bool:
    """ref[i] split into two adjacent hypothesis words, or merged with its previous or next reference word."""
    w, anchors = ref[i], [pos[k] for k in (i - 1, i, i + 1) if 0 <= k < len(ref) and pos[k] is not None]
    window = range(max(0, min(anchors, default=0) - 1), min(len(hyp), max(anchors, default=-1) + 2))
    if any(hyp[a] + hyp[a + 1] == w for a in window if a + 1 < len(hyp)):
        return True
    joined = {w + ref[i + 1]} if i + 1 < len(ref) else set()
    joined |= {ref[i - 1] + w} if i else set()
    return any(hyp[a] in joined for a in window)


def category(word: str, hyp: str | None, uneven: bool, lexicon: set[str]) -> str:
    if any(c.isdigit() for c in word):
        return 'numeral'
    if uneven:
        return 'split_merge'
    if hyp is None:
        return 'deleted'
    if len(word) <= 3:
        return 'short'
    return 'real_word' if hyp in lexicon else 'non_word'


def analyse(raw: dict, run: dict, refs: dict, pages: list[int], lexicon: set[str]) -> dict:
    errors, fixed, broken = Counter(), Counter(), Counter()
    for page in pages:
        for (w, h_raw, u_raw), (_, h_run, u_run) in zip(aligned(refs[page], raw[page]), aligned(refs[page], run[page])):
            ok_raw, ok_run = h_raw == w and not u_raw, h_run == w and not u_run
            if not ok_raw:
                cat = category(w, h_raw, u_raw, lexicon)
                errors[cat] += 1; fixed[cat] += ok_run
            elif not ok_run:
                broken[category(w, h_run, u_run, lexicon)] += 1
    return {c: {'raw_errors': errors[c], 'fixed': fixed[c], 'fix_rate': round(100 * fixed[c] / max(1, errors[c]), 1),
                'broken': broken[c]} for c in CATS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--pages', required=True)
    parser.add_argument('--images', type=Path, default=Path('data/images'))
    parser.add_argument('--references', type=Path, default=Path('data/ground_truth'))
    parser.add_argument('--lexicon', type=Path, default=Path('models/lexicon/pt_hist_legal_w1.txt'))
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    pages = [int(p) for p in args.pages.split(',')]
    refs = {p.number: p.reference.read_text(encoding='utf-8') for p in discover(args.images, args.references)}
    lexicon = {l.split()[0] for l in args.lexicon.read_text(encoding='utf-8').splitlines() if l.strip()}
    raw = page_texts(args.raw)
    report = {'raw': str(args.raw), 'pages': pages, 'lexicon': str(args.lexicon), 'runs': {}}
    for label, run in zip(args.labels, args.runs):
        report['runs'][label] = {'path': str(run), **analyse(raw, page_texts(run), refs, pages, lexicon)}
    for c in CATS:
        logger.info('%-11s raw %4d | %s', c, report['runs'][args.labels[0]][c]['raw_errors'],
                    '  '.join(f"{l} {report['runs'][l][c]['fixed']:>3}/{report['runs'][l][c]['broken']:<3}"
                              for l in args.labels))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
