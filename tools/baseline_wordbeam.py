"""Baseline B: CTC Word Beam Search (Scheidl et al., ICFHR 2018) on the exported st_c1m posteriors.

The decoder constrains every word to the dictionary of a corpus and scores word sequences with the corpus bigram
(modes Words / NGrams / NGramsForecast). Three adaptations to our recognizer, nothing else changed:
- case: the corpus is lowercased (as our n-gram model), so each uppercase class's probability is added to its
  lowercase class and the output is lowercase. Only the normalized (lowercase) metric is meaningful for B.
- accents: the recognizer writes base letters plus combining accents (NFD), so the corpus is NFD-normalized and
  combining marks count as word characters. Output is NFC-normalized back.
- matrix layout: Word Beam Search wants the blank last; our blank is class 0.
Corpus words with a character the recognizer cannot write are dropped.

Runs in its own environment (numpy, word-beam-search); it does not import ocr_pipeline.

  python tools/baseline_wordbeam.py --posteriors outputs/posteriors/v6_gt12 --pages 5,6,8,9,11,13,14,15,16 \
      --corpus outputs/lm/period_w1.txt --mode Words NGrams --out outputs/baselines/wbs
"""
import argparse
import itertools
import json
import logging
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from word_beam_search import WordBeamSearch

logger = logging.getLogger(__name__)
FLOOR = 1e-6  # as ocr_pipeline.ctc_rescore


def class_map(labels: list[str]) -> tuple[str, np.ndarray]:
    """Kept characters and, for each recognizer class 1..n-1, its column in the reduced matrix (case merged)."""
    kept = [c for c in labels[1:] if not (c.lower() != c and c.lower() in labels)]
    column = {c: i for i, c in enumerate(kept)}
    target = np.array([column.get(c, column.get(c.lower(), -1)) for c in labels[1:]])
    if (target < 0).any():
        raise ValueError('unmapped class')
    return ''.join(kept), target


def word_chars(chars: str) -> str:
    return ''.join(c for c in chars if unicodedata.category(c)[0] in 'LM')


def nfd_corpus(corpus: Path, allowed: set[str]) -> str:
    out, dropped = [], 0
    with corpus.open(encoding='utf-8') as f:
        for line in f:
            words = unicodedata.normalize('NFD', line).split()
            keep = [w for w in words if set(w) <= allowed]
            dropped += len(words) - len(keep)
            if keep:
                out.append(' '.join(keep))
    logger.info('corpus: %d words dropped (characters the recognizer cannot write)', dropped)
    return '\n'.join(out)


def load_page(path: Path, n_classes: int, target: np.ndarray, n_chars: int) -> list[np.ndarray]:
    data = np.load(path)
    lines = []
    for i in range(int(data['n_lines'])):
        idx, val = data[f'idx_{i}'].astype(np.int64), data[f'val_{i}'].astype(np.float32)
        dense = np.full((idx.shape[0], n_classes), FLOOR, np.float32)
        np.put_along_axis(dense, idx, np.maximum(val, FLOOR), axis=1)
        dense /= dense.sum(1, keepdims=True)
        mat = np.zeros((dense.shape[0], n_chars + 1), np.float32)
        np.add.at(mat.T, target, dense[:, 1:].T)
        mat[:, n_chars] = dense[:, 0]
        lines.append(mat[:, None, :])
    return lines


_STATE: dict = {}


def _init(mode: str, smoothing: float, beam: int, corpus: str, chars: str, wchars: str,
          n_classes: int, target: np.ndarray) -> None:
    _STATE.update(wbs=WordBeamSearch(beam, mode, smoothing, corpus.encode('utf8'), chars.encode('utf8'),
                                     wchars.encode('utf8')), chars=chars, n_classes=n_classes, target=target)


def _decode(path: str) -> tuple[str, list[str]]:
    s = _STATE
    texts = []
    for mat in load_page(Path(path), s['n_classes'], s['target'], len(s['chars'])):
        labels = s['wbs'].compute(mat)[0]
        texts.append(unicodedata.normalize('NFC', ''.join(s['chars'][k] for k in labels)))
    return Path(path).stem, texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--posteriors', type=Path, required=True)
    parser.add_argument('--pages', default='all', help="'all' or a comma list of page numbers")
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--mode', nargs='+', default=['NGrams'], choices=['Words', 'NGrams', 'NGramsForecast'])
    parser.add_argument('--smoothing', type=float, nargs='+', default=[0.01])
    parser.add_argument('--beam', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    alphabet = json.loads((args.posteriors / 'alphabet.json').read_text())
    n_classes = alphabet['n_classes']
    labels = [''] + [alphabet['l2c'][str(k)] for k in range(1, n_classes)]
    chars, target = class_map(labels)
    wchars = word_chars(chars)
    corpus = nfd_corpus(args.corpus, set(wchars))
    paths = sorted(args.posteriors.glob('page_*.npz'))
    if args.pages != 'all':
        keep = {int(p) for p in args.pages.split(',')}
        paths = [p for p in paths if int(p.stem.split('_')[1]) in keep]
    logger.info('%d pages, %d chars (%d word chars)', len(paths), len(chars), len(wchars))
    for mode, smoothing in itertools.product(args.mode, args.smoothing):
        with ProcessPoolExecutor(args.workers, initializer=_init,
                                 initargs=(mode, smoothing, args.beam, corpus, chars, wchars, n_classes, target)) as pool:
            pages = dict(pool.map(_decode, [str(p) for p in paths]))
        out = args.out / f'{mode}_s{smoothing:g}'
        out.mkdir(parents=True, exist_ok=True)
        lines = {str(int(stem.split('_')[1])): text for stem, text in pages.items()}
        (out / 'lines.json').write_text(json.dumps(lines, ensure_ascii=False))
        (out / 'params.json').write_text(json.dumps({'mode': mode, 'smoothing': smoothing, 'beam': args.beam,
                                                     'corpus': str(args.corpus)}))
        logger.info('%s smoothing %g -> %s', mode, smoothing, out)


if __name__ == '__main__':
    main()
