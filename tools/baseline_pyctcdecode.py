"""Baseline A: CTC prefix beam search with a KenLM word n-gram (pyctcdecode), on the exported st_c1m posteriors.

This is the standard LM-fused CTC decoding used in HTR toolkits. Our recognizer writes case, punctuation and
combining accents, while the LM holds lowercased letter-only words (the tokenization of our own n-gram model), so
each beam word is normalized before LM lookup: NFC, lowercased, split into letter runs. A token with no letters
(numbers, punctuation) gets no LM score and no OOV penalty. Everything else is pyctcdecode as released.

Runs in its own environment (numpy, pyctcdecode==0.5.0, kenlm); it does not import ocr_pipeline. Output: one
lines.json per (alpha, beta) for tools/score_lines.py.

  python tools/baseline_pyctcdecode.py --posteriors outputs/posteriors/v6_gt12 --pages 5,6,8,9,11,13,14,15,16 \
      --lm outputs/lm/kn3_w1.bin --corpus outputs/lm/period_w1.txt --alpha 0.5 1.0 --beta 0 2 --out outputs/baselines/pyctc
"""
import argparse
import itertools
import json
import logging
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import kenlm
import numpy as np
from pyctcdecode.alphabet import Alphabet
from pyctcdecode.decoder import BeamSearchDecoderCTC
from pyctcdecode.language_model import LanguageModel

logger = logging.getLogger(__name__)
FLOOR = 1e-6  # as ocr_pipeline.ctc_rescore
_WORD = re.compile(r'[^\W\d_]+')  # as tools/build_period_lexicon.py


def lm_words(token: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize('NFC', token).lower())


class NormalizedLM(LanguageModel):
    """pyctcdecode's KenLM wrapper with beam words normalized to the LM's tokenization."""

    def score_partial_token(self, partial_token: str) -> float:
        words = lm_words(partial_token)
        return super().score_partial_token(words[-1]) if words else 0.0

    def score(self, prev_state: 'kenlm.State', word: str, is_last_word: bool = False) -> tuple[float, 'kenlm.State']:
        words = lm_words(word)
        if not words:
            return self.beta, prev_state
        total, state = 0.0, prev_state
        for k, w in enumerate(words):
            value, state = super().score(state, w, is_last_word and k == len(words) - 1)
            total += value
        return total, state


def load_page(path: Path, n_classes: int) -> list[np.ndarray]:
    data = np.load(path)
    lines = []
    for i in range(int(data['n_lines'])):
        idx, val = data[f'idx_{i}'].astype(np.int64), data[f'val_{i}'].astype(np.float32)
        dense = np.full((idx.shape[0], n_classes), FLOOR, np.float32)
        np.put_along_axis(dense, idx, np.maximum(val, FLOOR), axis=1)
        lines.append(np.log(dense / dense.sum(1, keepdims=True)))
    return lines


_DECODER = None


def _init(labels: list[str], lm_path: str, unigrams: list[str], alpha: float, beta: float, unk: float) -> None:
    global _DECODER
    lm = NormalizedLM(kenlm.Model(lm_path), unigrams, alpha=alpha, beta=beta, unk_score_offset=unk)
    _DECODER = BeamSearchDecoderCTC(Alphabet.build_alphabet(labels), lm)  # what build_ctcdecoder does, our LM


def _decode(args: tuple[str, int, int]) -> tuple[str, list[str]]:
    path, n_classes, beam = args
    return Path(path).stem, [_DECODER.decode(line, beam_width=beam) for line in load_page(Path(path), n_classes)]


def read_unigrams(corpus: Path) -> list[str]:
    vocab: set[str] = set()
    with corpus.open(encoding='utf-8') as f:
        for line in f:
            vocab.update(line.split())
    return sorted(vocab)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--posteriors', type=Path, required=True)
    parser.add_argument('--pages', default='all', help="'all' or a comma list of page numbers")
    parser.add_argument('--lm', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True, help='LM training text, for the unigram list')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5])
    parser.add_argument('--beta', type=float, nargs='+', default=[1.0])
    parser.add_argument('--unk', type=float, default=-10.0, help='pyctcdecode OOV log-score offset (default)')
    parser.add_argument('--beam', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    alphabet = json.loads((args.posteriors / 'alphabet.json').read_text())
    n_classes = alphabet['n_classes']
    labels = [''] + [alphabet['l2c'].get(str(k), f'{k}') for k in range(1, n_classes)]
    paths = sorted(args.posteriors.glob('page_*.npz'))
    if args.pages != 'all':
        keep = {int(p) for p in args.pages.split(',')}
        paths = [p for p in paths if int(p.stem.split('_')[1]) in keep]
    unigrams = read_unigrams(args.corpus)
    logger.info('%d pages, %d LM unigrams, %d labels', len(paths), len(unigrams), len(labels))
    for alpha, beta in itertools.product(args.alpha, args.beta):
        with ProcessPoolExecutor(args.workers, initializer=_init,
                                 initargs=(labels, str(args.lm), unigrams, alpha, beta, args.unk)) as pool:
            pages = dict(pool.map(_decode, [(str(p), n_classes, args.beam) for p in paths]))
        out = args.out / f'a{alpha:g}_b{beta:g}'
        out.mkdir(parents=True, exist_ok=True)
        lines = {str(int(stem.split('_')[1])): text for stem, text in pages.items()}
        (out / 'lines.json').write_text(json.dumps(lines, ensure_ascii=False))
        (out / 'params.json').write_text(json.dumps({'alpha': alpha, 'beta': beta, 'unk': args.unk, 'beam': args.beam,
                                                     'lm': str(args.lm)}))
        logger.info('alpha %g beta %g -> %s', alpha, beta, out)


if __name__ == '__main__':
    main()
