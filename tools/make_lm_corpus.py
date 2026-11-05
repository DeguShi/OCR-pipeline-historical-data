"""Write the period corpus of our trigram model (configs/final_v6.json: Gutenberg 443 books + São Paulo laws x4 +
training-page text x8) as plain text for external LM toolkits (KenLM lmplz), one source line per line, lowercased
words only (the same tokenization as tools/build_period_lexicon.py). Books are taken from the manifest and checked by
SHA-256, so the external LM sees exactly the text our model was counted on.

  python tools/make_lm_corpus.py --manifest records/corpus/pt_hist_legal_w4_livro78.manifest.json \
      --sources <extracted sources.tgz>/sources --out outputs/lm/period_w4_livro78.txt
"""
import argparse
import hashlib
import json
import logging
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_period_lexicon import _WORD, body, book_text  # noqa: E402  (path set above)

logger = logging.getLogger(__name__)


def word_lines(text: str) -> Iterator[str]:
    for line in unicodedata.normalize('NFC', text).lower().splitlines():
        words = _WORD.findall(line)
        if words:
            yield ' '.join(words)


def book_sources(manifest: dict, sources: Path) -> list[str]:
    texts = []
    for book in manifest['books']:
        if not book['sha256']:
            continue
        path = sources / 'gutenberg' / f'pg{book["id"]}.txt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != book['sha256']:
            raise ValueError(f'{path}: SHA-256 differs from the manifest')
        texts.append(body(path.read_text(encoding='utf-8', errors='replace')))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True, help='folder with gutenberg/ and alesp/text/')
    parser.add_argument('--book-pages', type=Path, default=Path('data/extra/transcricoes/treino'))
    parser.add_argument('--law-weight', type=int, default=None, help='override the manifest weight')
    parser.add_argument('--book-weight', type=int, default=None, help='override the manifest weight')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    manifest = json.loads(args.manifest.read_text())
    rule = dict(manifest['rule'])
    rule['extra_weight'] = rule['extra_weight'] if args.law_weight is None else args.law_weight
    rule['book_weight'] = rule['book_weight'] if args.book_weight is None else args.book_weight
    books = book_sources(manifest, args.sources)
    laws = [p.read_text(encoding='utf-8', errors='replace') for p in sorted((args.sources / 'alesp' / 'text').glob('*'))]
    if len(laws) != manifest['extra_texts']['files']:
        raise ValueError(f'{len(laws)} law texts, manifest says {manifest["extra_texts"]["files"]}')
    book = book_text(args.book_pages)
    if hashlib.sha256(book.encode()).hexdigest() != manifest['book_texts']['sha256']:
        logger.warning('book text SHA-256 differs from the manifest (manifest hashes the first page file)')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokens = 0
    with args.out.open('w', encoding='utf-8') as out:
        for texts, weight in ((books, 1), (laws, rule['extra_weight']), ([book], rule['book_weight'])):
            for _ in range(weight):
                for text in texts:
                    for line in word_lines(text):
                        out.write(line + '\n')
                        tokens += line.count(' ') + 1
    logger.info('%d books, %d laws x%d, book x%d: %d tokens -> %s', len(books), len(laws), rule['extra_weight'],
                rule['book_weight'], tokens, args.out)


if __name__ == '__main__':
    main()
