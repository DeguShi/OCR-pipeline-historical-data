"""E20 rule 5: map characters outside our transcription convention, the same way for every system.

The convention is the character set of the st_c1m training transcripts (records/training/st_c1m/labels_r1), so the
map is derived from training lines only. A character c outside that set is replaced by the characters of NFD(c)
that are inside it (ū -> u, ȩ -> e), or deleted when none are (a bare macron, ^, ⟦). The map is applied to decoded
text before scoring; for P6 that is after our post-correction (the outside characters are at most 0.5% of the
characters the published recognizers write on the tuning pages, records/e20/tune/chars_*.json).

Input: a lines.json ({"<page>": [line, ...]}) or a search results.jsonl (one configuration per line, with a
"lines" field of the same shape). The output has the same format plus a <out>.charmap.json report of every
character that was mapped and how often.

  python tools/charmap.py --input outputs/e20/tune/p0/r2_traprinq/lines.json --out outputs/e20/map/p0_r2/lines.json
"""
import argparse
import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)
LABELS = Path('records/training/st_c1m/labels_r1/lines.jsonl')


def alphabet(labels: Path) -> set[str]:
    chars: set[str] = {' ', '\n'}
    for line in labels.read_text(encoding='utf-8').splitlines():
        chars |= set(unicodedata.normalize('NFC', json.loads(line).get('transcript') or ''))
    return chars


def map_char(c: str, keep: set[str]) -> str:
    return c if c in keep else ''.join(d for d in unicodedata.normalize('NFD', c) if d in keep)


def map_lines(pages: dict[str, list[str]], keep: set[str], counts: Counter) -> dict[str, list[str]]:
    out = {}
    for page, lines in pages.items():
        mapped = []
        for line in lines:
            text = unicodedata.normalize('NFC', line)
            counts.update(c for c in text if c not in keep)
            mapped.append(unicodedata.normalize('NFC', ''.join(map_char(c, keep) for c in text)))
        out[page] = mapped
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='lines.json or results.jsonl')
    parser.add_argument('--labels', type=Path, default=LABELS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    keep, counts = alphabet(args.labels), Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.input.suffix == '.jsonl':
        rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
        for row in rows:
            row['lines'] = map_lines(row['lines'], keep, counts)
        args.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    else:
        pages = json.loads(args.input.read_text(encoding='utf-8'))
        args.out.write_text(json.dumps(map_lines(pages, keep, counts), ensure_ascii=False, indent=1), encoding='utf-8')
    report = {'input': str(args.input), 'alphabet_size': len(keep), 'mapped': sum(counts.values()),
              'map': {c: {'n': n, 'to': map_char(c, keep), 'name': unicodedata.name(c, '?')}
                      for c, n in counts.most_common()}}
    Path(f'{args.out}.charmap.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    logger.info('mapped %d characters (%d distinct) -> %s', report['mapped'], len(counts), args.out)


if __name__ == '__main__':
    main()
