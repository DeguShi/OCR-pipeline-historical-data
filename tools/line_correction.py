"""Per-line over/under-correction of each post-correction method (E17 reporting item: lines improved / unchanged /
worsened, and the change in error against the raw line's error).

Inputs are the frozen outputs in records/postcorr/method_outputs. The page reference is split into lines by aligning
it to the RAW recognizer's words (the common input of every method, so the split favours no corrector); see
qualitative_examples.reference_by_line. Each method's page text is then aligned to the whole page reference and its
words are attributed to the reference line they align to (hyp_by_ref_line): methods may merge or drop lines, so
pairing lines by index is wrong (it shifted whole pages in a first version). Lines whose reference segment has < 4 words
or differs from the raw line by > 2 words are dropped (alignment drift). Per line and method: word errors and
character errors (Levenshtein on the normalized text) against the reference segment.

  python tools/line_correction.py --out records/postcorr/lines
Outputs: OUT/lines.jsonl (one row per line: set, page, line, ref, n_ref_words, n_ref_chars, word and char errors per
method) and OUT/summary.json (per set and method: lines improved / unchanged / worsened vs raw, by word and by char
errors, plus the same split by raw-line CER bins).
"""
import argparse
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualitative_examples import METHODS, ROOT, SETS, method_lines, reference_by_line, word_errors, words  # noqa: E402

BINS = ((0.0, 0.0), (0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 10.0))  # raw-line CER bins; first = exactly 0


def char_errors(ref: str, hyp: str) -> int:
    return Levenshtein.distance(ref, ' '.join(words(hyp)))


def hyp_by_ref_line(hyp_page: list[str], ref: list[str], ref_line: list[int], n_lines: int) -> list[list[str]]:
    """Attribute a method's page words to reference lines through one word alignment of the whole page, so the
    method's own line structure (merged, split or dropped lines) does not shift lines. Substituted/equal words go
    to their reference word's line; inserted words to the line of the previous reference word."""
    hyp = [w for line in hyp_page for w in words(line)]
    vocab: dict[str, int] = {}
    a = [vocab.setdefault(w, len(vocab)) for w in hyp]
    b = [vocab.setdefault(w, len(vocab)) for w in ref]
    out: list[list[str]] = [[] for _ in range(n_lines)]
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(a, b).as_list():
        for k, i in enumerate(range(i1, i2)):
            j = min(j1 + k, j2 - 1) if j2 > j1 else j1 - 1
            out[ref_line[max(j, 0)] if ref_line else 0].append(hyp[i])
    return out


def collect(set_name: str) -> list[dict]:
    ref_dir, pages = SETS[set_name]
    texts = method_lines(set_name)
    rows = []
    for page in pages:
        ref_file = next(p for p in (ROOT / ref_dir).glob('*.txt') if p.name.split('.')[0] == str(page))
        segments = reference_by_line(ref_file.read_text(encoding='utf-8'), texts['raw'][page])
        ref = [w for seg in segments for w in seg]
        ref_line = [i for i, seg in enumerate(segments) for _ in seg]
        hyps = {m: hyp_by_ref_line(texts[m].get(page, []), ref, ref_line, len(segments)) for m in METHODS}
        for i, seg in enumerate(segments):
            raw_line = texts['raw'][page][i]
            if len(seg) < 4 or abs(len(seg) - len(words(raw_line))) > 2:
                continue
            ref_text = ' '.join(seg)
            line = {m: ' '.join(hyps[m][i]) for m in METHODS}
            rows.append({'set': set_name, 'page': page, 'line': i, 'ref': ref_text, 'n_ref_words': len(seg),
                         'n_ref_chars': len(ref_text), 'texts': line,
                         'words': {m: word_errors(seg, t) for m, t in line.items()},
                         'chars': {m: char_errors(ref_text, t) for m, t in line.items()}})
    return rows


def split(rows: list[dict], method: str, key: str) -> dict:
    d = [r[key][method] - r[key]['raw'] for r in rows]
    return {'improved': sum(x < 0 for x in d), 'unchanged': sum(x == 0 for x in d), 'worsened': sum(x > 0 for x in d),
            'lines': len(d)}


def in_bin(r: dict, lo: float, hi: float) -> bool:
    cer = r['chars']['raw'] / max(r['n_ref_chars'], 1)
    return cer == 0 if hi == 0 else lo < cer <= hi


def summarise(rows: list[dict]) -> dict:
    out = {}
    for m in METHODS[1:]:
        out[m] = {'by_words': split(rows, m, 'words'), 'by_chars': split(rows, m, 'chars'),
                  'by_raw_cer_bin': {f'{lo}-{hi}': split([r for r in rows if in_bin(r, lo, hi)], m, 'chars')
                                     for lo, hi in BINS}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'records/postcorr/lines')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with (args.out / 'lines.jsonl').open('w', encoding='utf-8') as f:
        for set_name in SETS:
            rows = collect(set_name)
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            summary[set_name] = summarise(rows)
            print(f'{set_name}: {len(rows)} usable lines')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    for set_name, s in summary.items():
        for m, v in s.items():
            w, c = v['by_words'], v['by_chars']
            print(f"{set_name:8s} {m:8s} words +{w['improved']}/={w['unchanged']}/-{w['worsened']}  "
                  f"chars +{c['improved']}/={c['unchanged']}/-{c['worsened']}")


if __name__ == '__main__':
    main()
