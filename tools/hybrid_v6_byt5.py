"""v6 + C hybrid: keep v6's text and adopt the ByT5 corrector's version only for the error kinds v6 cannot fix by
construction (one candidate word per recognized word): numerals and word-boundary (split/merge) errors.

Both scored texts (result.json 'text', i.e. after the configuration's text rules) are aligned word by word per page
(one optimal Levenshtein path over word tokens). Each differing block (v6 words V, ByT5 words B) is adopted from
ByT5 when the variant's rule fires:
- split_strict: len(V) != len(B) and ''.join(V) == ''.join(B) (ByT5 only moves word boundaries);
- split_loose: len(V) != len(B), both sides non-empty, and the joined strings have normalized similarity >= 0.8;
- numeral: a token on either side contains a digit;
- numeral_split: numeral or split_loose.
The variant is chosen on the tuning pages only; held-out and test are run once with it. Adopted words go to the line
of the first v6 word of the block (or the previous v6 word for insertions). Output lines.json is scored with
tools/score_lines.py --no-rules.

  python tools/hybrid_v6_byt5.py --v6 outputs/repro/gt12/score/result.json \
      --byt5 outputs/baselines/byt5/gt12/score/result.json --rule split_strict --out outputs/baselines/hybrid/gt12_split_strict
"""
import argparse
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

RULES = ('split_strict', 'split_loose', 'numeral', 'numeral_split')


def adopt(rule: str, v: list[str], b: list[str]) -> bool:
    joined_v, joined_b = ''.join(v), ''.join(b)
    split_strict = len(v) != len(b) and joined_v == joined_b
    split_loose = len(v) != len(b) and bool(v) and bool(b) and \
        Levenshtein.normalized_similarity(joined_v, joined_b) >= 0.8
    numeral = any(c.isdigit() for w in v + b for c in w)
    return {'split_strict': split_strict, 'split_loose': split_loose, 'numeral': numeral,
            'numeral_split': numeral or split_loose}[rule]


def tokens(text: str) -> list[tuple[str, int]]:
    return [(w, i) for i, line in enumerate(text.split('\n')) for w in line.split()]


def blocks(a: list[int], b: list[int]) -> list[tuple[str, int, int, int, int]]:
    """Levenshtein opcodes with adjacent non-equal ones merged (rapidfuzz splits a 1->2 word split into a replace
    and an insert, which would hide every boundary change)."""
    out: list[list] = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(a, b).as_list():
        if out and tag != 'equal' and out[-1][0] != 'equal':
            out[-1][2], out[-1][4] = i2, j2
        else:
            out.append([tag if tag == 'equal' else 'diff', i1, i2, j1, j2])
    return [tuple(o) for o in out]


def hybrid_page(v6_text: str, byt5_text: str, rule: str) -> tuple[list[str], int]:
    tv, tb = tokens(v6_text), tokens(byt5_text)
    vocab: dict[str, int] = {}
    ids_v = [vocab.setdefault(w, len(vocab)) for w, _ in tv]
    ids_b = [vocab.setdefault(w, len(vocab)) for w, _ in tb]
    out: list[tuple[str, int]] = []
    adopted = 0
    for tag, i1, i2, j1, j2 in blocks(ids_v, ids_b):
        v, b = [w for w, _ in tv[i1:i2]], [w for w, _ in tb[j1:j2]]
        if tag != 'equal' and adopt(rule, v, b):
            line = tv[i1][1] if i1 < i2 else (out[-1][1] if out else 0)
            out += [(w, line) for w in b]
            adopted += 1
        else:
            out += tv[i1:i2]
    n_lines = len(v6_text.split('\n'))
    lines = [' '.join(w for w, l in out if l == i) for i in range(n_lines)]
    return lines, adopted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--v6', type=Path, required=True)
    parser.add_argument('--byt5', type=Path, required=True)
    parser.add_argument('--rule', choices=RULES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    v6 = {r['page']: r['text'] for r in json.loads(args.v6.read_text(encoding='utf-8'))['pages']}
    by = {r['page']: r['text'] for r in json.loads(args.byt5.read_text(encoding='utf-8'))['pages']}
    lines, adopted = {}, {}
    for page in sorted(v6):
        lines[str(page)], adopted[str(page)] = hybrid_page(v6[page], by[page], args.rule)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'lines.json').write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding='utf-8')
    (args.out / 'params.json').write_text(json.dumps({'v6': str(args.v6), 'byt5': str(args.byt5), 'rule': args.rule,
                                                      'adopted_blocks': adopted}, indent=1), encoding='utf-8')
    print(f'{args.rule}: {sum(adopted.values())} blocks adopted')


if __name__ == '__main__':
    main()
