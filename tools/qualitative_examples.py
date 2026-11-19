"""Qualitative examples for the paper: line image + reference + the output of each post-correction method.

Inputs are the frozen outputs in records/postcorr/method_outputs (page text, one line per segmented line) and the
line crops of tools/export_line_crops.py (same segmentation, outputs/line_crops/<set>/page_XX/line_NNN.png).
References are page-level, so each page's reference words are aligned to v6's words (one Levenshtein path over the
normalized words) and every reference word inherits the line of the v6 word it is aligned to (insertions go to the
previous v6 word's line). Word errors per line are counted against that reference segment under the `norm` policy.

Lines are picked automatically, per category (`CATEGORIES`, criteria in each docstring), at most --per-category
each, preferring lines with more words; ties broken by set, page, line. Outputs:
  OUT/examples.json                      every picked line with all texts and per-method word errors
  OUT/crops/<set>_p<page>_l<line>.png    the picked line images
  OUT/examples.tex                       a pt-BR LaTeX fragment (one table per line; wrong words in \\erro{...})

  python tools/qualitative_examples.py --crops outputs/line_crops --out records/postcorr/examples
"""
import argparse
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_pipeline.metrics import normalize  # noqa: E402  (path set above)

ROOT = Path(__file__).resolve().parents[1]
OUTS = ROOT / 'records/postcorr/method_outputs'
METHODS = ('raw', 'v6', 'A_pyctc', 'B_wbs', 'C_byt5', 'D_mlm', 'E_vlm')
LABEL = {'raw': 'sem pós-correção', 'v6': '\\textbf{v6 (nossa)}', 'A_pyctc': 'A: KenLM', 'B_wbs': 'B: \\textit{word beam}',
         'C_byt5': 'C: ByT5', 'D_mlm': 'D: v6 + PortOldBERT', 'E_vlm': 'E: VLM'}
SETS = {'test10': ('data/extra/teste/referencias', [18, 22, 23, 24, 25, 31, 32, 34, 44, 55]),
        'heldout': ('data/ground_truth', [7, 10, 12])}
Line = dict  # {'set', 'page', 'line', 'ref', 'texts': {method: str}, 'errors': {method: int}}


def words(text: str) -> list[str]:
    return normalize(text, strict=False).split()


def method_lines(set_name: str) -> dict[str, dict[int, list[str]]]:
    """{method: {page: [line text, ...]}} for one evaluation set."""
    suffix = {'test10': 'test10', 'heldout': 'gt12'}[set_name]
    raw = json.loads((OUTS / f'raw_greedy_{suffix}_lines.json').read_text(encoding='utf-8'))
    out = {'raw': {int(p): lines for p, lines in raw.items()}}
    for m in METHODS[1:]:
        name = f'E_vlm_{set_name}' if m == 'E_vlm' else f'{m}_{suffix}'
        pages = json.loads((OUTS / f'{name}.json').read_text(encoding='utf-8'))['pages']
        out[m] = {r['page']: r['text'].split('\n') for r in pages}
    return out


def reference_by_line(reference: str, v6_lines: list[str]) -> list[list[str]]:
    """Split the page reference into one word list per v6 line, through a word alignment."""
    v6 = [(w, i) for i, line in enumerate(v6_lines) for w in words(line)]
    ref = words(reference)
    vocab: dict[str, int] = {}
    a = [vocab.setdefault(w, len(vocab)) for w, _ in v6]
    b = [vocab.setdefault(w, len(vocab)) for w in ref]
    out: list[list[str]] = [[] for _ in v6_lines]
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(a, b).as_list():
        for k, j in enumerate(range(j1, j2)):
            i = min(i1 + k, i2 - 1) if i2 > i1 else i1 - 1
            out[v6[max(i, 0)][1] if v6 else 0].append(ref[j])
    return out


def word_errors(ref: list[str], hyp: str) -> int:
    vocab: dict[str, int] = {}
    return Levenshtein.distance([vocab.setdefault(w, len(vocab)) for w in ref],
                                [vocab.setdefault(w, len(vocab)) for w in words(hyp)])


def collect(set_name: str) -> list[Line]:
    ref_dir, pages = SETS[set_name]
    texts = method_lines(set_name)
    rows = []
    for page in pages:
        ref_file = next(p for p in (ROOT / ref_dir).glob('*.txt') if p.name.split('.')[0] == str(page))
        segments = reference_by_line(ref_file.read_text(encoding='utf-8'), texts['v6'][page])
        for i, seg in enumerate(segments):
            line = {m: texts[m][page][i] if i < len(texts[m].get(page, [])) else '' for m in METHODS}
            rows.append({'set': set_name, 'page': page, 'line': i, 'ref': ' '.join(seg), 'texts': line,
                         'errors': {m: word_errors(seg, t) for m, t in line.items()}})
    return rows


def sane(row: Line) -> bool:
    """Reference segment long enough, and about as long as v6's line (guards against alignment drift)."""
    n_ref, n_v6 = len(row['ref'].split()), len(words(row['texts']['v6']))
    return n_ref >= 4 and abs(n_ref - n_v6) <= 2


def v6_fixes_others_miss(r: Line) -> bool:
    """The recognizer erred, v6 gets the line right, and at least two of A, B, C still err."""
    e = r['errors']
    return e['raw'] > 0 and e['v6'] == 0 and sum(e[m] > 0 for m in ('A_pyctc', 'B_wbs', 'C_byt5')) >= 2


def vlm_modernizes(r: Line) -> bool:
    """The recognizer already read the line right; v6 keeps it; the VLM introduces errors."""
    e = r['errors']
    return e['raw'] == 0 and e['v6'] == 0 and e['E_vlm'] > 0


def byt5_better(r: Line) -> bool:
    """ByT5 beats v6 on the line (the numeral / split-merge cases v6 cannot reach)."""
    return r['errors']['C_byt5'] < r['errors']['v6']


def v6_breaks(r: Line) -> bool:
    """v6 makes the line worse than the raw recognizer output (a v6 failure case)."""
    return r['errors']['v6'] > r['errors']['raw']


CATEGORIES: dict[str, Callable[[Line], bool]] = {f.__name__: f for f in
                                                 (v6_fixes_others_miss, vlm_modernizes, byt5_better, v6_breaks)}


def pick(rows: list[Line], per_category: int) -> dict[str, list[Line]]:
    usable = [r for r in rows if sane(r)]
    order = sorted(usable, key=lambda r: (-len(r['ref'].split()), r['set'], r['page'], r['line']))
    return {name: [r for r in order if test(r)][:per_category] for name, test in CATEGORIES.items()}


def tex_escape(s: str) -> str:
    table = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
             '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(table.get(c, c) for c in s)


def marked(text: str, ref: list[str]) -> str:
    """Original tokens of `text`, those not matched to the reference wrapped in \\erro{...}."""
    tokens = text.split()
    norm = [' '.join(words(t)) for t in tokens]
    keep = [i for i, n in enumerate(norm) if n]
    vocab: dict[str, int] = {}
    a = [vocab.setdefault(norm[i], len(vocab)) for i in keep]
    b = [vocab.setdefault(w, len(vocab)) for w in ref]
    bad = set()
    for tag, i1, i2, _, _ in Levenshtein.opcodes(a, b).as_list():
        if tag != 'equal':
            bad.update(keep[i] for i in range(i1, i2))
    return ' '.join(f'\\erro{{{tex_escape(t)}}}' if i in bad else tex_escape(t) for i, t in enumerate(tokens))


def tex_block(r: Line, crop: str) -> str:
    ref = r['ref'].split()
    rows = ''.join(f"{LABEL[m]} & {marked(r['texts'][m], ref) or '--'} & {r['errors'][m]} \\\\\n" for m in METHODS)
    where = {'test10': 'teste', 'heldout': 'reservadas'}[r['set']]
    return (f"\\noindent\\includegraphics[width=\\linewidth]{{{crop}}}\\\\[2pt]\n"
            f"{{\\footnotesize\\begin{{tabular}}{{@{{}}lp{{0.68\\linewidth}}r@{{}}}}\n\\toprule\n"
            f"referência (normalizada) & {tex_escape(r['ref'])} & erros \\\\\n\\midrule\n{rows}\\bottomrule\n"
            f"\\end{{tabular}}}}\\\\[-2pt]\n{{\\scriptsize página {r['page']} ({where}), linha {r['line'] + 1}}}\n\n")


def write(picked: dict[str, list[Line]], crops: Path, out: Path) -> None:
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    tex = ['% Gerado por tools/qualitative_examples.py; \\erro marca palavra que difere da referência.\n'
           '\\providecommand{\\erro}[1]{\\underline{#1}}\n\n']
    for name, rows in picked.items():
        tex.append(f'% --- {name}: {CATEGORIES[name].__doc__}\n')
        for r in rows:
            src = crops / {'test10': 'test10', 'heldout': 'gt12'}[r['set']] / f"page_{r['page']:02d}" / f"line_{r['line']:03d}.png"
            dst = f"crops/{r['set']}_p{r['page']:02d}_l{r['line']:03d}.png"
            shutil.copy(src, out / dst)
            r['crop'] = dst
            tex.append(tex_block(r, dst))
    (out / 'examples.json').write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'examples.tex').write_text(''.join(tex), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--crops', type=Path, default=ROOT / 'outputs/line_crops')
    parser.add_argument('--out', type=Path, default=ROOT / 'records/postcorr/examples')
    parser.add_argument('--per-category', type=int, default=3)
    args = parser.parse_args()
    rows = collect('test10') + collect('heldout')
    picked = pick(rows, args.per_category)
    write(picked, args.crops, args.out)
    counts = {name: sum(test(r) for r in rows if sane(r)) for name, test in CATEGORIES.items()}
    print(f'{len(rows)} lines, {sum(map(sane, rows))} usable; lines per category: {counts}')


if __name__ == '__main__':
    main()
