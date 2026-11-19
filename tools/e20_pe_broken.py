"""E20: words right in R1 greedy (P0) and wrong after PE, on test10 (Gemini ref); classify the real-word breaks.

Same alignment and categories as tools/error_types.py. Classes of a broken real word (ref -> PE):
accent only = equal after removing diacritics; modernized = equal after removing diacritics and simplifying
ph/th/y/double consonants/ct/pt/gm/mn/initial h (rule of records/postcorr/vlm_broken_words_tuning.json); other.
Run from the repo root with sys.argv = [script, out_json].
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, 'tools')
sys.path.insert(0, '.')
from error_types import aligned, category  # noqa: E402
from block_bootstrap import page_texts  # noqa: E402
from ocr_pipeline.data import discover  # noqa: E402

E = Path('outputs/e20_colab_e20ev_ev/outputs/e20/eval/score/test10/r1_st_c1m')


def strip(w: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', w) if not unicodedata.combining(c))


def modern(w: str) -> str:
    w = strip(w)
    for a, b in (('ph', 'f'), ('th', 't'), ('y', 'i'), ('ct', 't'), ('pt', 't'), ('gm', 'm'), ('mn', 'n')):
        w = w.replace(a, b)
    w = re.sub(r'([bcdfglmnprst])\1', r'\1', w)
    return w[1:] if w.startswith('h') else w


lexicon = set(Path('models/lexicon/pt_hist_legal_w1.txt').read_text(encoding='utf-8').split())
pages = {p.number: p for p in discover(Path('data/extra/teste/imagens'), Path('data/extra/teste/referencias'))}
raw, pe = page_texts(E / 'p0' / 'gemini'), page_texts(E / 'pe' / 'gemini')
cls, examples = Counter(), {}
for n in sorted(pages):
    ref = pages[n].reference_text()
    for (w, hr, ur), (_, hp, up) in zip(aligned(ref, raw[n]), aligned(ref, pe[n])):
        if hr == w and not ur and not (hp == w and not up) and category(w, hp, up, lexicon) == 'real_word':
            c = 'accent only' if strip(w) == strip(hp) else 'modernized' if modern(w) == modern(hp) else 'other'
            cls[c] += 1
            examples.setdefault(c, []).append(f'{w}->{hp}')
out = {'counts': dict(cls), 'total': sum(cls.values()), 'examples': {k: v[:15] for k, v in examples.items()}}
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False))
