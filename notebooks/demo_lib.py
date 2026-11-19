"""Helpers for notebooks/demo_artigo.ipynb: load saved results, render word diffs, draw the figures.

Nothing here trains, searches or decodes. Every number is read from a committed record
(records/e20, records/x02, records/e19) or from notebooks/demo_data (copied by tools/build_demo_notebook.py).
"""
from __future__ import annotations

import difflib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E20 = ROOT / 'records/e20/eval'
X02 = ROOT / 'records/x02/run_20261119/report'
E19 = ROOT / 'records/e19'
DEMO = ROOT / 'notebooks/demo_data'

INK, MUTED, OURS, OTHER, GOOD, BAD = '#1f1f1f', '#8a8a8a', '#1b5e9e', '#b0b0b0', '#2e7d32', '#c62828'
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.titleweight': 'bold', 'axes.titlesize': 11})

REC = {'r0_medium': 'R0 medium publicado', 'r1_st_c1m': 'R1 st_c1m (nosso)', 'r2_traprinq': 'R2 TraPrInq',
       'r3_mccatmus': 'R3 McCATMuS', 'r4_traprinq': 'R4 TraPrInq + nossos rótulos',
       'r5_mccatmus': 'R5 McCATMuS + nossos rótulos'}
DEC = {'p0': 'P0 guloso', 'pa': 'PA KenLM', 'pb': 'PB word beam', 'pe': 'PE VLM corrige', 'p6': 'P6 (nossa)'}

# Systems shown page by page (book 1). Key = (recognizer, decoder) as stored in records/e20/eval/score.
SHOWCASE = [('r0_medium', 'p0raw', 'Publicado (medium, imagem crua, guloso)'),
            ('v0', 'v0', 'VLM lê a linha (Qwen2.5-VL 7B)'),
            ('r1_st_c1m', 'pe', 'Nosso reconhecedor + VLM corrige'),
            ('r4_traprinq', 'pa', 'TraPrInq adaptado + KenLM (melhor receita publicada)'),
            ('r1_st_c1m', 'pa', 'Nosso reconhecedor + KenLM'),
            ('r1_st_c1m', 'p6', 'v6: nosso reconhecedor + P6')]

# Version table of the paper (outputs/paper/main.tex, tab:progressao): tuning WER, held-out WER/CER vs official GT.
PROGRESSION = [('publicado', 'medium na imagem crua', 63.15, 64.57, 21.71),
               ('v1', 'imagens separadas, léxico moderno, reescore CTC', 46.95, 48.66, 17.79),
               ('v3', 'dicionário de época + n-gramas', 36.00, 36.85, 14.00),
               ('v4 méd.', 'ordinais, palavras curtas, corpus legislativo', 29.63, 28.50, 10.64),
               ('v4', 'reconhecedor por auto-treino', 24.77, 22.05, 8.16),
               ('v5', 'rótulos do Gemini, texto do livro no contexto', 19.98, 18.43, 4.13),
               ('v6', 'mesma configuração, cadeia limpa', 19.64, 18.74, 4.43)]


def pt(x: float, nd: int = 2, sign: bool = False) -> str:
    """Number with a decimal comma (the notebook is in Portuguese)."""
    return f'{x:{"+" if sign else ""}.{nd}f}'.replace('.', ',')


def load_json(path: Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def grid() -> dict:
    """E20 grid plus the X02 rows (PA with 181 configurations)."""
    g = dict(load_json(E20 / 'grid.json'))
    g.update(load_json(X02 / 'grid_pa181.json'))
    return g


def paired() -> list[dict]:
    return list(load_json(E20 / 'paired.json')) + list(load_json(X02 / 'paired.json'))


# ---------------------------------------------------------------- page texts

def norm_words(text: str) -> list[str]:
    """Same normalisation as ocr_pipeline.metrics.normalize(strict=False), plus line-end hyphen joining."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[-=¬]\s*\n\s*', '', text)
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


@dataclass(frozen=True)
class PageRun:
    label: str
    text: str
    wer: float  # normalised page WER, % (from the saved score)


def e20_page(set_: str, rec: str, dec: str, ref: str, page: int) -> PageRun | None:
    path = E20 / 'score' / set_ / rec / dec / ref / 'result.json'
    if not path.exists():
        return None
    for p in load_json(path)['pages']:
        if p['page'] == page:
            return PageRun('', p['text'], 100 * p['wer_norm'])
    return None


def reference(set_: str, ref: str, page: int) -> str:
    if set_ == 'test10':
        return (ROOT / f'data/extra/teste/referencias/{page}.txt').read_text(encoding='utf-8')
    folder = ROOT / ('data/ground_truth' if ref == 'oficial' else 'data/ground_truth_gemini')
    return next(folder.glob(f'{page}.*.txt')).read_text(encoding='utf-8')


def page_image(set_: str, page: int) -> Path:
    if set_ == 'test10':
        return ROOT / f'data/extra/teste/imagens/{page}.jpg'
    if set_ == 'atas':
        return DEMO / f'livro_atas/{page}.jpg'
    return next((ROOT / 'data/images').glob(f'{page}.*.jpg'))


def median_pages(per_page: dict[int, float], k: int = 2) -> list[int]:
    """The k pages whose WER is closest to the median: a fixed rule, so examples are not cherry-picked."""
    med = float(np.median(list(per_page.values())))
    return sorted(sorted(per_page, key=lambda p: (abs(per_page[p] - med), p))[:k])


# ---------------------------------------------------------------- word diff

def diff_html(ref: str, hyp: str, max_words: int = 70) -> str:
    """Hypothesis words in ink when right; wrong words struck in red with the reference word in green."""
    r, h = norm_words(ref), norm_words(hyp)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if tag == 'equal':
            out += [html.escape(w) for w in h[j1:j2]]
            continue
        wrong = ' '.join(h[j1:j2])
        right = ' '.join(r[i1:i2])
        cell = ''
        if wrong:
            cell += f'<s style="color:{BAD}">{html.escape(wrong)}</s>'
        if right:
            cell += f'<sup style="color:{GOOD}">{html.escape(right)}</sup>'
        out.append(cell)
        if len(out) >= max_words:
            break
    return ' '.join(out[:max_words]) + ' …'


def runs_table_html(runs: list[PageRun], ref: str, max_words: int = 70) -> str:
    rows = []
    for run in runs:
        bold = 'font-weight:bold;' if run.label.startswith('v6') else ''
        rows.append(f'<tr><td style="{bold}white-space:nowrap;vertical-align:top;padding:4px 10px 4px 0">'
                    f'{html.escape(run.label)}</td><td style="vertical-align:top;text-align:right;padding:4px 10px;'
                    f'{bold}">{pt(run.wer, 1)}%</td><td style="vertical-align:top;padding:4px 0;line-height:1.5">'
                    f'{diff_html(ref, run.text, max_words)}</td></tr>')
    head = ('<tr><th style="text-align:left">sistema</th><th>WER</th><th style="text-align:left">'
            f'primeiras {max_words} palavras (<s style="color:{BAD}">erro</s>'
            f'<sup style="color:{GOOD}">referência</sup>)</th></tr>')
    return f'<table style="font-size:12px;border-collapse:collapse">{head}{"".join(rows)}</table>'


def show_image(path: Path, title: str, crop: float = 0.55) -> None:
    img = plt.imread(path)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.imshow(img[: int(img.shape[0] * crop)])
    ax.set_title(title, loc='left')
    ax.axis('off')
    plt.show()


# ---------------------------------------------------------------- figures

def plot_split() -> None:
    roles = [('treino (rótulos do Gemini)', 78, OTHER), ('ajuste (GT oficial, toda escolha)', 9, MUTED),
             ('reservadas (GT oficial)', 3, OURS), ('teste Livro nº 1 (ref. Gemini)', 10, OURS),
             ('Livro de Atas (ref. Gemini, sem treino)', 20, OURS)]
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for i, (name, n, c) in enumerate(roles):
        ax.barh(i, n, color=c, height=0.6)
        ax.text(n + 1, i, str(n), va='center', color=INK)
    ax.set_yticks(range(len(roles)), [r[0] for r in roles])
    ax.invert_yaxis()
    ax.set_xlabel('páginas')
    ax.set_title('Papel de cada página (azul = só avaliação)', loc='left')
    plt.show()


def plot_pipeline() -> None:
    steps = [('Foto da página', 'ampliação 2×\ncorreção de\ninclinação'),
             ('Duas imagens', 'segmentar: contraste,\nsem pauta\nreconhecer: crua'),
             ('Linhas', 'Kraken blla\npolígonos valem\nnas duas imagens'),
             ('Reconhecedor CTC', 'st_c1m = medium\n+ 2.425 linhas\nrotuladas pelo Gemini'),
             ('Pós-correção P6', 'reescore CTC com\nléxico de época +\nViterbi de trigramas')]
    fig, ax = plt.subplots(figsize=(11, 2.3))
    for i, (head, body) in enumerate(steps):
        c = OURS if i in (3, 4) else MUTED
        ax.add_patch(plt.Rectangle((i * 2.2, 0), 1.9, 1, fill=False, ec=c, lw=1.6))
        ax.text(i * 2.2 + 0.95, 0.80, head, ha='center', weight='bold', color=c, fontsize=9)
        ax.text(i * 2.2 + 0.95, 0.36, body, ha='center', va='center', fontsize=8, color=INK)
        if i:
            ax.annotate('', (i * 2.2, 0.5), (i * 2.2 - 0.3, 0.5), arrowprops={'arrowstyle': '->', 'color': MUTED})
    ax.set_xlim(-0.1, 10.9)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title('Leitura de uma página (azul: o que é nosso). O LLM não entra na leitura.', loc='left')
    plt.show()


def plot_progression() -> None:
    names = [p[0] for p in PROGRESSION]
    wer = [p[3] for p in PROGRESSION]
    cer = [p[4] for p in PROGRESSION]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(x - 0.2, wer, 0.38, color=[OURS if n == 'v6' else OTHER for n in names], label='WER')
    ax.bar(x + 0.2, cer, 0.38, color=[OURS if n == 'v6' else OTHER for n in names], alpha=0.45, hatch='//',
           label='CER')
    for xi, w in zip(x, wer):
        ax.text(xi - 0.2, w + 1, f'{w:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylabel('% (reservadas, GT oficial)')
    ax.legend(frameon=False)
    ax.set_title('Progressão das versões: 64,6% → 18,7% de WER', loc='left')
    plt.show()


def plot_grid(set_ref: str, title: str) -> None:
    g = grid()
    recs, decs = list(REC), list(DEC)
    m = np.full((len(recs), len(decs)), np.nan)
    for i, r in enumerate(recs):
        for j, d in enumerate(decs):
            v = g.get(f'{r}·{d}', {}).get(set_ref)
            if v:
                m[i, j] = v['wer']
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    im = ax.imshow(np.log10(m), cmap='Blues_r', aspect='auto')
    for i in range(len(recs)):
        for j in range(len(decs)):
            if not np.isnan(m[i, j]):
                best = (recs[i], decs[j]) == ('r1_st_c1m', 'p6')
                ax.text(j, i, f'{m[i, j]:.1f}', ha='center', va='center', fontsize=9,
                        color='white' if m[i, j] < np.nanmedian(m) else INK, weight='bold' if best else None)
    ax.set_xticks(range(len(decs)), [DEC[d] for d in decs], rotation=15)
    ax.set_yticks(range(len(recs)), [REC[r] for r in recs])
    ax.set_title(title, loc='left')
    ax.spines[:].set_visible(False)
    plt.show()


def system_label(key: str) -> str:
    rec, dec = key.split('·')
    if rec == 'v0':
        return 'VLM lê a linha'
    names = {'r0_medium': 'medium publicado', 'r1_st_c1m': 'nosso rec.', 'r4_traprinq': 'TraPrInq adapt.',
             'r5_mccatmus': 'McCATMuS adapt.'}
    decs = {'pa': 'KenLM', 'pa181': 'KenLM 181 configs', 'p6': 'P6', 'pe': 'VLM corrige', 'p0': 'guloso',
            'pb': 'word beam'}
    return f'{names.get(rec, rec)} + {decs.get(dec, dec)}'


def plot_paired(set_: str = 'test10', ref: str = 'gemini') -> None:
    rows = [p for p in paired() if p['set'] == set_ and p['ref'] == ref and p['a'] == 'r1_st_c1m·p6']
    rows.sort(key=lambda p: p['dwer'])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 0.8))
    for i, p in enumerate(rows):
        lo, hi = p['ci95']
        c = OURS if hi < 0 else MUTED
        ax.plot([lo, hi], [i, i], color=c, lw=2)
        ax.plot(p['dwer'], i, 'o', color=c, ms=6)
        ax.text(hi + 0.3, i, pt(p['dwer'], sign=True), va='center', fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)), [system_label(p['b']) for p in rows])
    ax.set_xscale('symlog', linthresh=5)
    ticks = [-50, -20, -10, -5, -2, 0]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(-60, 3)
    ax.set_xlabel('ΔWER = v6 − outro (pontos; IC95 bootstrap pareado). Negativo: v6 erra menos.')
    ax.set_title('v6 contra cada sistema (10 páginas de teste, ref. Gemini)', loc='left')
    plt.show()


def plot_atas() -> None:
    rep = load_json(E19 / 'e19_report.json')['runs']
    pages = sorted(int(p) for p in rep['default']['per_page'])
    d = [rep['default']['per_page'][str(p)] for p in pages]
    lx = [rep['lex']['per_page'][str(p)] for p in pages]
    x = np.arange(len(pages))
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(x - 0.2, d, 0.38, color=OTHER, label=f"v6 padrão ({pt(100 * rep['default']['wer_norm'])}%)")
    ax.bar(x + 0.2, lx, 0.38, color=OURS,
           label=f"v6, imagem escolhida sem gabarito ({pt(100 * rep['lex']['wer_norm'])}%)")
    ax.set_xticks(x, pages)
    ax.set_xlabel('página do Livro de Atas')
    ax.set_ylabel('WER %')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Livro de Atas (1894–1902), sem treino: WER por página', loc='left')
    plt.show()


def plot_references() -> None:
    g = grid()
    systems = [('r0_medium·p0raw', 'publicado'), ('r4_traprinq·pa', 'TraPrInq adapt. + KenLM'), ('r1_st_c1m·p6', 'v6')]
    x = np.arange(len(systems))
    fig, ax = plt.subplots(figsize=(6.5, 3))
    of = [g[s]['heldout/oficial']['wer'] for s, _ in systems]
    ge = [g[s]['heldout/gemini']['wer'] for s, _ in systems]
    ax.bar(x - 0.2, of, 0.38, color=INK, label='GT oficial')
    ax.bar(x + 0.2, ge, 0.38, color=OURS, label='referência Gemini')
    for xi, a, b in zip(x, of, ge):
        ax.text(xi - 0.2, a + 1, f'{a:.1f}', ha='center', fontsize=8)
        ax.text(xi + 0.2, b + 1, f'{b:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x, [s[1] for s in systems])
    ax.set_ylabel('WER % (3 reservadas)')
    ax.legend(frameon=False)
    ax.set_title('As mesmas 3 páginas, duas referências', loc='left')
    plt.show()


def final_table_html() -> str:
    g = grid()
    rows = [('r0_medium·p0raw', 'Publicado (medium, imagem crua)'), ('r2_traprinq·p0raw', 'TraPrInq publicado, imagem crua'),
            ('v0·v0', 'VLM lê a linha'), ('r1_st_c1m·pe', 'Nosso reconhecedor + VLM corrige'),
            ('r4_traprinq·pa181', 'TraPrInq adaptado + KenLM (181 configs)'),
            ('r4_traprinq·p6', 'TraPrInq adaptado + P6'), ('r1_st_c1m·pa181', 'Nosso reconhecedor + KenLM (181)'),
            ('r1_st_c1m·p6', 'v6 (nosso reconhecedor + P6)')]
    cols = [('heldout/oficial', 'reservadas, GT oficial'), ('heldout/gemini', 'reservadas, Gemini'),
            ('test10/gemini', 'teste, Gemini')]
    head = '<tr><th style="text-align:left">sistema</th>' + ''.join(f'<th>{c[1]}</th>' for c in cols) + '</tr>'
    body = []
    for key, name in rows:
        b = 'font-weight:bold;' if key == 'r1_st_c1m·p6' else ''
        cells = ''.join(f'<td style="text-align:right;padding:2px 12px;{b}">{g[key][c]["wer"]:.2f}</td>' for c, _ in cols)
        body.append(f'<tr><td style="padding:2px 12px 2px 0;{b}">{name}</td>{cells}</tr>')
    return f'<table style="font-size:13px">{head}{"".join(body)}</table><p style="font-size:11px">WER %, normalizado.</p>'


def plot_gains() -> None:
    g = grid()
    base = g['r0_medium·p0raw']['test10/gemini']['wer']
    steps = [('publicado, imagem crua', 'r0_medium·p0raw'), ('publicado, nossa imagem', 'r0_medium·p0'),
             ('só a P6 (reconhecedor publicado)', 'r0_medium·p6'),
             ('só os rótulos (st_c1m, guloso)', 'r1_st_c1m·p0'), ('rótulos + P6 = v6', 'r1_st_c1m·p6')]
    vals = [g[k]['test10/gemini']['wer'] for _, k in steps]
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.barh(range(len(steps)), vals, color=[OTHER] * (len(steps) - 1) + [OURS], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.6, i, f'{pt(v, 1)}%  ({pt(v - base, 1, sign=True)})', va='center', fontsize=9)
    ax.set_yticks(range(len(steps)), [s[0] for s in steps])
    ax.invert_yaxis()
    ax.set_xlabel('WER % nas 10 páginas de teste (ref. Gemini)')
    ax.set_title('Cada peça sozinha e as duas juntas', loc='left')
    plt.show()
