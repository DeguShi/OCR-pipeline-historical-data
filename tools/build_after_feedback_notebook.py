"""Build notebooks/after_feedback.ipynb: E17 (post-correction vs other methods), E18 (adaptive preprocessing) and E19
(a new book, automatic crop), read only from committed records (no OCR is run). Execute it with --execute.

  python tools/build_after_feedback_notebook.py --out notebooks/after_feedback.ipynb --execute
"""
import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

SETUP = r'''
import json
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from PIL import Image as PILImage

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
R = ROOT / 'records'
plt.rcParams.update({'figure.dpi': 110, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def load(path: str) -> dict:
    return json.loads((R / path).read_text(encoding='utf-8'))


def num(x: float, n: int = 2) -> str:
    return f'{x + 0.0:.{n}f}'.replace('.', ',').replace('-0,' + '0' * n, '0,' + '0' * n)


def ci(lo: float, hi: float) -> str:
    return f'[{num(lo)}; {num(hi)}]'


def table(header: list[str], rows: list[list]) -> None:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    display(Markdown('\n'.join(lines)))


def wer_of(path: Path) -> tuple[float, float]:
    c = json.loads(path.read_text(encoding='utf-8'))['corpus']
    return 100 * c['wer_norm'], 100 * c['cer_norm']
'''

INTRO = '''# OCR do Livro de Registro de Leis nº 1 — o que foi feito depois da entrega da v6

Este notebook reúne os três experimentos feitos depois da entrega da v6 (01/11/2026). Ele **não roda OCR**: lê os
resultados já registrados em `records/` (JSONs, figuras), gerados pelos scripts em `tools/` e pelo pipeline em
`ocr_pipeline/`. Cada número aponta para o arquivo de onde veio.

| experimento | pergunta |
|---|---|
| **E17** | Quanto a nossa pós-correção acrescenta, e como ela se compara com outros métodos de pós-correção sobre a mesma saída do reconhecedor? |
| **E18** | Dá para escolher o pré-processamento de imagem de um livro novo a partir de uma página de amostra, sem rótulos? |
| **E19** | O pipeline transfere, sem treino, para outro livro do mesmo acervo? E o recorte automático da página corta texto? |

**Protocolo comum.** Páginas de ajuste 5, 6, 8, 9, 11, 13, 14, 15, 16; reservadas 7, 10, 12 (GT oficial); teste com
10 páginas (18, 22, 23, 24, 25, 31, 32, 34, 44, 55; referência diplomática do Gemini). Nada é treinado ou selecionado em
reservadas ou teste. WER/CER normalizados, em nível de corpus. Comparações pareadas: bootstrap por blocos de 10
palavras de referência (10 000 reamostragens, seed 42, p por inversão de sinal), mais páginas ganhas/empatadas/perdidas.
'''

E17_MD = '''## E17 — Pós-correção contra outros métodos

Todos os métodos leem **a mesma saída** do reconhecedor `st_c1m` (posteriors CTC das mesmas linhas). Assim a
diferença medida é só da pós-correção. Hiperparâmetros de cada método: escolhidos só nas 9 páginas de ajuste.

### Degraus da nossa pós-correção (GT oficial)
'''

E17_LADDER = r'''
lad = load('postcorr/ladder_norm.json')
names = {'ladder_L0_raw': 'L0 CTC guloso', 'ladder_L1_rules': 'L1 + regras de texto',
         'ladder_L2_edit': 'L2 + palavra mais próxima do dicionário de época', 'ladder_L3_ctc': 'L3 + reescore CTC',
         'ladder_L4_tri': 'L4 + Viterbi de trigramas (básico)', 'ladder_L5b_bigram': 'v6 com bigrama no lugar do trigrama',
         'ladder_L5c_nobeam': 'v6 sem candidatos do beam', 'v6_all12_cache': '**v6**'}
table(['etapa', 'WER ajuste', 'WER reservadas', 'WER 12 páginas', 'CER 12 páginas', 'redução de WER (12 pág.)'],
      [[names[k], num(lad['tuning'][k]['wer']), num(lad['heldout'][k]['wer']), num(lad['all12'][k]['wer']),
        num(lad['all12'][k]['cer']), f"{num(lad['all12'][k]['wer_reduction_pct'], 1)}%"] for k in names])
'''

E17_FINAL_MD = '''### Tabela final: v6 contra os outros métodos

- **A** pyctcdecode 0.5.0 + KenLM (n-gramas do mesmo corpus de época), 181 configurações buscadas no ajuste;
- **B** CTC Word Beam Search (Scheidl 2018; mesmo dicionário);
- **C** ByT5-small treinado com ruído sintético aprendido nas páginas de ajuste;
- **D** v6 + PortOldBERT (pseudo-log-verossimilhança no Viterbi da v6, λ escolhido no ajuste);
- **E** VLM Qwen2.5-VL-7B (protocolo one-shot de Thomas et al. 2026), vê o recorte da linha e o texto guloso.

Δ = WER(v6) − WER(método): negativo quer dizer que a v6 erra menos.
'''

E17_FINAL = r'''
methods = [('A: pyctcdecode + KenLM', 'base_A2_pyctc', False), ('B: Word Beam Search', 'base_B2_wbs', False),
           ('C: ByT5-small', 'base_C_byt5', False), ('D: v6 + PortOldBERT', 'D_mlm_vs_v6', True),
           ('E: VLM Qwen2.5-VL-7B', 'base_E2_vlm_period', False)]
raw = {s: load(f'postcorr/curve/st_c1m_{s}.json')['wer_b'] for s in ('heldout', 'test')}
v6 = {s: load(f'postcorr/blocks/blk_{s}_base_A2_pyctc_10.json')['wer_a'] for s in ('heldout', 'test')}
rows, wers = [['sem pós-correção (CTC guloso)', num(raw['heldout']), '', num(raw['test']), '']], {}
rows.append(['**v6 (nossa)**', f"**{num(v6['heldout'])}**", '', f"**{num(v6['test'])}**", ''])
for label, key, flipped in methods:
    row = [label]
    for s in ('heldout', 'test'):
        b = load(f'postcorr/blocks/blk_{s}_{key}_10.json')
        wer = b['wer_a'] if flipped else b['wer_b']
        d, lo, hi = (-b['dwer'], -b['ci95'][1], -b['ci95'][0]) if flipped else (b['dwer'], *b['ci95'])
        w, t, l = b['pages_won_tied_lost']
        if flipped:
            w, l = l, w
        row += [num(wer), f"{num(d)} {ci(lo, hi)}, p={num(b['p_signflip'], 3)}, págs. {w}/{t}/{l}"]
        wers.setdefault(label, {})[s] = wer
    rows.append(row)
table(['método', 'WER reservadas', 'Δ v6 − método (IC95), p, páginas v6 ganha/empata/perde', 'WER teste', 'Δ idem'], rows)

fig, ax = plt.subplots(figsize=(7.5, 2.8))
labels = ['sem pós-corr.', 'v6'] + [m[0].split(':')[0] for m in methods]
vals = {s: [raw[s], v6[s]] + [wers[m[0]][s] for m in methods] for s in ('heldout', 'test')}
x = range(len(labels))
ax.bar([i - 0.2 for i in x], vals['heldout'], 0.4, color='0.35', label='reservadas (GT oficial)')
ax.bar([i + 0.2 for i in x], vals['test'], 0.4, color='0.75', hatch='//', edgecolor='0.3', label='teste (ref. diplomática)')
ax.set_xticks(list(x), labels)
ax.set_ylabel('WER (%)')
ax.legend(frameon=False)
ax.set_title('Mesma saída do reconhecedor, pós-correções diferentes')
plt.show()
'''

E17_READ = '''**Leitura.**
- A v6 é a melhor no teste contra A, B, C e E com IC que exclui 0. Contra o A (KenLM, após busca justa de
  hiperparâmetros) a vantagem no teste é de ~0,8 ponto; nas reservadas (3 páginas) a diferença está dentro do ruído.
- D (v6 + modelo de linguagem mascarado) fica um pouco abaixo da v6, sem significância, e custa ~44 s de GPU por página.
- O VLM fica bem atrás da v6 nos dois conjuntos. No teste (referência diplomática) ele piora o texto em relação ao OCR
  cru (18,05 contra 11,54): moderniza grafia e acentos e inventa palavras plausíveis. Nas reservadas (GT oficial, que
  moderniza) ele melhora o cru (26,30 contra 31,50).

### Que erros cada método corrige (teste)
Para cada categoria de erro do OCR cru: quantos o método corrigiu e quantas palavras certas ele estragou.
'''

E17_ERR = r'''
et = load('postcorr/error_types_test.json')
cats = {'non_word': 'não-palavra', 'real_word': 'palavra real errada', 'short': 'palavras curtas',
        'numeral': 'numerais', 'split_merge': 'separação/junção', 'deleted': 'palavra perdida'}
runs = ['v6', 'A', 'B', 'C', 'VLM']
table(['categoria (erros no OCR cru)'] + [f'{r}: corrigidos / estragados' for r in runs],
      [[f"{cats[c]} ({et['runs']['v6'][c]['raw_errors']})"]
       + [f"{et['runs'][r][c]['fixed']} / {et['runs'][r][c]['broken']}" for r in runs] for c in cats])
'''

E17_CURVE_MD = '''A v6 resolve quase todas as não-palavras. O ponto fraco que sobra é a palavra real errada (lida como outra palavra
que existe). Numerais e separações/junções ela não corrige por construção.

### O ganho depende da qualidade do reconhecedor?
A mesma v6 aplicada a três reconhecedores (do mais fraco ao nosso).
'''

E17_CURVE = r'''
recs = [('medium (sem adaptação)', 'medium'), ('st_v3', 'st_v3'), ('st_c1m (nosso)', 'st_c1m')]
rows = []
fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6), sharey=True)
for ax, s, title in zip(axes, ('heldout', 'test'), ('reservadas', 'teste')):
    for (label, key), marker in zip(recs, 'osD'):
        b = load(f'postcorr/curve/{key}_{s}.json')
        ax.plot([0, 1], [b['wer_b'], b['wer_a']], marker=marker, color='0.2', label=label)
        if s == 'heldout':
            rows.append([label])
        rows[[r[0] for r in rows].index(label)] += [f"{num(b['wer_b'])} → {num(b['wer_a'])}", ci(*b['ci95'])]
    ax.set_xticks([0, 1], ['sem pós-correção', 'v6'])
    ax.set_title(title)
axes[0].set_ylabel('WER (%)')
axes[1].legend(frameon=False, fontsize=8)
plt.show()
table(['reconhecedor', 'reservadas', 'Δ IC95', 'teste', 'Δ IC95'], rows)
'''

E17_EX_MD = '''### Exemplos de linhas
Uma linha por situação, com o recorte da imagem, a referência e a saída de cada método (entre parênteses, o número de
palavras erradas). Seleção automática registrada em `records/postcorr/paper_examples/`.
'''

E17_EX = r'''
ex = load('postcorr/paper_examples/examples.json')
titles = {'v6_fixes_others_miss': 'v6 corrige o que os outros não corrigem', 'vlm_modernizes': 'o VLM moderniza a grafia',
          'byt5_better': 'onde o ByT5 é melhor', 'v6_breaks': 'onde a v6 estraga'}
shown = set()
for key, title in titles.items():
    e = next(x for x in ex[key] if (x['set'], x['page'], x['line']) not in shown)
    shown.add((e['set'], e['page'], e['line']))
    display(Markdown(f"**{title}** — {e['set']}, página {e['page']}, linha {e['line']}"))
    display(Image(filename=str(R / 'postcorr/paper_examples' / e['crop']), width=700))
    table(['', 'texto (erros)'], [['referência (normalizada)', e['ref']]]
          + [[m, f"{t} ({e['errors'][m]})"] for m, t in e['texts'].items()])
'''

E18_MD = '''## E18 — Pré-processamento adaptativo a partir de uma página

Ideia: para um livro novo, testar combinações das operações de imagem que o pipeline já tem (contraste, CLAHE,
iluminação, gamma, canal, ruído, ampliação ×1/×2/×3, remoção de pautas...) numa **página de amostra** e escolher a
combinação com um critério **sem rótulos**. A busca é em duas etapas: 8 candidatos de segmentação, depois 13 de
imagem para o reconhecimento. Critérios comparados:

- `lex`: número de palavras da leitura gulosa que existem no léxico de época (top 100 mil);
- `conf`: confiança média do CTC; `conflex` = conf × lex;
- `stat`: aproximar as estatísticas da imagem (fundo, tinta, contraste, ruído, traço) das páginas do nosso livro;
- `oracle`: CER guloso na amostra **usando a transcrição dela** — o limite do que rótulos de uma página dariam.

### Seis degradações sintéticas das nossas páginas (WER nas reservadas)
A busca vê só a página de ajuste 6; a avaliação é a v6 completa nas reservadas 7, 10 e 12. Entre parênteses: parte do
dano da degradação que foi recuperada.
'''

E18_SHIFTS = r'''
rep = load('e18/e18_report/e18_report.json')
names = {'uneven_light': 'iluminação desigual', 'low_contrast_cast': 'baixo contraste + dominante de cor',
         'faded_ink': 'tinta desbotada', 'bleed_through': 'tinta vazada do verso', 'jpeg_noise': 'JPEG + ruído',
         'low_res_blur': 'baixa resolução + desfoque'}
objs = ['conf', 'lex', 'stat', 'oracle']
rows = []
for s in rep['shifts']:
    cells = [names[s['shift']], num(s['wer']['default'])]
    for o in objs:
        rec = s['recovery'][o]
        note = '' if s['loss'] < 1 else (' (piora)' if rec < 0 else f' ({rec:.0%})')
        cells.append(num(s['wer'][o]) + note)
    rows.append(cells)
display(Markdown(f"Sem degradação, a v6 dá **{num(rep['clean'])}** nas reservadas."))
table(['degradação', 'padrão'] + objs, rows)
table(['degradação'] + [f'Spearman({o}, CER guloso)' for o in ('lex', 'conflex', 'conf', 'stat')],
      [[names[s['shift']]] + [num(s['spearman'][o]) for o in ('lex', 'conflex', 'conf', 'stat')] for s in rep['shifts']])
'''

E18_READ = '''**Leitura.** O `lex` cumpre o critério definido antes de rodar: nas degradações que custam ≥ 2 pontos, recupera em
média 57% do dano (89%, 28%, 55%) e nunca piora em relação ao padrão. Ele também ordena os candidatos quase como o CER
real (Spearman 0,94–0,97). O `stat` ("deixar a imagem parecida com o nosso livro") falha: piora em 3 de 6. O oráculo
de uma página não é melhor que o `lex`.

Exemplo (iluminação desigual e baixa resolução): entrada degradada, imagem de segmentação padrão, e as imagens de
segmentação e de reconhecimento escolhidas pelo `lex`.
'''

E18_FIG = r'''
for shift in ('uneven_light', 'low_res_blur'):
    parts = [('input', 'entrada'), ('default_seg', 'segmentação padrão'), ('chosen_seg', 'segmentação (lex)'),
             ('chosen_rec', 'reconhecimento (lex)')]
    fig, axes = plt.subplots(4, 1, figsize=(9, 6.4))
    for ax, (part, title) in zip(axes, parts):
        ax.imshow(PILImage.open(R / f'e18/e18_figures/{shift}_{part}.png').convert('L'), cmap='gray')
        ax.set_title(title, fontsize=8, loc='left')
        ax.axis('off')
    fig.suptitle(names[shift], fontsize=9)
    plt.show()
'''

E18_CTRL_MD = '''### Controles
- **Identidade:** a busca inteira no nosso próprio livro, sem degradação. Um bom critério deve manter a cadeia padrão.
- **Cadeia fixa:** CLAHE + remoção de pautas em tudo (a escolha mais frequente do `lex` e do oráculo). Se ela fosse tão
  boa quanto o `lex`, o ganho viria do CLAHE, não de escolher por livro.
'''

E18_CTRL = r'''
ident = {o: wer_of(R / f'e18/controls/identity/runs/{o}/result.json')[0] for o in ('default', 'lex', 'conf', 'stat', 'oracle')}
table(['identidade: critério', 'WER reservadas'], [[o, num(w)] for o, w in ident.items()])
rows = [['sem degradação', num(rep['clean']), num(wer_of(R / 'e18/controls/clean/runs/fixed_clahe/result.json')[0]),
         num(ident['lex'])]]
for s in rep['shifts']:
    rows.append([names[s['shift']], num(s['wer']['default']),
                 num(wer_of(R / f"e18/controls/{s['shift']}/runs/fixed_clahe/result.json")[0]), num(s['wer']['lex'])])
table(['degradação', 'padrão', 'cadeia fixa CLAHE', 'lex'], rows)
'''

E18_BOOKS_MD = '''O `lex` mantém o padrão no próprio livro (18,74). O `conf` piora +1,4 e o oráculo de uma página sobreajusta. O `lex` é
melhor que a cadeia fixa em todas as 6 degradações: o ganho vem de escolher por livro.

### Livros reais: 4 processos do TraPrInq (Inquisição portuguesa, séc. XVI–XIX)
Validação do TraPrInq (Zenodo 13986218, CC BY 4.0), com transcrição diplomática em PAGE XML. Cada processo é tratado
como um livro novo: 1 página de amostra para a busca, v6 nas demais. O reconhecedor nunca viu essas mãos, então o
erro absoluto é alto. O que importa é a comparação dentro de cada livro.
'''

E18_BOOKS = r'''
rows = []
for case in sorted((R / 'e18/books').glob('PT-*')):
    cells = [case.name.replace('PT-TT-TSO-', '')]
    same_file = case / 'same_as.txt'  # configs identical to an earlier one were not re-run
    same = dict(l.split('/')[-1].split(' = ') for l in same_file.read_text().split('\n') if ' = ' in l) if same_file.exists() else {}
    for o in ('default', 'lex', 'conf', 'oracle'):
        if o in same:
            cells.append(f'= {same[o]}')
            continue
        w, c = wer_of(case / f'runs/{o}/result.json')
        cells.append(f'{num(w)} / {num(c)}')
    rows.append(cells)
table(['processo', 'padrão (WER / CER)', 'lex', 'conf', 'oracle'], rows)
'''

E18_BOOKS_READ = '''O `lex` fica igual ou melhor que o padrão em WER nos 4 processos (0 a −5,3 pontos). São 4 livros e 18 páginas:
evidência de direção, sem IC. Uma checagem de segmentação contra as linhas do PAGE XML (`records/e18/seg/`) mostrou
recall de 93–100% para todos os candidatos: o ganho vem da imagem entregue ao reconhecedor, não de achar mais linhas.
'''

E19_MD = '''## E19 — Um livro novo do mesmo acervo, sem treino

**Livro de Atas da Câmara Municipal de São Carlos, 1894–1902** (fotos da Pró-Memória): outro escrivão, atas em vez de
leis. 50 fotos (fólios 36–65, 1895–1896) com transcrição do Gemini. O sorteio, feito antes de qualquer OCR, separou 3
páginas de amostra para a busca, 20 de avaliação e deixou 27 sem uso. O reconhecedor e a pós-correção são os da v6,
sem nenhum treino; só a cadeia de pré-processamento é escolhida nas 3 amostras.

### v6 nas 20 páginas de avaliação (referência Gemini)
'''

E19_MAIN = r'''
rep19 = load('e19/e19_report.json')
cfg = {'default': 'contraste, ×2 (v6)', 'lex': 'CLAHE, ×2', 'oracle': 'contraste, ×3'}
rows = [['padrão', cfg['default'], num(100 * rep19['runs']['default']['wer_norm']),
         num(100 * rep19['runs']['default']['cer_norm']), '']]
for o in ('lex', 'oracle'):
    b = load(f'e19/bootstrap_default_vs_{o}.json')  # a = default, b = chosen; dwer = a - b
    w, t, l = b['pages_won_tied_lost']
    rows.append([o, cfg[o], num(100 * rep19['runs'][o]['wer_norm']), num(100 * rep19['runs'][o]['cer_norm']),
                 f"{num(-b['dwer'])} {ci(-b['ci95'][1], -b['ci95'][0])}, p={num(b['p_signflip'], 3)}, págs. {l}/{t}/{w}"])
table(['critério', 'cadeia escolhida (segmentação)', 'WER', 'CER', 'Δ WER vs padrão (IC95), p, págs. ganha/empata/perde'], rows)

pp = {o: rep19['runs'][o]['per_page'] for o in ('default', 'lex')}
pages = sorted(pp['default'], key=int)
fig, ax = plt.subplots(figsize=(8, 2.6))
x = range(len(pages))
ax.bar([i - 0.2 for i in x], [pp['default'][p] for p in pages], 0.4, color='0.35', label='padrão')
ax.bar([i + 0.2 for i in x], [pp['lex'][p] for p in pages], 0.4, color='0.75', hatch='//', edgecolor='0.3', label='lex')
ax.set_xticks(list(x), pages)
ax.set_xlabel('página do livro')
ax.set_ylabel('WER (%)')
ax.legend(frameon=False)
plt.show()
'''

E19_READ = '''**Leitura.** O critério definido antes de rodar (lex ≤ padrão e nunca pior por mais de 1 ponto) foi cumprido: o `lex`
é melhor, com IC que exclui 0. A escolha sem rótulos foi tão boa quanto a com rótulos das amostras. Os critérios que não
rodaram completos: `conflex` escolhe a mesma cadeia que o `lex`; `conf` e `stat` escolhem segmentações piores nas
amostras e na avaliação (tabela abaixo), então a v6 completa não foi rodada para eles.

### Segmentação contra as linhas das transcrições
Os 8 candidatos de segmentação nas 20 páginas de avaliação, comparando as linhas encontradas com as 773 linhas
transcritas (pareamento por texto; CER guloso, sem pós-correção).
'''

E19_SEG = r'''
sw = rep19['seg_weak']
rows = []
for name, s in sorted(sw.items(), key=lambda kv: kv[1]['greedy_cer']):
    rows.append([name, f"{s['found']} ({s['recall']:.1%})".replace('.', ','), s['extra'], s['split'], s['missed'],
                 num(100 * s['greedy_cer'])])
table(['cadeia de segmentação', 'linhas encontradas', 'extras', 'partidas', 'perdidas', 'CER guloso'], rows)
'''

E19_CROP_MD = '''A escolha do `lex` (up2 CLAHE) também é a melhor segmentação por CER nas páginas de avaliação. O que mais pesa é
remover as pautas: sem isso aparecem 77 linhas extras.

### Recorte automático da página
O livro 2 não tem os recortes manuais do livro 1. O `ocr_pipeline/auto_crop.py` recorta a partir da foto: acha a logo
d'água da Pró-Memória e pinta por cima, acha o papel e corta só margens sem tinta. A regra é nunca cortar traço.
'''

E19_CROP = r'''
c = rep19['crop']
b1h = rep19['book1_runs']['hand']
b1a = rep19['book1_runs']['auto']
bo = load('e19/bootstrap_book1_hand_vs_auto_official.json')
bg = load('e19/bootstrap_book1_hand_vs_auto_gemini.json')
table(['verificação (primeiro recorte)', 'resultado'], [
    ['livro 1: linhas do Kraken dentro do recorte', f"automático {c['book1']['auto_inside']}/{c['book1']['lines']}, manual {c['book1']['hand_inside']}/{c['book1']['lines']} (o manual corta a rubrica)"],
    ['livro 1, v6, GT oficial (WER/CER)', f"manual {num(100 * b1h['wer_norm'])} / {num(100 * b1h['cer_norm'])}; automático {num(100 * b1a['wer_norm'])} / {num(100 * b1a['cer_norm'])}; auto − manual {num(-bo['dwer'])} {ci(-bo['ci95'][1], -bo['ci95'][0])}, p={num(bo['p_signflip'])}"],
    ['livro 1, v6, ref. Gemini (WER/CER)', f"manual {num(100 * b1h['vs_gemini']['wer_norm'])} / {num(100 * b1h['vs_gemini']['cer_norm'])}; automático {num(100 * b1a['vs_gemini']['wer_norm'])} / {num(100 * b1a['vs_gemini']['cer_norm'])}; auto − manual {num(-bg['dwer'])} {ci(-bg['ci95'][1], -bg['ci95'][0])}, p={num(bg['p_signflip'])}"],
    ['livro 2: linhas do Kraken dentro', f"{c['book2']['inside']}/{c['book2']['lines']}; {len(c['book2']['outside_photos'])} fotos com linha fora"],
    ['TraPrInq (504 páginas, outro acervo): linhas do PAGE XML dentro', f"{c['pagexml']['inside']}/{c['pagexml']['lines']} ({c['pagexml']['inside'] / c['pagexml']['lines']:.1%})".replace('.', ',')],
])
'''

E19_CROP2_MD = '''No livro 2, das 8 fotos com linha fora, só 2 eram texto cortado de verdade (rubrica e número do fólio na borda: fotos
81 e 107). As outras eram linhas segmentadas passando da borda do papel ou linhas falsas sobre a logo pintada. No
TraPrInq, papel escuro, manchado ou muito escrito fazia o limiar separar papel de tinta em vez de papel de fundo.

**Duas correções (recorte v2):** (1) manter blocos de escrita que encostam na borda do papel; (2) caixa do papel como
união da caixa simples com uma caixa sobre a luminosidade "fechada" (tinta apagada). Comparação, só das caixas:
'''

E19_CROP2 = r'''
t = load('e19/crop2_traprinq.json')
b2 = load('e19/crop2_book2.json')
b1 = load('e19/crop2_book1.json')
n = sum(r['n_lines'] for r in t)
worse = sum(r['inside_new'] < r['inside_old'] for r in t)
table(['conjunto', 'primeiro recorte', 'recorte v2'], [
    ['TraPrInq: linhas dentro', f"{sum(r['inside_old'] for r in t)}/{n}", f"{sum(r['inside_new'] for r in t)}/{n} ({sum(r['inside_new'] for r in t) / n:.2%}); {worse} páginas pioram".replace('.', ',')],
    ['livro 2: caixas que mudam', '', f"{sum(r['old'] != r['new'] for r in b2)}/50, {'todas contêm a antiga' if all(r['contains'] for r in b2) else 'nem todas contêm a antiga'}"],
    ['livro 1: caixas que mudam', '', f"{sum(r['old'] != r['new'] for r in b1)}/{len(b1)}, área ×{num(max(r['area_ratio'] for r in b1))} no máximo"],
])
for f, title in [('crop2_book2_81.jpg', 'livro 2, foto 81: a rubrica "Souza Franco 39" na borda direita volta para dentro'),
                 ('crop2_traprinq_16809_m0014b.jpg', 'TraPrInq 16809_m0014b: o caso que o recorte v2 ainda não resolvia (metade de baixo fora)')]:
    display(Markdown(f'**{title}**'))
    display(Image(filename=str(R / 'e19/figures' / f), width=420))
'''

E19_CROP3_MD = '''**Terceira correção (recorte v3).** A página 16809_m0014b não tem fundo: linhas de escrita muito densa ficam com
menos de 50% de "papel" e quebram a faixa. Agora buracos de até 1% também são tapados na caixa simples, e a caixa
nunca fica menor que a v2. Comparação v2 × v3, só das caixas:
'''

E19_CROP3 = r'''
t3 = load('e19/crop3_traprinq.json')
n3 = sum(r['n_lines'] for r in t3)
table(['conjunto', 'recorte v2', 'recorte v3'], [
    ['TraPrInq: linhas dentro', f"{sum(r['inside_old'] for r in t3)}/{n3}",
     f"{sum(r['inside_new'] for r in t3)}/{n3} ({sum(r['inside_new'] for r in t3) / n3:.2%}); {sum(r['inside_new'] < r['inside_old'] for r in t3)} páginas pioram".replace('.', ',')],
    ['livro 1: caixas que mudam', '', f"{sum(r['old'] != r['new'] for r in load('e19/crop3_book1.json'))}/106 (1 px; nenhuma das 12 páginas com GT)"],
    ['livro 2: caixas que mudam', '', f"{sum(r['old'] != r['new'] for r in load('e19/crop3_book2.json'))}/50 (1 px; nenhuma página de avaliação)"],
])
display(Markdown('**16809_m0014b com o recorte v3** (vermelho: caixa v2; a caixa v3 vai até o pé da página)'))
display(Image(filename=str(R / 'e19/figures/crop3_traprinq_16809_m0014b.jpg'), width=420))
'''

E19_CROPV6_MD = '''### O recorte novo muda os resultados?
A v6 completa foi rodada de novo com o recorte v2. No livro 2 mudam 5 das 20 caixas de avaliação; as 15 páginas com a
mesma caixa deram exatamente o mesmo texto (o pipeline é determinístico). Como o v3 não muda nenhuma caixa avaliada,
estes números valem para o recorte atual.
'''

E19_CROPV6 = r'''
def flip(b: dict) -> str:  # bootstrap files store a - b; show b - a
    w, t, l = b['pages_won_tied_lost']
    return f"{num(-b['dwer'])} {ci(-b['ci95'][1], -b['ci95'][0])}, p={num(b['p_signflip'], 3)}, págs. {l}/{t}/{w}"


table(['comparação', 'WER', 'Δ (IC95), p, páginas ganha/empata/perde'], [
    ['livro 2, padrão: recorte v1 → v2', '22,50 → 22,55', flip(load('e19/bootstrap_default_v1_vs_v2.json'))],
    ['livro 2, lex: recorte v1 → v2', '21,14 → 21,47', flip(load('e19/bootstrap_lex_v1_vs_v2.json'))],
    ['**livro 2 com recorte v2: lex − padrão**', '21,47 vs 22,55', flip(load('e19/bootstrap_v2_default_vs_lex.json'))],
    ['livro 1, GT oficial: automático v2 − manual', '19,94 vs 19,40', flip(load('e19/bootstrap_book1_hand_vs_auto_v2_official.json'))],
    ['livro 1, ref. Gemini: automático v2 − manual', '8,33 vs 8,61', flip(load('e19/bootstrap_book1_hand_vs_auto_v2_gemini.json'))],
])
'''

E19_CROPV6_READ = '''Nenhuma mudança de recorte é significativa. Com o recorte v2, o `lex` continua melhor que o padrão (IC exclui 0), mas
a vantagem cai de 1,36 para 1,08 ponto e o p fica em 0,051: o lex perde 6 palavras na página 7 e 8 na página 35.
'''

CAVEATS = '''## Ressalvas e o que está em aberto

- **Referências.** O teste do livro 1 e todo o E19 usam transcrições do Gemini, não revisadas por pessoas; a `st_c1m`
  foi treinada com rótulos do Gemini de outras páginas, o que torna os números absolutos otimistas. O GT oficial das
  reservadas moderniza a grafia. Com só 3 páginas, as reservadas são indicativas.
- **E17** compara pós-correções sobre a mesma saída. Ainda não há comparação de **sistemas completos** (reconhecedor
  da literatura + sua decodificação, com os mesmos dados de treino): o plano está escrito (E20).
- **E18/E19:** um livro novo do mesmo acervo (20 páginas, 30 fólios seguidos de um escrivão) e 4 processos do
  TraPrInq. Evidência de direção forte no E19 (IC exclui 0), fraca no TraPrInq (sem IC).
- **Recorte:** 7 linhas do TraPrInq (5 páginas) ainda ficam fora; a logo d'água dá falso positivo em carimbos de outro
  acervo (sem perda de linha).
- **Uso:** o caminho opcional para outro livro (recorte automático → busca `lex` → batch) está no README do repositório.
'''


def build() -> nbformat.NotebookNode:
    md, code = new_markdown_cell, new_code_cell
    cells = [md(INTRO), code(SETUP), md(E17_MD), code(E17_LADDER), md(E17_FINAL_MD), code(E17_FINAL), md(E17_READ),
             code(E17_ERR), md(E17_CURVE_MD), code(E17_CURVE), md(E17_EX_MD), code(E17_EX), md(E18_MD), code(E18_SHIFTS),
             md(E18_READ), code(E18_FIG), md(E18_CTRL_MD), code(E18_CTRL), md(E18_BOOKS_MD), code(E18_BOOKS),
             md(E18_BOOKS_READ), md(E19_MD), code(E19_MAIN), md(E19_READ), code(E19_SEG), md(E19_CROP_MD), code(E19_CROP),
             md(E19_CROP2_MD), code(E19_CROP2), md(E19_CROP3_MD), code(E19_CROP3), md(E19_CROPV6_MD),
             code(E19_CROPV6), md(E19_CROPV6_READ), md(CAVEATS)]
    for cell in cells:
        cell.source = cell.source.strip('\n')
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('notebooks/after_feedback.ipynb'))
    parser.add_argument('--execute', action='store_true', help='run the notebook (reads records only) and keep outputs')
    args = parser.parse_args()
    nb = build()
    if args.execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=120, kernel_name='python3',
                       resources={'metadata': {'path': str(args.out.resolve().parent)}}).execute()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.out)
    print(f'wrote {args.out} ({len(nb.cells)} cells, executed={args.execute})')


if __name__ == '__main__':
    main()
