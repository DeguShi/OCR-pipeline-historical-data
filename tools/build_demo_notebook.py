"""Build notebooks/demo_artigo.ipynb, a presentation of the paper that only reads saved results.

Also copies the two Livro de Atas example pages (image, Gemini reference, v6 default and lex outputs) into
notebooks/demo_data/livro_atas, because the E19 run outputs are not in git.

    .venv/bin/python -I tools/build_demo_notebook.py \
        --e19-book outputs/e19_colab_e19a/e19/book --e19-runs outputs/e19_merged/b/runs
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
import demo_lib as L  # noqa: E402

logger = logging.getLogger(__name__)


def copy_atas(book: Path, runs: Path, pages: list[int]) -> None:
    out = L.DEMO / 'livro_atas'
    out.mkdir(parents=True, exist_ok=True)
    for p in pages:
        img = next((book / 'images').glob(f'{p}.*.jpg'))
        ref = next((book / 'references').glob(f'{p}.*.txt'))
        shutil.copy2(img, out / f'{p}.jpg')
        shutil.copy2(ref, out / f'{p}_ref.txt')
        for run in ('default', 'lex'):
            shutil.copy2(runs / run / f'page_{p:02d}.txt', out / f'{p}_{run}.txt')
    (out / 'SOURCE.json').write_text(json.dumps({'book': str(book), 'runs': str(runs), 'pages': pages}, indent=1))


def pick_pages() -> dict[str, list[int]]:
    """Median rule on the v6 page WER of each set (demo_lib.median_pages)."""
    picks = {}
    for set_, ref in (('heldout', 'oficial'), ('test10', 'gemini')):
        res = L.load_json(L.E20 / 'score' / set_ / 'r1_st_c1m' / 'p6' / ref / 'result.json')
        picks[set_] = L.median_pages({p['page']: 100 * p['wer_norm'] for p in res['pages']})
    rep = L.load_json(L.E19 / 'e19_report.json')['runs']['default']['per_page']
    picks['atas'] = L.median_pages({int(k): v for k, v in rep.items()})
    return picks


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def page_cells(set_: str, ref: str, page: int, title: str) -> list[nbf.NotebookNode]:
    return [md(f'#### {title}'), code(f"show_page_book1('{set_}', '{ref}', {page})")]


def build(picks: dict[str, list[int]]) -> nbf.NotebookNode:
    h, t, a = picks['heldout'], picks['test10'], picks['atas']
    cells = [
        md("""
# OCR do *Livro de Registro de Leis nº 1* (São Carlos, 1892–1894)

Um reconhecedor de linhas de 64 MB, ajustado com transcrições que o Gemini fez de *outras* páginas do livro, e uma
pós-correção com recursos de época levam o WER de **64,6% para 18,7%** nas três páginas reservadas, contra o
gabarito oficial. A leitura roda em CPU, sem LLM.

Este notebook só lê resultados salvos (`records/e20`, `records/x02`, `records/e19`). Não treina, não busca, não decodifica.
"""),
        code("""
import sys
from pathlib import Path
from IPython.display import HTML, display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'notebooks/demo_lib.py').exists())
sys.path.insert(0, str(ROOT / 'notebooks'))
import demo_lib as L
G = L.grid()

def show_page_book1(set_, ref, page):
    L.show_image(L.page_image(set_, page), f'Página {page}')
    ref_text = L.reference(set_, ref, page)
    runs = []
    for rec, dec, label in L.SHOWCASE:
        r = L.e20_page(set_, rec, dec, ref, page)
        if r:
            runs.append(L.PageRun(label, r.text, r.wer))
    display(HTML(L.runs_table_html(runs, ref_text)))

def show_page_atas(page):
    d = L.DEMO / 'livro_atas'
    rep = L.load_json(L.E19 / 'e19_report.json')['runs']
    L.show_image(d / f'{page}.jpg', f'Livro de Atas, página {page}')
    runs = [L.PageRun('v6 padrão', (d / f'{page}_default.txt').read_text(), rep['default']['per_page'][str(page)]),
            L.PageRun('v6, imagem escolhida sem gabarito', (d / f'{page}_lex.txt').read_text(),
                      rep['lex']['per_page'][str(page)])]
    display(HTML(L.runs_table_html(runs, (d / f'{page}_ref.txt').read_text())))
"""),
        md("""
## 1. Dados e protocolo

- **12 páginas com GT oficial** da Fundação Pró-Memória. Toda escolha é feita em 9 delas (**ajuste**); as outras 3
  (**reservadas**: 7, 10 e 12) só avaliam.
- **78 páginas de treino**, com transcrições do Gemini como rótulo. Nenhum GT humano vira rótulo.
- **10 páginas de teste** do mesmo livro e **20 páginas de outro livro** (Livro de Atas). A referência delas é a
  transcrição do Gemini, sem revisão humana.

Nenhuma página avaliada entra em treino ou escolha, e a configuração foi congelada antes da avaliação.
"""),
        code('L.plot_split()'),
        md("""
## 2. O pipeline (v6)

Cada foto gera duas imagens. A de segmentação recebe contraste e perde a pauta. A de reconhecimento fica sem
tratamento, porque binarizar ajuda o segmentador, mas custa cerca de 10 pontos ao reconhecedor, que foi treinado com
imagens reais.

O reconhecedor, **st_c1m**, é o *medium* publicado (PP-OCRv6, Kraken) com ajuste fino em 2.425 linhas rotuladas pelo
Gemini. A pós-correção, **P6**, parte das probabilidades do CTC e não só do texto lido. Ela reescora candidatas do
dicionário de época e depois escolhe a sequência da linha com um Viterbi de trigramas.
"""),
        code('L.plot_pipeline()'),
        md("""
O dicionário e os trigramas vêm de 443 livros do Gutenberg (autores nascidos entre 1790 e 1875), de 3.565 leis de
São Paulo (1889–1910) e do texto das 78 páginas de treino. Com um léxico moderno, *approvação* virava *aprovação*,
o que conta como erro.

Os parâmetros foram escolhidos por uma busca gulosa condicional: uma etapa por vez, sempre testada sobre o melhor
conjunto atual, para captar interações entre etapas. No mesmo espaço, ela chegou a 36,00% de WER no ajuste, contra
43,62% da busca aleatória, com menos da metade das avaliações. As buscas gulosas somaram cerca de 2.300 avaliações e
92 horas.
"""),
        md('## 3. Progressão das versões'),
        code('L.plot_progression()'),
        md("""
Cada versão soma peças à anterior, então o tamanho de cada degrau depende da ordem em que a peça entrou; a
contribuição de cada peça isolada está na Seção 4. A v6 repete a configuração da v5, mas com um reconhecedor
retreinado numa cadeia que não usou nenhuma página avaliada. Por isso ela é a versão relatada, mesmo com WER um pouco
maior nas reservadas.

O "publicado" desta tabela é a linha de base original do projeto. Medida de novo na comparação da Seção 4, a mesma
condição dá 60,8%.
"""),
        md("""
## 4. Comparação controlada com a literatura

A comparação cruza seis reconhecedores com até cinco formas de transformar a saída do CTC em texto (decodificação
ou pós-correção), todos sobre a mesma imagem e a mesma segmentação. R4 e R5 são o TraPrInq e o McCATMuS ajustados
**com os mesmos rótulos** do nosso reconhecedor. Assim, a diferença entre R1 e R4 mede o modelo de partida, não os
rótulos. A cor segue o WER em escala logarítmica.
"""),
        code("L.plot_grid('test10/gemini', 'WER % nas 10 páginas de teste (ref. Gemini)')"),
        code("L.plot_grid('heldout/oficial', 'WER % nas 3 reservadas (GT oficial)')"),
        md("""
Cada ponto é a diferença de WER entre a v6 e um sistema, com IC95 de um bootstrap pareado. As linhas "181 configs"
vêm do X02, que repetiu a busca do KenLM com 181 configurações, quase o dobro das 92 originais.
"""),
        code('L.plot_paired()'),
        md("""
No teste, nenhum intervalo cruza o zero: a v6 erra menos que cada um dos 11 sistemas. Contra a melhor receita
publicada, o TraPrInq ajustado com KenLM, a v6 erra 2,34 pontos a menos. Desses, 1,24 vêm de trocar o KenLM pela P6
no mesmo reconhecedor, e o 1,11 restante vem de trocar o reconhecedor. Com a busca do KenLM quase dobrada, a
diferença continua (−2,57).

Nas 3 reservadas, contra o GT oficial, a diferença para essa mesma receita é de −0,47 [−2,07; 1,13]. O intervalo
contém o zero, então ali os dois sistemas não se distinguem.
"""),
        code('L.plot_gains()'),
        md("""
Cada peça sozinha já ajuda muito, e os rótulos ajudam mais que a pós-correção. Combinadas, chegam a 3,5%.
"""),
        md(f"""
## 5. Página por página

De cada conjunto entram as duas páginas com o WER da v6 mais próximo da mediana. A regra é fixa, para não escolher
as páginas em que a v6 vai melhor. Para cada sistema aparecem as primeiras 70 palavras: o erro riscado e, acima
dele, a palavra da referência. Um ordinal escrito "3.º" aparece como "3 º" contra "3º", porque a pontuação é
removida antes da comparação, como na métrica.

### Reservadas, contra o GT oficial (páginas {h[0]} e {h[1]})
"""),
        *page_cells('heldout', 'oficial', h[0], f'Página {h[0]} (reservada, GT oficial)'),
        *page_cells('heldout', 'oficial', h[1], f'Página {h[1]} (reservada, GT oficial)'),
        md(f'### Teste do Livro nº 1, contra o Gemini (páginas {t[0]} e {t[1]})'),
        *page_cells('test10', 'gemini', t[0], f'Página {t[0]} (teste, ref. Gemini)'),
        *page_cells('test10', 'gemini', t[1], f'Página {t[1]} (teste, ref. Gemini)'),
        md(f"""
### Livro de Atas, outro livro e outro escrivão, sem treino (páginas {a[0]} e {a[1]})

Neste livro só a v6 foi rodada; a comparação com a literatura não. As duas linhas são a v6 com a imagem padrão e
com a imagem escolhida, sem gabarito, pela regra "mais palavras do dicionário de época lidas".
"""),
        md(f'#### Página {a[0]}'), code(f'show_page_atas({a[0]})'),
        md(f'#### Página {a[1]}'), code(f'show_page_atas({a[1]})'),
        md('## 6. Um livro novo, sem treino'),
        code('L.plot_atas()'),
        md("""
Sem nenhum ajuste a este escrivão, a v6 faz 22,50% de WER. Com a imagem escolhida pela regra, faz 21,14%: imagem
escolhida menos padrão, −1,36 [−2,38; −0,36] (bootstrap pareado). Os dois números são contra o Gemini, a mesma fonte
dos rótulos de treino, então tendem a ser otimistas.
"""),
        md("""
## 7. Quanto vale a referência?

O GT oficial moderniza a grafia e escreve os ordinais como *1º*. O Gemini, e portanto o nosso reconhecedor, escreve
*1.º*. Nas 12 páginas com GT oficial, a v6 tem 19,40% de WER contra ele, e 5,6 desses pontos são convenções
juntadas ou separadas, quase todas ordinais.

Os dois sistemas ajustados com rótulos do Gemini, a v6 e o TraPrInq ajustado, caem de cerca de 19% para 6–7% quando
a referência passa a ser o Gemini. O publicado não cai. Parte da queda é, portanto, a convenção que os rótulos
ensinaram. Nenhuma das duas referências é neutra, e por isso os números aparecem contra as duas.
"""),
        code('L.plot_references()'),
        md('## 8. Comparação final'),
        code('display(HTML(L.final_table_html()))'),
        md("""
## 9. O que um revisor vai perguntar

- **A referência de teste é o Gemini**, a mesma fonte dos rótulos. Uma transcrição diplomática humana das 13 páginas
  avaliadas está em andamento (X01).
- **Só 3 páginas reservadas têm GT oficial**, poucas para testes de significância contra ele.
- **O reconhecedor foi adaptado a um só livro.** O segundo livro só tem referência do Gemini.
- **O orçamento de ajuste é assimétrico.** A P6 foi desenvolvida sobre o nosso reconhecedor, com cerca de 2.300
  avaliações. O X02 quase dobrou a busca do KenLM, e a diferença no teste continuou.
- **Próximos passos:** X01 (referência humana), X03 (Transkribus) e X04 (pré-processamento igual no treino e na
  leitura).
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--e19-book', type=Path, help='E19 book dir (images/, references/); omit to keep demo_data')
    ap.add_argument('--e19-runs', type=Path, help='E19 runs dir (default/, lex/); omit to keep demo_data')
    ap.add_argument('--out', type=Path, default=ROOT / 'notebooks/demo_artigo.ipynb')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    picks = pick_pages()
    logger.info('pages: %s', picks)
    if args.e19_book and args.e19_runs:
        copy_atas(args.e19_book, args.e19_runs, picks['atas'])
    elif not all((L.DEMO / 'livro_atas' / f'{p}.jpg').exists() for p in picks['atas']):
        ap.error('notebooks/demo_data/livro_atas is missing the chosen pages; pass --e19-book and --e19-runs')
    nbf.write(build(picks), args.out)
    logger.info('wrote %s', args.out)


if __name__ == '__main__':
    main()
