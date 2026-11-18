"""X01: build the human-transcription kit for an independent diplomatic reference.

Pages: the 10 test pages (never in training, labelling or selection; today scored only against Gemini) and the 3
held-out pages (official GT only). For each page the kit holds:
- page.jpg: the page photograph;
- revisao.html: the page image, then each segmented line crop (v6 segmentation), to help locate lines; with
  --aids machine also the readings of R1·P6 and R4·PA (off by default: they anchor the annotator on the systems the
  reference will score);
- transcricao.txt: the file the annotator writes, one line per manuscript line. Empty by default (--prefill none);
  --prefill gemini copies the Gemini text as a draft to post-edit (faster, but anchors on the teacher; the README of
  the kit says which was used, and the paper must report it).
Machine texts come from committed records (records/e20/eval/lines); crops from outputs/line_crops (not in git).

  python tools/x01_make_kit.py --out outputs/x01_kit --prefill none
"""
import argparse
import html
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)
TEST = [18, 22, 23, 24, 25, 31, 32, 34, 44, 55]
HELDOUT = [7, 10, 12]
BOOK = 'BR FPMSC FUN CAM LDL - Livro de Registro de Leis n° 1, 1892 - 1984'


def sources(page: int) -> dict:
    """Paths of the image, references, crops and machine lines of one page."""
    if page in TEST:
        return {'image': Path(f'data/extra/teste/imagens/{page}.jpg'),
                'gemini': Path(f'data/extra/teste/referencias/{page}.txt'), 'oficial': None,
                'crops': Path('outputs/line_crops/test10'), 'set': 'test10'}
    return {'image': Path(f'data/images/{page}.{BOOK}.jpg'),
            'gemini': Path(f'data/ground_truth_gemini/{page}.{BOOK}.txt'),
            'oficial': Path(f'data/ground_truth/{page}.{BOOK}.txt'),
            'crops': Path('outputs/line_crops/gt12'), 'set': 'heldout'}


def machine_lines(s: str, page: int) -> dict[str, list[str]]:
    out = {}
    for name, (rec, dec) in {'R1·P6 (nosso)': ('r1_st_c1m', 'p6'), 'R4·PA': ('r4_traprinq', 'pa')}.items():
        data = json.loads(Path(f'records/e20/eval/lines/{s}/{rec}/{dec}.json').read_text())
        out[name] = data.get(str(page), [])
    return out


def page_html(page: int, src: dict, crops: list[dict], machine: dict[str, list[str]]) -> str:
    rows = []
    for c in crops:
        reads = ''.join(f'<div class="r"><b>{html.escape(k)}:</b> {html.escape(v[c["line"]] if c["line"] < len(v) else "")}</div>'
                        for k, v in machine.items())
        rows.append(f'<div class="ln"><div class="n">linha segmentada {c["line"] + 1}</div>'
                    f'<img src="linhas/{Path(c["file"]).name}">{reads}</div>')
    return f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Página {page}</title>
<style>body{{font-family:Georgia,serif;margin:16px;max-width:1300px;background:#fff;color:#111}}
img{{max-width:100%}} .ln{{border-top:1px solid #ccc;padding:8px 0}} .n{{font:12px sans-serif;color:#666}}
.r{{font-size:15px;margin:2px 0}} b{{font:12px sans-serif;color:#555}}</style>
<h1>Página {page}</h1>
<p>Transcreva em <code>transcricao.txt</code> lendo a <b>imagem</b>, conforme o LEIAME. Os recortes abaixo servem só para localizar as linhas. A segmentação pode juntar, partir ou omitir linhas.</p>
<p><a href="page.jpg">abrir a página inteira</a></p>
<img src="page.jpg">
{''.join(rows)}
</html>'''


def build_page(page: int, out: Path, prefill: str, aids: str) -> None:
    src = sources(page)
    d = out / f'pagina_{page:02d}'
    (d / 'linhas').mkdir(parents=True, exist_ok=True)
    shutil.copy(src['image'], d / 'page.jpg')
    manifest = json.loads((src['crops'] / 'manifest.json').read_text())['pages'][str(page)]
    for c in manifest:
        shutil.copy(src['crops'] / c['file'], d / 'linhas' / Path(c['file']).name)
    (d / 'revisao.html').write_text(page_html(page, src, manifest, machine_lines(src['set'], page) if aids == 'machine' else {}))
    draft = src['gemini'].read_text() if prefill == 'gemini' else ''
    (d / 'transcricao.txt').write_text(draft)
    logger.info('page %d: %d crops, prefill %s', page, len(manifest), prefill)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prefill', choices=['none', 'gemini'], default='none')
    parser.add_argument('--aids', choices=['none', 'machine'], default='none',
                        help='machine: show R1·P6 and R4·PA readings under each crop (anchors the annotator on the '
                             'systems under test; off by default)')
    parser.add_argument('--pages', default=','.join(map(str, TEST + HELDOUT)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    for p in map(int, args.pages.split(',')):
        build_page(p, args.out, args.prefill, args.aids)


if __name__ == '__main__':
    main()
