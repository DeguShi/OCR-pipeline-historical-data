"""E18 Q3: turn TraPrInq case files (Zenodo 13986218, CC BY 4.0; Portuguese Inquisition records, Transkribus PAGE XML)
into pipeline-ready "books": one folder per case file with images/<i>.<stem>.jpg and references/<i>.<stem>.txt
(the page's TextLine Unicode in reading order, one line per line), numbered 1..n so ocr_pipeline.run can read them.

Cases with at least --min-pages pages are eligible; --cases of them are drawn with --seed (pages sorted by name).
The first page of each case (after a seeded shuffle) is the search sample; the rest are evaluation pages.

  python tools/traprinq_prepare.py --src /content/traprinq/val/1499414 --out outputs/traprinq --cases 4
"""
import argparse
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def page_text(xml_path: Path) -> tuple[str, str]:
    """(image file name, reference text) of one PAGE XML file."""
    root = ET.parse(xml_path).getroot()
    ns = {'p': root.tag.split('}')[0][1:]}
    page = root.find('p:Page', ns)
    lines = []
    for line in root.iterfind('.//p:TextLine', ns):
        text = line.find('p:TextEquiv/p:Unicode', ns)
        if text is not None and text.text and text.text.strip():
            lines.append(text.text.strip())
    return page.get('imageFilename'), '\n'.join(lines)


def cases(src: Path) -> dict[str, list[tuple[Path, str]]]:
    folder = next(src.glob('TRAINING_VALIDATION_SET_*_'))
    by_case: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for xml_path in sorted((folder / 'page').glob('*.xml')):
        image_name, text = page_text(xml_path)
        image = folder / image_name
        if image.is_file() and text:
            by_case[re.sub(r'_[mc]\d+.*$', '', image.stem)].append((image, text))
    return by_case


def write_case(name: str, pages: list[tuple[Path, str]], out: Path, rng: random.Random) -> dict:
    pages = sorted(pages, key=lambda p: p[0].name)
    order = pages.copy()
    rng.shuffle(order)
    (out / name / 'images').mkdir(parents=True, exist_ok=True)
    (out / name / 'references').mkdir(parents=True, exist_ok=True)
    numbered = {}
    for i, (image, text) in enumerate(pages, start=1):
        stem = f'{i}.{image.stem}'
        shutil.copy(image, out / name / 'images' / f'{stem}.jpg')
        (out / name / 'references' / f'{stem}.txt').write_text(text + '\n', encoding='utf-8')
        numbered[image.name] = i
    sample = numbered[order[0][0].name]
    return {'case': name, 'pages': len(pages), 'sample': sample,
            'eval': sorted(n for n in numbered.values() if n != sample)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--min-pages', type=int, default=4)
    parser.add_argument('--max-pages', type=int, default=8, help='skip very long cases (CPU budget)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    eligible = {k: v for k, v in sorted(cases(args.src).items()) if args.min_pages <= len(v) <= args.max_pages}
    chosen = sorted(rng.sample(sorted(eligible), min(args.cases, len(eligible))))
    manifest = [write_case(name, eligible[name], args.out, rng) for name in chosen]
    (args.out / 'manifest.json').write_text(json.dumps({'source': 'Zenodo 13986218 Validation_Set_FINAL',
                                                        'eligible': len(eligible), 'cases': manifest}, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
