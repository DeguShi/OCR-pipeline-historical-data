"""Compare two versions of the automatic crop without segmentation: an old crop (--base: the first E19 crop, commit
32eaf79, or crop v2, commit 30a7e59) against the current defaults.

  pagexml  scans with PAGE XML (TraPrInq): share of the XML TextLine boxes inside each crop (>= 90% of the box area)
  photos   raw photos without line GT (book 1, book 2): whether the new box contains the old one (then every line
           inside the old crop is still inside), how much the area grows, and overlays of every changed photo

  python tools/crop_compare.py pagexml --xml TRAPRINQ/page --out outputs/e19/crop_v2/traprinq
  python tools/crop_compare.py photos --raw data/extra/ldl1 --out outputs/e19/crop_v2/book1
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_pipeline.auto_crop import Box, CropParams, auto_crop_box  # noqa: E402
from tools.crop_eval import INSIDE, inside_share, xml_lines  # noqa: E402

logger = logging.getLogger('crop_compare')
BASES = {'first': CropParams(edge_min_side=0, paper_closing=False, paper_gap=0.0),  # first E19 crop
         'v2': CropParams(paper_gap=0.0)}  # crop v2 (commit 30a7e59)


def contains(outer: Box, inner: Box) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlay(image: Image.Image, old: Box, new: Box, lines: list[Box], path: Path) -> None:
    """Red old box, blue new box, magenta lines outside the new box; 600 px tall."""
    canvas = image.convert('RGB')
    draw, lw = ImageDraw.Draw(canvas), max(3, image.width // 300)
    for line in lines:
        draw.rectangle(line, outline=(255, 0, 255), width=lw)
    draw.rectangle(old, outline=(255, 0, 0), width=2 * lw)
    draw.rectangle(new, outline=(0, 120, 255), width=lw)
    scale = 600 / canvas.height
    canvas.resize((int(canvas.width * scale), 600)).save(path, quality=85)


def compare(image: Image.Image, base: str) -> tuple[Any, Any]:
    return auto_crop_box(image, BASES[base])[0], auto_crop_box(image)[0]


def run_pagexml(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for xml in sorted(args.xml.glob('*.xml')):
        name, lines = xml_lines(xml)
        image = Image.open(xml.parent.parent / name)
        old, new = compare(image, args.base)
        s_old, s_new = [inside_share(l, old.box) for l in lines], [inside_share(l, new.box) for l in lines]
        row = {'image': name, 'n_lines': len(lines), 'inside_old': sum(s >= INSIDE for s in s_old),
               'inside_new': sum(s >= INSIDE for s in s_new), 'old': list(old.box), 'new': list(new.box),
               'contains': contains(new.box, old.box), 'watermark': new.watermark is not None,
               'area_ratio': round(area(new.box) / max(area(old.box), 1), 3)}
        rows.append(row)
        if row['inside_new'] != row['inside_old'] or row['inside_new'] < row['n_lines']:
            outside = [l for l, s in zip(lines, s_new) if s < INSIDE]
            overlay(image, old.box, new.box, outside, args.out / f'{Path(name).stem}.jpg')
        logger.info('%s', row)
    logger.info('pagexml: %d pages, lines %d, inside old %d, new %d', len(rows), sum(r['n_lines'] for r in rows),
                sum(r['inside_old'] for r in rows), sum(r['inside_new'] for r in rows))
    return rows


def run_photos(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(args.raw.glob('*.jpg')):
        image = Image.open(path)
        old, new = compare(image, args.base)
        row = {'photo': path.name, 'old': list(old.box), 'new': list(new.box), 'contains': contains(new.box, old.box),
               'area_ratio': round(area(new.box) / max(area(old.box), 1), 3)}
        rows.append(row)
        if row['old'] != row['new']:
            overlay(image, old.box, new.box, [], args.out / f'{path.stem[:40]}.jpg')
        logger.info('%s', row)
    logger.info('photos: %d, changed %d, new box not containing old %d', len(rows),
                sum(r['old'] != r['new'] for r in rows), sum(not r['contains'] for r in rows))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('pagexml', 'photos'))
    parser.add_argument('--xml', type=Path, help='pagexml: folder of PAGE XML files (images in the parent folder)')
    parser.add_argument('--raw', type=Path, help='photos: folder of raw photos')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', choices=list(BASES), default='first', help='the old crop the current one is compared with')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = run_pagexml(args) if args.mode == 'pagexml' else run_photos(args)
    (args.out / f'crop_compare_{args.mode}.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                                             encoding='utf-8')


if __name__ == '__main__':
    main()
