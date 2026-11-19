"""E19: does the automatic page crop (ocr_pipeline.auto_crop) keep every text line? Three checks, one per mode.

  book1    raw FPMSC photos of book 1 vs the hand crops of the 12 GT pages (same photo number): the hand box is
           found by template matching at scale 1; reports IoU(auto, hand) and, for the Kraken lines segmented on the
           whole photo (watermark painted out), the share inside each box.
  pagexml  any scans with PAGE XML (TraPrInq): share of the XML TextLine boxes inside the auto box; also whether a
           watermark was (wrongly) detected.
  book2    raw photos of a new book: Kraken lines on the whole photo inside the auto box, and their number vs the
           transcript's lines; half-size overlays (blue paper, red crop, green lines) for a visual check.
A line counts as inside when >= 90% of its bounding-box area is in the crop.

  python tools/crop_eval.py book1 --raw data/extra/ldl1 --hand data/images --out outputs/e19/crop
  python tools/crop_eval.py pagexml --raw DIR_WITH_IMAGES --page-xml DIR/page --out outputs/e19/crop
  python tools/crop_eval.py book2 --raw PHOTOS --transcripts TXT --out outputs/e19/crop
"""
import argparse
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_pipeline.auto_crop import Box, auto_crop_box  # noqa: E402  (path set above)
from ocr_pipeline.config import load_config  # noqa: E402
from ocr_pipeline.engine import Engine, filter_lines  # noqa: E402
from ocr_pipeline.pipeline import stable_hash  # noqa: E402
from ocr_pipeline.preprocessing import apply_chain  # noqa: E402

logger = logging.getLogger(__name__)
INSIDE = 0.9


def iou(a: Box, b: Box) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / max(area(a) + area(b) - inter, 1)


def inside_share(line: Box, box: Box) -> float:
    inter = max(0, min(line[2], box[2]) - max(line[0], box[0])) * max(0, min(line[3], box[3]) - max(line[1], box[1]))
    return inter / max((line[2] - line[0]) * (line[3] - line[1]), 1)


def kraken_lines(engine: Engine, rgb: np.ndarray, config: Any, name: str) -> list[Box]:
    """Line boxes of the whole photo, segmented with the config's seg chain (no geometric ops) and line filter."""
    image = Image.fromarray(rgb)
    seg_image = apply_chain(image, config.seg_ops, config.params, geometric=False)
    seg = filter_lines(engine.segment(seg_image, config.seg, stable_hash(['crop_eval', name, config.seg.model])),
                       config.seg, *image.size)
    boxes = []
    for line in seg.lines:
        pts = np.asarray(line.boundary or line.baseline, float)
        boxes.append((int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())))
    return boxes


def line_stats(lines: list[Box], box: Box) -> dict[str, Any]:
    shares = [inside_share(l, box) for l in lines]
    return {'n_lines': len(lines), 'inside': sum(s >= INSIDE for s in shares),
            'min_share': round(min(shares), 3) if shares else None}


def overlay(rgb: np.ndarray, result: Any, lines: list[Box], path: Path) -> None:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for l in lines:
        draw.rectangle(l, outline=(0, 170, 0), width=2)
    draw.rectangle(result.paper, outline=(0, 0, 255), width=3)
    draw.rectangle(result.box, outline=(255, 0, 0), width=3)
    img.resize((img.width // 2, img.height // 2)).save(path, quality=85)


def photo_number(path: Path) -> int:
    return int(re.match(r'\d+', path.name).group())


def run_book1(args: argparse.Namespace, engine: Engine, config: Any) -> list[dict[str, Any]]:
    raws = {photo_number(p): p for p in args.raw.glob('*.jpg')}
    rows = []
    for hand_path in sorted(args.hand.glob('*.jpg'), key=photo_number):
        n = photo_number(hand_path)
        with Image.open(raws[n]) as raw:
            result, rgb = auto_crop_box(raw)
            raw_arr = np.asarray(raw.convert('RGB'))
        hand = np.asarray(Image.open(hand_path).convert('RGB'))
        _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(raw_arr, hand, cv2.TM_CCOEFF_NORMED))
        hand_box = (x, y, x + hand.shape[1], y + hand.shape[0])
        lines = kraken_lines(engine, rgb, config, f'book1_{n}')
        rows.append({'page': n, 'auto': list(result.box), 'hand': list(hand_box), 'hand_match': round(score, 3),
                     'iou': round(iou(result.box, hand_box), 3), 'watermark_score': round(result.watermark_score, 3),
                     'auto_lines': line_stats(lines, result.box), 'hand_lines': line_stats(lines, hand_box)})
        overlay(rgb, result, lines, args.out / f'book1_{n}.jpg')
        logger.info('%s', rows[-1])
    return rows


def xml_lines(path: Path) -> tuple[str, list[Box]]:
    root = ET.parse(path).getroot()
    ns = {'p': root.tag.split('}')[0][1:]}
    boxes = []
    for coords in root.iterfind('.//p:TextLine/p:Coords', ns):
        pts = np.array([[float(v) for v in xy.split(',')] for xy in coords.get('points').split()])
        boxes.append((int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())))
    return root.find('p:Page', ns).get('imageFilename'), boxes


def run_pagexml(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for xml in sorted(args.page_xml.glob('*.xml')):
        image_name, lines = xml_lines(xml)
        image = args.raw / image_name
        if not image.is_file() or not lines:
            continue
        with Image.open(image) as im:
            result, _ = auto_crop_box(im)
        rows.append({'image': image_name, 'auto': list(result.box), 'size': list(result.size),
                     'watermark': result.watermark is not None, 'watermark_score': round(result.watermark_score, 3),
                     **line_stats(lines, result.box)})
    logger.info('pagexml: %d pages, %d/%d lines inside, %d watermarks detected', len(rows),
                sum(r['inside'] for r in rows), sum(r['n_lines'] for r in rows), sum(r['watermark'] for r in rows))
    return rows


def run_book2(args: argparse.Namespace, engine: Engine, config: Any) -> list[dict[str, Any]]:
    rows = []
    for photo in sorted(args.raw.glob('*.jpg'), key=lambda p: (photo_number(p), p.name)):
        with Image.open(photo) as im:
            result, rgb = auto_crop_box(im)
        lines = kraken_lines(engine, rgb, config, f'book2_{photo.stem}')
        text = args.transcripts / f'{photo.stem}.txt'
        n_text = sum(1 for l in text.read_text(encoding='utf-8').splitlines() if l.strip()) if text.is_file() else None
        rows.append({'photo': photo.stem, 'auto': list(result.box), 'size': list(result.size),
                     'watermark_score': round(result.watermark_score, 3), 'transcript_lines': n_text,
                     **line_stats(lines, result.box)})
        overlay(rgb, result, lines, args.out / f'book2_{photo_number(photo)}_{len(rows)}.jpg')
        logger.info('%s', rows[-1])
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('book1', 'pagexml', 'book2'))
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--hand', type=Path, help='book1: hand-cropped GT images')
    parser.add_argument('--page-xml', type=Path, help='pagexml: folder of PAGE XML files')
    parser.add_argument('--transcripts', type=Path, help='book2: transcripts named like the photos')
    parser.add_argument('--config', type=Path, default=Path('configs/final_v6.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('outputs/e19/crop_cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'pagexml':
        rows = run_pagexml(args)
    else:
        config, engine = load_config(args.config), Engine(args.cache, args.device, args.seed)
        rows = (run_book1 if args.mode == 'book1' else run_book2)(args, engine, config)
    (args.out / f'crop_{args.mode}.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding='utf-8')
    logger.info('CROP EVAL %s DONE %d rows', args.mode, len(rows))


if __name__ == '__main__':
    main()
