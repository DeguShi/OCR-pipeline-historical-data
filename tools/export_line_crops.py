"""Export one image crop per recognized line, index-aligned with the exported posteriors, for image-informed
post-correctors (VLM baseline). Reads only the disk cache (segmentation) and the posterior export; no model runs.

Each line is cropped from the page after the configuration's geometric operations (the same image the
segmentation polygons refer to), in colour, at the bounding box of its boundary polygon plus --pad pixels.
The manifest pairs every crop with the recognizer's greedy text for that line (greedy_i of the posterior
export), which is the noisy OCR text the post-corrector receives.

  python tools/export_line_crops.py --config configs/final_v6.json --pages all \
      --posteriors outputs/posteriors/v6_gt12 --out outputs/line_crops/gt12
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_pipeline.config import load_config  # noqa: E402  (path set above)
from ocr_pipeline.data import discover, select  # noqa: E402
from ocr_pipeline.engine import Engine, filter_lines  # noqa: E402
from ocr_pipeline.pipeline import cache_keys  # noqa: E402
from ocr_pipeline.preprocessing import apply_chain  # noqa: E402

logger = logging.getLogger(__name__)


def line_boxes(engine: Engine, page, config) -> tuple[Image.Image, list[tuple[int, int, int, int] | None]]:
    seg_key, _ = cache_keys(page, config)
    seg_file = engine.cache_dir / 'seg' / f'{seg_key}.pkl'
    if not seg_file.is_file():
        raise FileNotFoundError(f'segmentation not cached for page {page.number}: {seg_file}')
    with Image.open(page.image) as original:
        base = apply_chain(original.convert('RGB'), config.geometric_ops, config.params, geometric=True)
    seg_image = apply_chain(base, config.seg_ops, config.params, geometric=False)
    segmentation = filter_lines(engine.segment(seg_image, config.seg, seg_key), config.seg, *base.size)
    boxes = []
    for line in segmentation.lines:
        poly = np.asarray(line.boundary if line.boundary else line.baseline, dtype=float)
        boxes.append(None if poly.size == 0 else (int(poly[:, 0].min()), int(poly[:, 1].min()),
                                                   int(poly[:, 0].max()), int(poly[:, 1].max())))
    return base, boxes


def export_page(engine: Engine, page, config, posteriors: Path, out: Path, pad: int) -> list[dict]:
    data = np.load(posteriors / f'page_{page.number:02d}.npz')
    greedy = [str(data[f'greedy_{i}']) for i in range(int(data['n_lines']))]
    base, boxes = line_boxes(engine, page, config)
    if len(boxes) != len(greedy):
        raise ValueError(f'page {page.number}: {len(boxes)} segmented lines vs {len(greedy)} posterior lines')
    folder = out / f'page_{page.number:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (box, text) in enumerate(zip(boxes, greedy)):
        row = {'line': i, 'ocr': text, 'file': None, 'bbox': None}
        if box is not None:
            x0, y0, x1, y1 = box
            bbox = (max(0, x0 - pad), max(0, y0 - pad), min(base.width, x1 + pad), min(base.height, y1 + pad))
            name = f'line_{i:03d}.png'
            base.crop(bbox).save(folder / name)
            row.update(file=f'{folder.name}/{name}', bbox=list(bbox))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pages', default='all')
    parser.add_argument('--images', type=Path, default=Path('data/images'))
    parser.add_argument('--references', type=Path, default=Path('data/ground_truth'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--posteriors', type=Path, required=True, help='posterior export of the same pages')
    parser.add_argument('--pad', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_config(args.config)
    engine = Engine(args.cache, 'cpu', 42)
    manifest = {}
    for page in select(discover(args.images, args.references), args.pages):
        manifest[str(page.number)] = export_page(engine, page, config, args.posteriors, args.out, args.pad)
        logger.info('page %d: %d lines', page.number, len(manifest[str(page.number)]))
    (args.out / 'manifest.json').write_text(json.dumps(
        {'config': str(args.config), 'posteriors': str(args.posteriors), 'pad': args.pad, 'pages': manifest},
        ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
