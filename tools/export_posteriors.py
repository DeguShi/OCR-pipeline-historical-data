"""Export the recognizer's per-line CTC posteriors (top-k per frame, from the disk cache) so that external
decoders (pyctcdecode, CTCWordBeamSearch, ...) read exactly the same recognizer output as our post-correction.

Writes <out>/alphabet.json (label -> character, blank = 0, n_classes) and <out>/page_NN.npz with, per line i,
idx_i (T, k) int16, val_i (T, k) float16 probabilities and greedy_i (the best-path text).

  python tools/export_posteriors.py --config configs/final_v6.json --pages all --out outputs/posteriors/v6_gt12
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
from ocr_pipeline.pipeline import cache_keys, page_logits  # noqa: E402
from ocr_pipeline.preprocessing import apply_chain  # noqa: E402

logger = logging.getLogger(__name__)


def page_lines(engine: Engine, page, config) -> list:
    seg_key, rec_key = cache_keys(page, config)
    with Image.open(page.image) as original:
        base = apply_chain(original.convert('RGB'), config.geometric_ops, config.params, geometric=True)
    seg_image = apply_chain(base, config.seg_ops, config.params, geometric=False)
    segmentation = filter_lines(engine.segment(seg_image, config.seg, seg_key), config.seg, *base.size)
    rec_image = apply_chain(base, config.rec_ops, config.params, geometric=False)
    return page_logits(engine, base, rec_image, segmentation, config, rec_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pages', default='all')
    parser.add_argument('--images', type=Path, default=Path('data/images'))
    parser.add_argument('--references', type=Path, default=Path('data/ground_truth'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_config(args.config)
    engine = Engine(args.cache, args.device, args.seed)
    _, l2c, n_classes = engine.codec(config.rec.model)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'alphabet.json').write_text(json.dumps(
        {'l2c': {str(k): v for k, v in sorted(l2c.items())}, 'n_classes': n_classes, 'blank': 0,
         'config': str(args.config), 'model': config.rec.model}, ensure_ascii=False, indent=1))
    for page in select(discover(args.images, args.references), args.pages):
        lines = page_lines(engine, page, config)
        arrays = {}
        for i, line in enumerate(lines):
            arrays[f'idx_{i}'], arrays[f'val_{i}'] = line.top_idx, line.top_val
            arrays[f'greedy_{i}'] = np.array(line.prediction)
        np.savez_compressed(args.out / f'page_{page.number:02d}.npz', n_lines=len(lines), **arrays)
        logger.info('page %d: %d lines', page.number, len(lines))


if __name__ == '__main__':
    main()
