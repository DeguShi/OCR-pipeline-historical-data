"""E18 segmentation check: does the chain chosen on one sample page give a better LINE SEGMENTATION on the other pages
of a new book? Measured against the TraPrInq PAGE XML baselines, for the 8 segmentation candidates of the E18
search (tools/adapt_preprocess.py), on the evaluation pages of each case file (tools/traprinq_prepare.py manifest).

The pipeline segments the geometric-transformed image (config.geometric_ops: upscale, deskew), then filters lines
(engine.filter_lines); the GT baselines get the same transform (scale, then the same deskew rotation about the
centre). A predicted baseline matches a GT baseline when it covers >= 50% of the GT's x-span and their mean vertical
distance over the overlap is <= 0.5 x the page's median GT line spacing; matching is one-to-one, greedy by distance.
Per page: n_gt, n_pred, matched (precision = matched/n_pred, recall = matched/n_gt) and covered (GT lines with any
prediction within tolerance over >= 50% of their span, so split lines still count).

  python tools/seg_eval.py --books outputs/traprinq --page-xml DIR --chosen records/e18/books --out records/e18/seg
"""
import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt_preprocess import apply_candidate, seg_candidates  # noqa: E402  (path set above)
from ocr_pipeline.config import PipelineConfig, load_config  # noqa: E402
from ocr_pipeline.data import Page  # noqa: E402
from ocr_pipeline.engine import Engine, filter_lines  # noqa: E402
from ocr_pipeline.pipeline import cache_keys  # noqa: E402
from ocr_pipeline.preprocessing import apply_chain, apply_geometric, deskew_angle  # noqa: E402

logger = logging.getLogger(__name__)
Line = np.ndarray  # (n, 2) baseline points, x increasing


def gt_baselines(xml_path: Path) -> list[Line]:
    root = ET.parse(xml_path).getroot()
    ns = {'p': root.tag.split('}')[0][1:]}
    out = []
    for base in root.iterfind('.//p:TextLine/p:Baseline', ns):
        pts = np.array([[float(v) for v in xy.split(',')] for xy in base.get('points').split()])
        if len(pts) >= 2:
            out.append(pts[np.argsort(pts[:, 0])])
    return out


def transform(image: Image.Image, lines: list[Line], config: PipelineConfig) -> tuple[Image.Image, list[Line]]:
    """Apply the config's geometric ops to the image and the same mapping to the GT points."""
    for op in config.geometric_ops:
        w, h = image.size
        if op == 'upscale':
            image = apply_geometric(image, op, config.params)
            sx, sy = image.size[0] / w, image.size[1] / h
            lines = [pts * [sx, sy] for pts in lines]
        elif op == 'deskew':
            theta = np.radians(deskew_angle(image, config.params))
            image = apply_geometric(image, op, config.params)
            c, s, cx, cy = np.cos(theta), np.sin(theta), w / 2, h / 2
            lines = [np.column_stack([cx + (p[:, 0] - cx) * c + (p[:, 1] - cy) * s,
                                      cy - (p[:, 0] - cx) * s + (p[:, 1] - cy) * c]) for p in lines]
        else:
            raise ValueError(f'seg_eval cannot map geometric op {op}')
    return image, lines


def distance(g: Line, p: Line) -> tuple[float, float]:
    """(share of g's x-span covered by p, mean |dy| over the overlap)."""
    lo, hi = max(g[0, 0], p[0, 0]), min(g[-1, 0], p[-1, 0])
    span = max(g[-1, 0] - g[0, 0], 1.0)
    if hi <= lo:
        return 0.0, float('inf')
    xs = np.linspace(lo, hi, 20)
    dy = np.abs(np.interp(xs, g[:, 0], g[:, 1]) - np.interp(xs, p[:, 0], p[:, 1]))
    return (hi - lo) / span, float(dy.mean())


def match(gt: list[Line], pred: list[Line]) -> dict[str, int]:
    ys = sorted(float(g[:, 1].mean()) for g in gt)
    tol = 0.5 * float(np.median(np.diff(ys))) if len(ys) > 1 else 20.0
    pairs = []
    covered = 0
    for i, g in enumerate(gt):
        near = [(j, *distance(g, p)) for j, p in enumerate(pred)]
        near = [(j, ov, dy) for j, ov, dy in near if dy <= tol]
        covered += sum(ov for _, ov, _ in near) >= 0.5
        pairs += [(dy, i, j) for j, ov, dy in near if ov >= 0.5]
    used_g, used_p = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
    return {'n_gt': len(gt), 'n_pred': len(pred), 'matched': len(used_g), 'covered': covered}


def eval_page(engine: Engine, page: Page, xml: Path, config: PipelineConfig, viz: Path | None = None) -> dict[str, int]:
    with Image.open(page.image) as original:
        base, gt = transform(original.convert('RGB'), gt_baselines(xml), config)
    seg_key, _ = cache_keys(page, config)
    seg_image = apply_chain(base, config.seg_ops, config.params, geometric=False)
    seg = filter_lines(engine.segment(seg_image, config.seg, seg_key), config.seg, *base.size)
    pred = [np.array(sorted(map(tuple, line.baseline))) for line in seg.lines if len(line.baseline) >= 2]
    if viz is not None:
        overlay(base, gt, pred).save(viz)
    return match(gt, [p.astype(float) for p in pred])


def overlay(base: Image.Image, gt: list[Line], pred: list[Line]) -> Image.Image:
    """GT baselines green, predicted red, on a half-size copy (visual check of the coordinate mapping)."""
    from PIL import ImageDraw
    img = base.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for lines, colour in ((gt, (0, 160, 0)), (pred, (220, 0, 0))):
        for p in lines:
            draw.line([tuple(map(float, q)) for q in p], fill=colour, width=4)
    return img.resize((img.width // 2, img.height // 2))


def chosen_names(adapt: Path, config: PipelineConfig) -> dict[str, str]:
    out = {}
    for f in sorted(adapt.glob('chosen_*.json')):
        c = json.loads(f.read_text())
        up = c['params']['upscale_factor']
        out[f.stem.removeprefix('chosen_')] = f'up{up:g}|{"+".join(c["seg_ops"]) or "none"}'
    out['default'] = f'up{config.params.upscale_factor:g}|{"+".join(config.seg_ops) or "none"}'
    return out


def xml_index(page_xml: Path) -> dict[str, Path]:
    """{image stem: PAGE XML path}, through each file's imageFilename."""
    out = {}
    for xml in page_xml.glob('*.xml'):
        root = ET.parse(xml).getroot()
        ns = {'p': root.tag.split('}')[0][1:]}
        out[Path(root.find('p:Page', ns).get('imageFilename')).stem] = xml
    return out


def run_case(engine: Engine, config: PipelineConfig, case: dict[str, Any], args: argparse.Namespace,
             xmls: dict[str, Path]) -> dict:
    folder = args.books / case['case'] / 'images'
    rows = []
    for cand in seg_candidates(config.params.upscale_factor):
        cfg = apply_candidate(config, cand)
        for n in case['eval']:
            image = next(folder.glob(f'{n}.*.jpg'))
            xml = xmls[image.stem.split('.', 1)[1]]
            page = Page(n, image, image)
            viz = args.out / f"overlay_{case['case']}_p{n}_{cand['name'].replace('|', '_')}.jpg" if n == case['eval'][0] else None
            rows.append({'case': case['case'], 'page': n, 'candidate': cand['name'], **eval_page(engine, page, xml, cfg, viz)})
            logger.info('%s p%d %s %s', case['case'], n, cand['name'], rows[-1])
    return {'rows': rows, 'chosen': chosen_names(args.chosen / case['case'] / 'adapt', config)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--books', type=Path, required=True)
    parser.add_argument('--page-xml', type=Path, required=True)
    parser.add_argument('--chosen', type=Path, required=True, help='folder with <case>/adapt/chosen_*.json')
    parser.add_argument('--config', type=Path, default=Path('configs/final_v6.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('outputs/seg_eval_cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-cases', type=int, default=0, help='smoke test: first N cases, first eval page')
    args = parser.parse_args()
    config = load_config(args.config)
    engine = Engine(args.cache, args.device, args.seed)
    cases = json.loads((args.books / 'manifest.json').read_text())['cases']
    if args.limit_cases:
        cases = [dict(c, eval=c['eval'][:1]) for c in cases[:args.limit_cases]]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'params.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    xmls = xml_index(args.page_xml)
    results = {c['case']: run_case(engine, config, c, args, xmls) for c in cases}
    (args.out / 'seg_eval.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
