"""E18 figure material: for the sample page of every shift, the same region (a band of lines) as
  shifted input | segmentation image of the default chain | segmentation and recognition images of the chain the
  lexicon objective chose,
plus the clean page for reference. PNGs go to OUT/<shift>_<panel>.png and a LaTeX fragment to OUT/e18_panels.tex.

  python tools/e18_figure.py --root outputs/e18 --out records/e18/figures --page 6
"""
import argparse
import json
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_pipeline.config import from_dict, load_config  # noqa: E402  (path set above)
from ocr_pipeline.preprocessing import apply_chain  # noqa: E402

BAND = (0.05, 0.30, 0.95, 0.42)  # left, top, right, bottom as page fractions: a few text lines


def band(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((round(BAND[0] * w), round(BAND[1] * h), round(BAND[2] * w), round(BAND[3] * h)))


def panels(path: Path, default_cfg, chosen_cfg) -> dict[str, Image.Image]:
    with Image.open(path) as im:
        rgb = im.convert('RGB')
    out = {'input': band(rgb)}
    for name, cfg in (('default_seg', default_cfg), ('chosen_seg', chosen_cfg), ('chosen_rec', chosen_cfg)):
        base = apply_chain(rgb, cfg.geometric_ops, cfg.params, geometric=True)
        ops = cfg.seg_ops if name.endswith('seg') else cfg.rec_ops
        out[name] = band(apply_chain(base, ops, cfg.params, geometric=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--page', type=int, default=6)
    parser.add_argument('--objective', default='lex')
    args = parser.parse_args()
    default_cfg = load_config(Path('configs/final_v6.json'))
    args.out.mkdir(parents=True, exist_ok=True)
    tex = ['% Gerado por tools/e18_figure.py\n']
    clean = next(Path('data/images').glob(f'{args.page}.*.jpg'))
    with Image.open(clean) as im:
        band(im.convert('RGB')).save(args.out / 'clean_input.png')
    for shift_dir in sorted(p for p in args.root.iterdir() if (p / 'adapt' / f'chosen_{args.objective}.json').is_file()):
        chosen = from_dict(json.loads((shift_dir / 'adapt' / f'chosen_{args.objective}.json').read_text()))
        image = next((args.root / 'shifts' / shift_dir.name / 'images').glob(f'{args.page}.*.jpg'))
        for name, img in panels(image, default_cfg, chosen).items():
            img.save(args.out / f'{shift_dir.name}_{name}.png')
        tex.append(f'% {shift_dir.name}: seg [{"+".join(chosen.seg_ops)}] rec [{"+".join(chosen.rec_ops)}] '
                   f'upscale {chosen.params.upscale_factor:g}\n')
    (args.out / 'e18_panels.tex').write_text(''.join(tex), encoding='utf-8')
    print(f'panels written to {args.out}')


if __name__ == '__main__':
    main()
