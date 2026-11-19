"""E18: degrade the 12 GT pages like a different scan / paper, to test adaptive preprocessing with known ground truth.

Each shift is a deterministic function of the page (seeded per page and shift). Output, one folder per shift:
  OUT/<shift>/images/<same file name>.jpg   OUT/<shift>/references/<same>.txt   OUT/<shift>/shift.json
The page numbering and the tuning / held-out split are unchanged, so `ocr_pipeline.run --pages heldout` works on it.
Keep one cache dir per shift: the pipeline's cache keys use the image file name, not its content.

  python tools/synth_shift.py --out outputs/e18/shifts
"""
import argparse
import hashlib
import io
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)
Shift = Callable[[Image.Image, np.random.Generator, Image.Image], Image.Image]


def _arr(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert('RGB')).astype(np.float32)


def _img(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def low_contrast_cast(image: Image.Image, rng: np.random.Generator, other: Image.Image) -> Image.Image:
    """Grey-brown paper and washed-out ink: blend 45% toward a tan colour."""
    return _img(0.55 * _arr(image) + 0.45 * np.array([196, 168, 112], np.float32))


def uneven_light(image: Image.Image, rng: np.random.Generator, other: Image.Image) -> Image.Image:
    """A lit corner and a dark gutter: multiplicative linear gradient times a vignette."""
    a = _arr(image)
    h, w = a.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(angle) * x / w + np.sin(angle) * y / h)
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
    vignette = 1 - 0.35 * (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 2
    return _img(a * (0.45 + 0.55 * ramp)[..., None] * vignette[..., None])


def low_res_blur(image: Image.Image, rng: np.random.Generator, other: Image.Image) -> Image.Image:
    """A lower-resolution, slightly out-of-focus capture: 70% of the width and height, Gaussian blur (at 50% the
    already small source scans become unreadable for every chain: smoke test, 2026-11-08)."""
    small = image.resize((round(image.width * 0.7), round(image.height * 0.7)), Image.Resampling.BICUBIC)
    return small.filter(ImageFilter.GaussianBlur(0.6))


def faded_ink(image: Image.Image, rng: np.random.Generator, other: Image.Image) -> Image.Image:
    """Faded iron-gall ink: ink darkness relative to the paper cut to 40%."""
    a = _arr(image)
    paper = np.percentile(a.reshape(-1, 3), 90, axis=0)
    return _img(paper - 0.4 * (paper - a))


def jpeg_noise(image: Image.Image, rng: np.random.Generator, other: Image.Image) -> Image.Image:
    """Sensor noise then heavy JPEG compression (quality 20)."""
    noisy = _img(_arr(image) + rng.normal(0, 14, (image.height, image.width, 3)))
    buffer = io.BytesIO()
    noisy.save(buffer, 'JPEG', quality=20)
    return Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')


def bleed_through(image: Image.Image, rng: np.random.Generator, other: Image.Image) -> Image.Image:
    """Ink of the reverse side showing through: another page, mirrored, darkening the paper at 30%."""
    back = _arr(other.resize(image.size).transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    a = _arr(image)
    return _img(np.minimum(a, a * (1 - 0.3 * (1 - back / 255))))


def identity(image: Image.Image, rng: np.random.Generator, other: Image.Image) -> Image.Image:
    """No shift (control: the adaptation run on our own book should keep the default and do no harm)."""
    return image


SHIFTS: dict[str, Shift] = {'identity': identity, 'low_contrast_cast': low_contrast_cast, 'uneven_light': uneven_light,
                            'low_res_blur': low_res_blur, 'faded_ink': faded_ink, 'jpeg_noise': jpeg_noise,
                            'bleed_through': bleed_through}


def seed_for(name: str, shift: str) -> int:
    return int(hashlib.sha256(f'{shift}/{name}'.encode()).hexdigest()[:8], 16)


def write_shift(shift: str, images: list[Path], references: Path, out: Path) -> None:
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'references').mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(images):
        other = images[(i + 1) % len(images)]  # the bleed-through source: the next page, fixed
        with Image.open(path) as page, Image.open(other) as back:
            result = SHIFTS[shift](page.convert('RGB'), np.random.default_rng(seed_for(path.name, shift)), back)
        result.save(out / 'images' / path.name, 'JPEG', quality=95)
        shutil.copy(references / f'{path.stem}.txt', out / 'references' / f'{path.stem}.txt')
    (out / 'shift.json').write_text(json.dumps({'shift': shift, 'doc': SHIFTS[shift].__doc__,
                                                'pages': [p.name for p in images]}, ensure_ascii=False, indent=1))
    logger.info('%s: %d pages', shift, len(images))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=Path, default=Path('data/images'))
    parser.add_argument('--references', type=Path, default=Path('data/ground_truth'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shifts', nargs='+', default=sorted(SHIFTS), choices=sorted(SHIFTS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    images = sorted(args.images.glob('*.jpg'), key=lambda p: int(p.name.split('.')[0]))
    for shift in args.shifts:
        write_shift(shift, images, args.references, args.out / shift)


if __name__ == '__main__':
    main()
