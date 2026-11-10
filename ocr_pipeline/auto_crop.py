"""Automatic page crop for archive photos: mask the archive watermark, keep the paper, trim empty margins, never cut ink.

The fixed box of page_crop.py only fits the FPMSC photos of book 1. This crop works from the photo itself:
  1. watermark: the Pró-Memória stamp (assets/fpmsc_watermark.png) is found by multi-scale template matching
     (normalized cross-correlation, so it survives the stamp's transparency over paper or background) in the top
     third of the photo; if the score passes `wm_threshold` its box is painted with the median colour of a ring
     around it. Book 1 has the stamp above the page, book 2 on the page's top-left margin.
  2. paper: Otsu threshold on the blurred HSV value channel, morphological opening, largest connected component;
     its bounding box drops the dark table, book edges and the background band.
  3. ink: black-hat (dark strokes on local background) inside the paper box, Otsu, minus long horizontal and
     vertical lines (ruling, red margin), minus specks and components touching the paper border (edge shadows).
     The crop is the union box of what is left plus a margin, clipped to the paper box. Trimming only ever removes
     rows/columns without ink, so a stain can enlarge the crop but no stroke is cut.
Validation: tools/crop_eval.py (hand crops of book 1, PAGE XML lines of TraPrInq, segmented lines of book 2).

  python -m ocr_pipeline.page_crop RAW_DIR CROP_DIR --box auto
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Box = tuple[int, int, int, int]  # left, top, right, bottom in pixels
WATERMARK = Path(__file__).resolve().parent / 'assets' / 'fpmsc_watermark.png'


@dataclass(frozen=True)
class CropParams:
    watermark: Path | None = WATERMARK
    wm_threshold: float = 0.55  # book 2 stamps score 0.80-0.92, book 1 0.70, hand crops without stamp <= 0.33
    wm_scales: tuple[float, ...] = tuple(round(0.75 + 0.05 * i, 2) for i in range(12))  # 0.75 .. 1.30
    margin_frac: float = 0.015  # of the paper box's longer side
    min_speck_px: int = 12
    rule_frac: float = 0.25  # a horizontal/vertical run this long (share of the paper side) is a rule, not ink


@dataclass
class CropResult:
    box: Box
    paper: Box
    ink: Box | None
    watermark: Box | None
    watermark_score: float
    size: tuple[int, int] = field(default=(0, 0))

    def to_dict(self) -> dict:
        return {'box': list(self.box), 'paper': list(self.paper), 'ink': list(self.ink) if self.ink else None,
                'watermark': list(self.watermark) if self.watermark else None,
                'watermark_score': round(self.watermark_score, 3), 'size': list(self.size)}


def find_watermark(gray: np.ndarray, params: CropParams) -> tuple[Box | None, float]:
    if params.watermark is None or not params.watermark.is_file():
        return None, 0.0
    template = cv2.imread(str(params.watermark), cv2.IMREAD_GRAYSCALE)
    band = gray[: gray.shape[0] // 3]
    best: tuple[float, Box | None] = (0.0, None)
    for scale in params.wm_scales:
        t = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= band.shape[0] or t.shape[1] >= band.shape[1]:
            continue
        _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED))
        if score > best[0]:
            best = (float(score), (x, y, x + t.shape[1], y + t.shape[0]))
    score, box = best
    return (box if score >= params.wm_threshold else None), score


def paint_out(rgb: np.ndarray, box: Box, ring: int = 6) -> np.ndarray:
    """Fill box with the per-channel median of a ring of pixels around it."""
    left, top, right, bottom = box
    h, w = rgb.shape[:2]
    outer = rgb[max(top - ring, 0):min(bottom + ring, h), max(left - ring, 0):min(right + ring, w)]
    mask = np.ones(outer.shape[:2], bool)
    mask[top - max(top - ring, 0):bottom - max(top - ring, 0), left - max(left - ring, 0):right - max(left - ring, 0)] = False
    out = rgb.copy()
    out[top:bottom, left:right] = np.median(outer[mask], axis=0).astype(rgb.dtype)
    return out


def longest_run(flags: np.ndarray) -> tuple[int, int]:
    """[start, end) of the longest run of True values (whole range if none)."""
    best, start = (0, len(flags)), None
    best_len = 0
    for i, f in enumerate(np.append(flags, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start > best_len:
                best, best_len = (start, i), i - start
            start = None
    return best


def paper_box(rgb: np.ndarray, exclude: Box | None = None, min_share: float = 0.5) -> Box:
    """Rows and columns where at least min_share of the pixels are paper (Otsu on blurred lightness); the longest
    run of each. Profiles ignore thin bright strips (page-stack edges) and the watermark box (exclude)."""
    light = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0], (9, 9), 0)
    paper = cv2.threshold(light, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1].astype(bool)
    if exclude is not None:
        paper[exclude[1]:exclude[3], exclude[0]:exclude[2]] = False
    top, bottom = longest_run(paper.mean(axis=1) >= min_share)
    left, right = longest_run(paper[top:bottom].mean(axis=0) >= min_share)
    return int(left), int(top), int(right), int(bottom)


def ink_mask(gray: np.ndarray, params: CropParams) -> np.ndarray:
    k = max(7, (min(gray.shape) // 45) | 1)
    hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ink = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    h, w = ink.shape
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(3, int(params.rule_frac * w))), np.uint8))
    rules |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(3, int(params.rule_frac * h)), 1), np.uint8))
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(rules, np.ones((3, 3), np.uint8))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    x, y, cw, ch, area = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3], stats[:, 4]
    touches = (x <= 1) | (y <= 1) | (x + cw >= w - 1) | (y + ch >= h - 1)
    keep = (area >= params.min_speck_px) & ~touches
    keep[0] = False
    return keep[labels]


def ink_box(gray: np.ndarray, params: CropParams) -> Box | None:
    mask = ink_mask(gray, params)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def auto_crop_box(image: Image.Image, params: CropParams = CropParams()) -> tuple[CropResult, np.ndarray]:
    """Crop box of one photo and the photo with the watermark painted out (RGB array)."""
    rgb = np.asarray(image.convert('RGB'))
    wm, wm_score = find_watermark(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), params)
    paper = paper_box(rgb, wm)
    if wm is not None:
        rgb = paint_out(rgb, wm)
    left, top, right, bottom = paper
    gray = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    ink = ink_box(gray, params)
    if ink is None:
        box = paper
    else:
        m = int(params.margin_frac * max(right - left, bottom - top))
        ink = (ink[0] + left, ink[1] + top, ink[2] + left, ink[3] + top)
        box = (max(ink[0] - m, left), max(ink[1] - m, top), min(ink[2] + m, right), min(ink[3] + m, bottom))
    return CropResult(box, paper, ink, wm, wm_score, image.size), rgb


def auto_crop(image: Image.Image, params: CropParams = CropParams()) -> tuple[Image.Image, CropResult]:
    result, rgb = auto_crop_box(image, params)
    left, top, right, bottom = result.box
    return Image.fromarray(rgb[top:bottom, left:right]), result
