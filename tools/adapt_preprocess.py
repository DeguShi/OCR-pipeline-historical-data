"""E18/E19: choose the preprocessing chain for a new book from one or more sample pages, without labels.

Candidates are combinations of the pipeline's own ops (ocr_pipeline/preprocessing.py). The search is coordinate-wise:
  stage 1 (segmentation image): upscale factor x seg_ops chain, recognition input left at the config's default;
  stage 2 (recognition image): rec_ops chain, on the stage-1 winner of each objective.
Each candidate is segmented and read greedily (no post-correction) and scored by label-free objectives:
  conf    mean top-1 CTC posterior over non-blank frames (how sure the recognizer is);
  lex     number of greedy words (2+ letters) found in the period lexicon; a count, not a share, so reading less
          is not rewarded (Ströbel et al. 2022 use lexicon hits as a GT-free
          HTR quality proxy);
  conflex conf x lex (expected number of confidently read lexicon words);
  stat    distance of the preprocessed page's statistics (paper and ink levels, contrast, noise, stroke width per line
          height, share of ink inside line polygons) to those of our book's training pages, where the pipeline is best
          ("bring the new book closer to our current best").
If --reference is given (a tuning page with GT), greedy WER/CER are recorded too, and the `oracle` objective
(lowest greedy CER) bounds what one-page selection can reach. One chosen config per objective is written.
With several --sample pages (E19) each candidate is run on all of them: counts are summed, means averaged, and
WER/CER are corpus rates over the sample pages; with one page the values are those of E18.

  python tools/adapt_preprocess.py reference --config configs/final_v6.json --images data/extra/treino_crop \
      --out outputs/e18/reference_stats.json
  python tools/adapt_preprocess.py search --config configs/final_v6.json --sample PAGE.jpg --reference PAGE.txt \
      --stats outputs/e18/reference_stats.json --out outputs/e18/<shift>/adapt --cache outputs/e18/<shift>/cache
"""
import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_pipeline.config import PipelineConfig, load_config  # noqa: E402  (path set above)
from ocr_pipeline.data import Page  # noqa: E402
from ocr_pipeline.engine import Engine, filter_lines  # noqa: E402
from ocr_pipeline.metrics import score  # noqa: E402
from ocr_pipeline.pipeline import cache_keys  # noqa: E402
from ocr_pipeline.preprocessing import apply_chain  # noqa: E402

logger = logging.getLogger(__name__)
Candidate = dict[str, Any]  # {'name', 'upscale', 'seg_ops'} or {'name', 'rec_ops', 'params'}
FEATURES = ('paper', 'ink', 'contrast', 'noise', 'stroke_rel', 'coverage')
OBJECTIVES = {'conf': ('conf', max), 'lex': ('lex', max), 'conflex': ('conflex', max), 'stat': ('stat', min),
              'oracle': ('cer_norm', min)}

SEG_CHAINS = [('contrast', 'remove_rules'), (), ('clahe', 'remove_rules'), ('illumination', 'remove_rules'),
              ('illumination', 'contrast', 'remove_rules'), ('denoise', 'contrast', 'remove_rules')]
REC_CHAINS: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    ((), {}), (('contrast',), {}), (('clahe',), {}), (('illumination',), {}), (('denoise',), {}),
    (('sharpen',), {}), (('gamma',), {'gamma': 0.7}), (('gamma',), {'gamma': 1.5}), (('channel',), {'channel': 'min'}),
    (('illumination', 'contrast'), {}), (('denoise', 'contrast'), {}), (('thicken',), {}), (('kraken_nlbin',), {})]


def seg_candidates(default_upscale: float) -> list[Candidate]:
    out = [{'name': f'up{u:g}|{"+".join(SEG_CHAINS[0])}', 'upscale': u, 'seg_ops': SEG_CHAINS[0]}
           for u in sorted({default_upscale, 1.0, 3.0})]
    return out + [{'name': f'up{default_upscale:g}|{"+".join(c) or "none"}', 'upscale': default_upscale,
                   'seg_ops': c} for c in SEG_CHAINS[1:]]


def rec_candidates() -> list[Candidate]:
    return [{'name': '+'.join(ops) + ''.join(f'[{k}={v}]' for k, v in p.items()) or 'none', 'rec_ops': ops,
             'params': p} for ops, p in REC_CHAINS]


def apply_candidate(config: PipelineConfig, cand: Candidate) -> PipelineConfig:
    if 'seg_ops' in cand:
        return replace(config, seg_ops=tuple(cand['seg_ops']),
                       params=replace(config.params, upscale_factor=cand['upscale']))
    return replace(config, rec_ops=tuple(cand['rec_ops']), params=replace(config.params, **cand['params']))


def load_lexicon(path: Path, size: int) -> set[str]:
    words = [line.split()[0] for line in path.read_text(encoding='utf-8').splitlines()[:size] if line.strip()]
    return {w.lower() for w in words}


def lexicon_hits(lines: list[str], lexicon: set[str]) -> tuple[int, int]:
    words = [w for line in lines for w in re.findall(r'[^\W\d_]{2,}', line.lower())]
    return sum(w in lexicon for w in words), len(words)


def confidence(logits: list[Any]) -> float:
    """Mean top-1 probability over frames whose top class is not the CTC blank (label 0)."""
    vals = [line.top_val[line.top_idx[:, 0] != 0, 0].astype(np.float32) for line in logits if len(line.top_idx)]
    vals = [v for v in vals if v.size]
    return float(np.concatenate(vals).mean()) if vals else 0.0


def ink_mask(gray: np.ndarray) -> np.ndarray:
    block = max(15, (min(gray.shape) // 20) | 1)
    return cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15)


def image_features(rec_image: Image.Image, base: Image.Image, segmentation: Any) -> dict[str, float]:
    gray = np.asarray(rec_image.convert('L'))
    paper, ink = np.percentile(gray, 90), np.percentile(gray, 3)
    mask = ink_mask(gray)
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 5)
    noise = float(np.mean(np.abs(residual[mask == 0])))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    stroke = 2 * mask.sum() / max(perimeter, 1)
    heights, poly = [], np.zeros(gray.shape, np.uint8)
    for line in (l for l in segmentation.lines if l.boundary):
        pts = np.asarray(line.boundary, np.int32)
        heights.append(np.ptp(pts[:, 1]))
        cv2.fillPoly(poly, [pts], 1)
    base_ink = ink_mask(np.asarray(base.convert('L')))
    coverage = float((base_ink & poly).sum() / max(base_ink.sum(), 1))
    stroke_rel = stroke / np.median(heights) if heights else 0.0
    return {'paper': float(paper), 'ink': float(ink), 'contrast': float(paper - ink), 'noise': noise,
            'stroke_rel': float(stroke_rel), 'coverage': coverage}


def stat_distance(features: dict[str, float], stats: dict[str, dict[str, float]]) -> float:
    return float(sum(abs(features[f] - stats[f]['mean']) / max(stats[f]['std'], 0.05 * abs(stats[f]['mean']), 1e-3)
                     for f in FEATURES))


def run_candidate(engine: Engine, page: Page, config: PipelineConfig) -> tuple[list[str], list[Any], dict[str, float]]:
    """Segment and read one page greedily with this config; return lines, logits and image features."""
    seg_key, rec_key = cache_keys(page, config)
    with Image.open(page.image) as original:
        base = apply_chain(original.convert('RGB'), config.geometric_ops, config.params, geometric=True)
    seg_image = apply_chain(base, config.seg_ops, config.params, geometric=False)
    segmentation = filter_lines(engine.segment(seg_image, config.seg, seg_key), config.seg, *base.size)
    rec_image = apply_chain(base, config.rec_ops, config.params, geometric=False)
    logits = engine.recognize_logits(rec_image, segmentation, config.rec, rec_key)
    return [line.prediction for line in logits], logits, image_features(rec_image, base, segmentation)


def evaluate_candidate(engine: Engine, page: Page, config: PipelineConfig, cand: Candidate, ctx: dict[str, Any],
                       reference: str | None = None) -> dict[str, Any]:
    start = time.time()
    lines, logits, feats = run_candidate(engine, page, config)
    conf, (hits, n_words) = confidence(logits), lexicon_hits(lines, ctx['lexicon'])
    row = {'stage': 'seg' if 'seg_ops' in cand else 'rec', 'name': cand['name'], 'n_lines': len(lines),
           'conf': conf, 'lex': hits, 'lex_share': hits / max(n_words, 1), 'conflex': conf * hits,
           'stat': stat_distance(feats, ctx['stats']), **feats}
    if reference is not None:
        s = score(reference, '\n'.join(lines))
        row.update({k: s[k] for k in ('wer_norm', 'cer_norm', 'norm_we', 'norm_rw', 'norm_ce', 'norm_rc')})
    row['seconds'] = round(time.time() - start, 1)
    logger.info('p%s %s %-45s lines %3d conf %.3f lex %4d stat %6.2f%s (%.0f s)', page.number, row['stage'], cand['name'],
                row['n_lines'], conf, hits, row['stat'],
                f" WER {100 * row['wer_norm']:.1f} CER {100 * row['cer_norm']:.1f}" if 'cer_norm' in row else '',
                row['seconds'])
    return row


SUMMED = ('n_lines', 'lex', 'conflex', 'norm_we', 'norm_rw', 'norm_ce', 'norm_rc', 'seconds')


def evaluate_pages(engine: Engine, pages: list[Page], config: PipelineConfig, cand: Candidate, ctx: dict[str, Any]
                   ) -> dict[str, Any]:
    """One candidate on every sample page: counts (lex, conflex, lines, edits) are summed, conf/stat/features averaged,
    WER/CER are corpus rates over the sample pages."""
    per_page = [evaluate_candidate(engine, p, config, cand, ctx, ctx['references'].get(p.number)) for p in pages]
    if len(per_page) == 1:
        return per_page[0]
    row = {k: per_page[0][k] for k in ('stage', 'name')}
    for key in per_page[0]:
        if key in row or key in ('wer_norm', 'cer_norm'):
            continue
        vals = [r[key] for r in per_page]
        row[key] = sum(vals) if key in SUMMED else float(np.mean(vals))
    if 'norm_rw' in row:
        row.update(wer_norm=row['norm_we'] / max(row['norm_rw'], 1), cer_norm=row['norm_ce'] / max(row['norm_rc'], 1))
    row['n_lines_min'] = min(r['n_lines'] for r in per_page)
    row['pages'] = [{'page': p.number, **{k: r.get(k) for k in ('n_lines', 'conf', 'lex', 'stat', 'cer_norm')}}
                    for p, r in zip(pages, per_page)]
    if ctx.get('partial'):  # one line per finished candidate, so a lost VM keeps what it computed
        with ctx['partial'].open('a', encoding='utf-8') as f:
            f.write(json.dumps({**row, 'seg_config': config.to_dict()['seg_ops']}) + '\n')
    return row


def pick(rows: list[dict[str, Any]], objective: str) -> int:
    key, best = OBJECTIVES[objective]
    usable = [i for i, r in enumerate(rows) if key in r and r.get('n_lines_min', r['n_lines']) > 0]
    return best(usable, key=lambda i: rows[i][key])


def search(engine: Engine, pages: list[Page], config: PipelineConfig, ctx: dict[str, Any], objectives: list[str]
           ) -> tuple[list[dict[str, Any]], dict[str, PipelineConfig]]:
    seg_cands = seg_candidates(config.params.upscale_factor)[:ctx['limit'] or None]
    seg_rows = [evaluate_pages(engine, pages, apply_candidate(config, c), c, ctx) for c in seg_cands]
    rows, chosen = list(seg_rows), {}
    done: dict[tuple[str, str], dict[str, Any]] = {}
    rec_cands = rec_candidates()[:ctx['limit'] or None]
    for objective in objectives:
        seg_cand = seg_cands[pick(seg_rows, objective)]
        seg_config = apply_candidate(config, seg_cand)
        rec_rows = []
        for cand in rec_cands:
            key = (seg_cand['name'], cand['name'])
            if key not in done:
                done[key] = {**evaluate_pages(engine, pages, apply_candidate(seg_config, cand), cand, ctx),
                             'seg_name': seg_cand['name']}
                rows.append(done[key])
            rec_rows.append(done[key])
        best = rec_cands[pick(rec_rows, objective)]
        chosen[objective] = apply_candidate(seg_config, best)
        logger.info('%s -> seg %s, rec %s', objective, seg_cand['name'], best['name'])
    return rows, chosen


def reference_stats(engine: Engine, config: PipelineConfig, images: list[Path]) -> dict[str, Any]:
    feats = []
    for path in images:
        _, _, f = run_candidate(engine, Page(0, path, path), config)
        feats.append(f)
        logger.info('%s %s', path.name, {k: round(v, 3) for k, v in f.items()})
    return {'pages': [p.name for p in images],
            **{k: {'mean': float(np.mean([f[k] for f in feats])), 'std': float(np.std([f[k] for f in feats]))}
               for k in FEATURES}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('reference', 'search'))
    parser.add_argument('--config', type=Path, default=Path('configs/final_v6.json'))
    parser.add_argument('--images', type=Path, help="reference: folder of our book's training pages")
    parser.add_argument('--n-pages', type=int, default=8, help='reference: pages sampled (seed 42)')
    parser.add_argument('--sample', type=Path, nargs='+', help='search: the sample page(s) of the new book (objective summed)')
    parser.add_argument('--reference', type=Path, nargs='+', default=None,
                        help='search: GT text of each sample page, same order (analysis/oracle only)')
    parser.add_argument('--stats', type=Path, help='search: output of the reference mode')
    parser.add_argument('--lexicon-size', type=int, default=100000)
    parser.add_argument('--objectives', nargs='+', default=list(OBJECTIVES), choices=list(OBJECTIVES))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('outputs/e18/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='smoke test: first N candidates per stage')
    return parser


def page_number(path: Path, default: int) -> int:
    return int(path.name.split('.')[0]) if path.name[0].isdigit() else default


def run_search(args: argparse.Namespace, engine: Engine, config: PipelineConfig) -> None:
    if args.reference is not None and len(args.reference) != len(args.sample):
        raise SystemExit('--reference needs one file per --sample page')
    objectives = [o for o in args.objectives if o != 'oracle' or args.reference is not None]
    refs = args.reference or [None] * len(args.sample)
    pages = [Page(page_number(s, i), s, r or s) for i, (s, r) in enumerate(zip(args.sample, refs))]
    ctx = {'limit': args.limit, 'lexicon': load_lexicon(Path(config.post.period_lexicon), args.lexicon_size),
           'stats': json.loads(args.stats.read_text()),
           'references': {p.number: r.read_text(encoding='utf-8') for p, r in zip(pages, refs) if r is not None}}
    args.out.mkdir(parents=True, exist_ok=True)
    ctx['partial'] = args.out / 'candidates_partial.jsonl'
    rows, chosen = search(engine, pages, config, ctx, objectives)
    (args.out / 'candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    for objective, cfg in chosen.items():
        (args.out / f'chosen_{objective}.json').write_text(json.dumps(cfg.to_dict(), indent=1), encoding='utf-8')
    (args.out / 'params.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_config(args.config)
    engine = Engine(args.cache, args.device, args.seed)
    if args.mode == 'reference':
        images = sorted(args.images.glob('*.jpg'))
        images = sorted(random.Random(args.seed).sample(images, min(args.n_pages, len(images))))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reference_stats(engine, config, images), indent=1))
    else:
        run_search(args, engine, config)


if __name__ == '__main__':
    main()
