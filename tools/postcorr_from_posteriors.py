"""Run our post-correction (the configuration's decoder: CTC rescoring of lexicon candidates + n-gram context) on
an exported posterior file instead of the recognizer, and time it. External decoders read the same export, so this
gives a like-for-like cost comparison (same input, same machine, one process) and needs neither kraken nor images.

The output lines.json holds the final lines: decoder output, then the configuration's text rules and lexicon edit
step (postprocess.correct, as in the pipeline), so score it with tools/score_lines.py --no-rules; on the same pages it
must reproduce the pipeline run exactly. (External baselines get the text rules only, from score_lines.)
params.json records the load time (lexicon + n-gram model, paid once per process), per-page decode wall and CPU
seconds, peak RSS and the model file sizes.

  python tools/postcorr_from_posteriors.py --config configs/final_v6.json --posteriors outputs/posteriors/v6_gt12 \
      --pages all --out outputs/baselines/v6_from_posteriors_gt12
  python tools/score_lines.py --lines outputs/baselines/v6_from_posteriors_gt12/lines.json --config configs/final_v6.json \
      --no-rules --out outputs/runs/v6_from_posteriors_gt12
"""
import argparse
import json
import logging
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_pipeline import pipeline  # noqa: E402  (path set above)
from ocr_pipeline.config import load_config  # noqa: E402
from ocr_pipeline.context_lm import load_lm  # noqa: E402
from ocr_pipeline.ctc_rescore import LineLogits  # noqa: E402
from ocr_pipeline.postprocess import correct, lexicon_choices  # noqa: E402

logger = logging.getLogger(__name__)


class PosteriorCodec:
    """Stands in for the engine: rescore_page only asks it for the codec."""

    def __init__(self, alphabet: dict) -> None:
        self.l2c = {int(k): v for k, v in alphabet['l2c'].items()}
        self.c2l = {c: [k] for k, c in self.l2c.items()}
        self.n_classes = int(alphabet['n_classes'])

    def codec(self, model: str) -> tuple[dict[str, list[int]], dict[int, str], int]:
        return self.c2l, self.l2c, self.n_classes


def load_page(path: Path) -> list[LineLogits]:
    data = np.load(path)
    return [LineLogits(str(data[f'greedy_{i}']), data[f'idx_{i}'], data[f'val_{i}'])
            for i in range(int(data['n_lines']))]


def decode_page(codec: PosteriorCodec, logits: list[LineLogits], config) -> list[str]:
    pipeline.page_logits = lambda *args, **kwargs: logits  # the recognizer's output comes from the export
    return correct(pipeline.rescore_page(codec, None, None, config, '', ())[1], config.post, ())


def model_files(config) -> dict[str, int]:
    paths = [config.post.period_lexicon, config.post.context_lm, config.post.edit_costs, config.post.lexicon]
    return {p: Path(p).stat().st_size for p in paths if p and Path(p).is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--posteriors', type=Path, required=True)
    parser.add_argument('--pages', default='all', help="'all' or a comma list of page numbers")
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_config(args.config)
    if config.post.domain_lexicon:
        raise ValueError('domain lexicon needs the tuning references; not supported from posteriors')
    codec = PosteriorCodec(json.loads((args.posteriors / 'alphabet.json').read_text()))
    paths = sorted(args.posteriors.glob('page_*.npz'))
    if args.pages != 'all':
        keep = {int(p) for p in args.pages.split(',')}
        paths = [p for p in paths if int(p.stem.split('_')[1]) in keep]
    w0, c0 = time.perf_counter(), time.process_time()
    lexicon_choices(config.post, ())
    if config.post.context_lm:
        load_lm(config.post.context_lm)
    load = {'wall': round(time.perf_counter() - w0, 2), 'cpu': round(time.process_time() - c0, 2)}
    lines, pages = {}, {}
    for path in paths:
        page = str(int(path.stem.split('_')[1]))
        logits = load_page(path)
        w0, c0 = time.perf_counter(), time.process_time()
        lines[page] = decode_page(codec, logits, config)
        pages[page] = {'wall': round(time.perf_counter() - w0, 2), 'cpu': round(time.process_time() - c0, 2),
                       'lines': len(logits)}
        logger.info('page %s: %d lines, %.1f s', page, len(logits), pages[page]['wall'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'lines.json').write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding='utf-8')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    params = {'config': str(args.config), 'posteriors': str(args.posteriors), 'load_seconds': load,
              'decode_wall': round(sum(p['wall'] for p in pages.values()), 2),
              'decode_cpu': round(sum(p['cpu'] for p in pages.values()), 2), 'pages': pages,
              'peak_rss_mb': round(rss / (2**20 if sys.platform == 'darwin' else 2**10), 1),
              'model_bytes': model_files(config), 'python': platform.python_version(),
              'machine': platform.machine(), 'processor': platform.processor()}
    (args.out / 'params.json').write_text(json.dumps(params, indent=1), encoding='utf-8')
    logger.info('ALL DONE: load %.1f s, decode %.1f s wall / %.1f s cpu, peak %.0f MB', load['wall'],
                params['decode_wall'], params['decode_cpu'], params['peak_rss_mb'])


if __name__ == '__main__':
    main()
