"""E20 rule 1: rebuild the exact st_c1m training lines (2,425 lines of the 78 Gemini training pages, E16) from the
committed records, so that other recognizers fine-tune on the same data without relabelling.

records/training/st_c1m/labels_r1/lines.jsonl holds every labelled line (baseline, boundary, transcript,
transcript_cer) and label_config.json the preprocessing of the labelling run. This script regenerates the page
images that labelling wrote (the configuration's geometric chain, then rec_ops, on data/extra/treino_crop), copies
the rows, and runs the same selection as the E16 job (pseudo_labels select --min-chars 10 --max-transcript-cer 0.4
--transcripts-only). It fails unless the kept lines, pages and characters equal records/.../select_gem_r1.json.
Output: <out>/images/*.png, <out>/train/*.xml (PAGE XML for `ketos compile -f page`), <out>/select.json.

  python tools/rebuild_training_lines.py --out outputs/e20/train_lines
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_pipeline.config import PipelineConfig, load_config  # noqa: E402  (path set above)
from ocr_pipeline.preprocessing import apply_chain  # noqa: E402
from ocr_pipeline.tuning import pseudo_labels  # noqa: E402

logger = logging.getLogger(__name__)
RECORDS = Path('records/training/st_c1m')
SELECT = ['--min-chars', '10', '--max-transcript-cer', '0.4', '--transcripts-only']


def write_images(config: PipelineConfig, images: Path, pages: set[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for page in sorted(pages):
        with Image.open(images / f'{page}.jpg') as original:
            base = apply_chain(original.convert('RGB'), config.geometric_ops, config.params, geometric=True)
        apply_chain(base, config.rec_ops, config.params, geometric=False).save(out / f'{page}.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labels', type=Path, default=RECORDS / 'labels_r1')
    parser.add_argument('--expected', type=Path, default=RECORDS / 'select_gem_r1.json')
    parser.add_argument('--images', type=Path, default=Path('data/extra/treino_crop'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_config(args.labels / 'label_config.json')
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.labels / 'lines.jsonl', args.out / 'lines.jsonl')
    rows = [json.loads(line) for line in (args.out / 'lines.jsonl').read_text(encoding='utf-8').splitlines()]
    write_images(config, args.images, {r['page'] for r in rows}, args.out / 'images')
    pseudo_labels.select(pseudo_labels.build_parser().parse_args(['select', str(args.out), *SELECT]))
    got = json.loads((args.out / 'select.json').read_text())
    want = json.loads(args.expected.read_text())
    keys = ('lines', 'kept', 'pages', 'kept_chars')
    if any(got[k] != want[k] for k in keys):
        raise SystemExit(f'rebuild differs from {args.expected}: {[(k, got[k], want[k]) for k in keys]}')
    logger.info('REBUILD OK: %s', {k: got[k] for k in keys})


if __name__ == '__main__':
    main()
