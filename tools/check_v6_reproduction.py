"""Regression check: v6 decoded from the committed recognizer posteriors must reproduce the committed final lines
exactly (sha256 per page), so a code or dependency change cannot silently move the held-out/test numbers.

The posteriors (records/postcorr/posteriors/{v6_gt12,v6_test10}, written by tools/export_posteriors.py from the
st_c1m recognizer) make the check independent of kraken, the images and a GPU. Expected hashes live in
records/postcorr/repro/v6_expected.json together with the WER they score to.

  python tools/check_v6_reproduction.py            # check all sets (about 10 CPU min)
  python tools/check_v6_reproduction.py --record   # rewrite the expected hashes (only after a deliberate change)
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_pipeline.config import load_config  # noqa: E402  (path set above)
from postcorr_from_posteriors import PosteriorCodec, decode_page, load_page  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
SETS = {'heldout': ('v6_gt12', [7, 10, 12]), 'test10': ('v6_test10', [18, 22, 23, 24, 25, 31, 32, 34, 44, 55])}


def page_hash(lines: list[str]) -> str:
    return hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest()


def decode_set(config_path: Path, posteriors: Path, pages: list[int]) -> dict[str, str]:
    config = load_config(config_path)
    codec = PosteriorCodec(json.loads((posteriors / 'alphabet.json').read_text()))
    out = {}
    for page in pages:
        out[str(page)] = page_hash(decode_page(codec, load_page(posteriors / f'page_{page:02d}.npz'), config))
        logger.info('page %d decoded', page)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/final_v6.json')
    parser.add_argument('--posteriors', type=Path, default=ROOT / 'records/postcorr/posteriors')
    parser.add_argument('--expected', type=Path, default=ROOT / 'records/postcorr/repro/v6_expected.json')
    parser.add_argument('--sets', nargs='+', default=sorted(SETS), choices=sorted(SETS))
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    expected = json.loads(args.expected.read_text()) if args.expected.is_file() else {}
    bad = 0
    for name in args.sets:
        folder, pages = SETS[name]
        got = decode_set(args.config, args.posteriors / folder, pages)
        if args.record:
            expected.setdefault(name, {})['pages'] = got
            continue
        diff = [p for p, h in got.items() if expected[name]['pages'].get(p) != h]
        bad += len(diff)
        logger.info('%s: %s', name, f'MISMATCH on pages {diff}' if diff else f'all {len(got)} pages reproduce')
    if args.record:
        args.expected.parent.mkdir(parents=True, exist_ok=True)
        args.expected.write_text(json.dumps(expected, indent=1), encoding='utf-8')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
