"""Compare post-correction runs on the same recognizer output: corpus WER/CER per page set, relative reduction
against a raw run, and a paired page bootstrap of the difference.

  python tools/postcorr_report.py --runs outputs/runs/ladder_L0_raw outputs/runs/ladder_L1_rules ... \
      --raw outputs/runs/ladder_L0_raw --out outputs/postcorr/ladder.json
"""
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SETS = {'tuning': {5, 6, 8, 9, 11, 13, 14, 15, 16}, 'heldout': {7, 10, 12}}
SETS['all12'] = SETS['tuning'] | SETS['heldout']


@dataclass(frozen=True)
class Counts:
    we: np.ndarray  # word errors per page
    rw: np.ndarray  # reference words per page
    ce: np.ndarray
    rc: np.ndarray

    def wer(self, idx: np.ndarray | None = None) -> float:
        idx = slice(None) if idx is None else idx
        return 100 * self.we[idx].sum() / self.rw[idx].sum()

    def cer(self, idx: np.ndarray | None = None) -> float:
        idx = slice(None) if idx is None else idx
        return 100 * self.ce[idx].sum() / self.rc[idx].sum()


def load_counts(run: Path, pages: set[int], kind: str) -> Counts:
    rows = sorted((r for r in json.loads((run / 'result.json').read_text())['pages'] if r['page'] in pages),
                  key=lambda r: r['page'])
    if {r['page'] for r in rows} != pages:
        raise ValueError(f'{run}: missing pages {pages - {r["page"] for r in rows}}')
    return Counts(*(np.array([r[f'{kind}_{k}'] for r in rows], float) for k in ('we', 'rw', 'ce', 'rc')))


def paired_bootstrap(a: Counts, b: Counts, n: int, seed: int) -> tuple[float, float]:
    """95% interval of WER(b) - WER(a), resampling pages with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a.we), size=(n, len(a.we)))
    diffs = [b.wer(i) - a.wer(i) for i in idx]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def summarize(runs: list[Path], raw: Path, kind: str, n_boot: int, seed: int) -> dict:
    report: dict = {}
    for name, pages in SETS.items():
        base = load_counts(raw, pages, kind)
        report[name] = {}
        for run in runs:
            counts = load_counts(run, pages, kind)
            low, high = paired_bootstrap(base, counts, n_boot, seed)
            report[name][run.name] = {
                'wer': round(counts.wer(), 2), 'cer': round(counts.cer(), 2),
                'wer_reduction_pct': round(100 * (1 - counts.wer() / base.wer()), 1),
                'cer_reduction_pct': round(100 * (1 - counts.cer() / base.cer()), 1),
                'delta_wer_vs_raw_ci95': [round(low, 2), round(high, 2)],
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--raw', type=Path, required=True, help='run used as the uncorrected reference')
    parser.add_argument('--kind', default='norm', choices=['norm', 'strict'])
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    report = summarize(args.runs, args.raw, args.kind, args.bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()}, **report}, indent=2))
    for name, rows in report.items():
        logger.info('== %s (%s)', name, args.kind)
        for run, r in rows.items():
            logger.info('%-24s WER %6.2f  CER %5.2f  redWER %5.1f%%  redCER %5.1f%%  dWER CI95 %s', run, r['wer'],
                        r['cer'], r['wer_reduction_pct'], r['cer_reduction_pct'], r['delta_wer_vs_raw_ci95'])


if __name__ == '__main__':
    main()
