"""X02 report: the E20 comparisons that involve PA, redone with the 181-configuration PA (decoder name pa181).

Reads the E20 scores (records/e20/eval/score/SET/REC/DEC/REF/result.json) and the X02 scores (same layout, decoder
pa181), and writes, for each pair, the word-block bootstrap of tools/e20_report.py and the page-level tests of
tools/e20_page_test.py (same seeds, blocks and resamples):
- R1·P6 against R1/R4/R5·pa181 (does our system's lead survive the larger PA search?);
- R4·P6 against R4·pa181 and R5·P6 against R5·pa181 (decoder share with the recognizer fixed);
- R·pa181 against R·pa (effect of the search size alone).

  python tools/x02_report.py --e20 records/e20/eval --x02 OUT/eval --out OUT/report
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e20_page_test  # noqa: E402  (path set above)
import e20_report  # noqa: E402

logger = logging.getLogger(__name__)
RECS = ['r1_st_c1m', 'r4_traprinq', 'r5_mccatmus']
PAIRS = ([(('r1_st_c1m', 'p6'), (r, 'pa181')) for r in RECS]
         + [(('r4_traprinq', 'p6'), ('r4_traprinq', 'pa181')), (('r5_mccatmus', 'p6'), ('r5_mccatmus', 'pa181'))]
         + [((r, 'pa181'), (r, 'pa')) for r in RECS])


def merged_scores(e20: Path, x02: Path, out: Path) -> Path:
    """One score tree with the E20 systems and the X02 pa181 runs side by side (copies, inputs untouched)."""
    ev = out / 'merged'
    if ev.exists():
        shutil.rmtree(ev)
    shutil.copytree(e20 / 'score', ev / 'score')
    for p in sorted(x02.glob('score/*/*/pa181')):
        shutil.copytree(p, ev / 'score' / p.parts[-3] / p.parts[-2] / 'pa181')
    return ev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--e20', type=Path, required=True, help='E20 eval dir (records/e20/eval)')
    parser.add_argument('--x02', type=Path, required=True, help='X02 eval dir (OUT/eval of tools/run_x02.sh)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args.out.mkdir(parents=True, exist_ok=True)
    ev = merged_scores(args.e20, args.x02, args.out)
    rows = e20_report.grid(ev)
    pa181 = {k: v for k, v in rows.items() if k.endswith('·pa181')}
    (args.out / 'grid_pa181.json').write_text(json.dumps(pa181, indent=1))
    blocks, pages = [], []
    for a, b in PAIRS:
        for s, refs in e20_page_test.SETS.items():
            for ref in refs:
                blocks.append(e20_report.paired(ev, a, b, s, ref))
                pages.append(e20_page_test.compare(ev, a, b, s, ref))
    (args.out / 'paired.json').write_text(json.dumps(blocks, indent=1))
    (args.out / 'paired_pages.json').write_text(json.dumps(pages, indent=1))
    for name, v in pa181.items():
        logger.info('GRID %s %s', name, json.dumps(v))
    for r, p in zip(blocks, pages):
        logger.info('PAIR %-18s vs %-20s %-7s %-7s dWER %+6.2f CI95 %s | pages %s p_page %s CI95_page %s',
                    r['a'], r['b'], r['set'], r['ref'], r['dwer'], r['ci95'], r['pages_won_tied_lost'],
                    p.get('p_signflip_pages'), p.get('ci95_page_bootstrap'))


if __name__ == '__main__':
    main()
