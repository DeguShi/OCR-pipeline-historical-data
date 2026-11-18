"""E20 robustness check: paired comparisons with the PAGE as the unit (the word-block bootstrap of e20_report.py
ignores the correlation of errors within a page).

For each pair (a, b) and each set/reference: per page d_i = word errors of a minus word errors of b (normalized
policy, from score/SET/REC/DEC/REF/result.json), dWER = sum(d_i) / sum(reference words). Two statistics:
- exact paired sign-flip permutation over pages (2^n sign patterns; two-sided p, |stat| >= |observed|);
- page bootstrap: resample pages with replacement (10k, seed 42), 95% percentile interval of dWER.
With 3 held-out pages the sign-flip p cannot go below 0.25; the held-out rows are descriptive.

  python tools/e20_page_test.py --eval records/e20/eval --out records/e20/eval/paired_pages.json
"""
import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
SETS = {'heldout': ['oficial', 'gemini'], 'test10': ['gemini']}
OURS = ('r1_st_c1m', 'p6')
VERSUS = [('r4_traprinq', 'pa'), ('r4_traprinq', 'p6'), ('r1_st_c1m', 'pa'), ('r5_mccatmus', 'pa'),
          ('r4_traprinq', 'pe'), ('r1_st_c1m', 'pe'), ('r0_medium', 'p6'), ('v0', 'v0')]
DECOMPOSE = [(('r4_traprinq', 'p6'), ('r4_traprinq', 'pa')), (('r1_st_c1m', 'pa'), ('r4_traprinq', 'pa')),
             (('r1_st_c1m', 'p0'), ('r4_traprinq', 'p0'))]


def page_counts(ev: Path, sys_: tuple, s: str, ref: str) -> dict[int, tuple[int, int]]:
    rows = json.loads((ev / 'score' / s / sys_[0] / sys_[1] / ref / 'result.json').read_text())['pages']
    return {r['page']: (r['norm_we'], r['norm_rw']) for r in rows}


def page_test(d: np.ndarray, rw: np.ndarray, n_boot: int, seed: int) -> dict:
    obs = d.sum()
    flips = np.array(list(itertools.product((1, -1), repeat=len(d))))
    p = float(np.mean(np.abs(flips @ d) >= abs(obs) - 1e-9))
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boot = 100 * d[idx].sum(1) / rw[idx].sum(1)
    return {'dwer': round(100 * obs / rw.sum(), 2), 'p_signflip_pages': round(p, 4),
            'ci95_page_bootstrap': [round(float(x), 2) for x in np.percentile(boot, [2.5, 97.5])]}


def compare(ev: Path, a: tuple, b: tuple, s: str, ref: str) -> dict:
    ca, cb = page_counts(ev, a, s, ref), page_counts(ev, b, s, ref)
    pages = sorted(ca)
    if [ca[p][1] for p in pages] != [cb[p][1] for p in pages]:
        raise ValueError(f'reference lengths differ: {a} {b} {s} {ref}')
    d = np.array([ca[p][0] - cb[p][0] for p in pages], dtype=float)
    rw = np.array([ca[p][1] for p in pages], dtype=float)
    return {'a': '·'.join(a), 'b': '·'.join(b), 'set': s, 'ref': ref, 'pages': len(pages),
            'won_tied_lost': [int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())],
            **page_test(d, rw, 10000, 42)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, required=True, help='dir with score/SET/REC/DEC/REF/result.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pairs = [(OURS, b) for b in VERSUS] + DECOMPOSE
    results = [compare(args.eval, a, b, s, ref) for a, b in pairs for s, refs in SETS.items() for ref in refs]
    args.out.write_text(json.dumps(results, indent=1))
    for r in results:
        logger.info('%-14s vs %-14s %-7s %-7s dWER %+6.2f page-CI95 %s p %.4f W/T/L %s', r['a'], r['b'], r['set'],
                    r['ref'], r['dwer'], r['ci95_page_bootstrap'], r['p_signflip_pages'], r['won_tied_lost'])


if __name__ == '__main__':
    main()
