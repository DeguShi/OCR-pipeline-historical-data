"""E20 step 7: tables and paired comparisons of the full systems on held-out (7, 10, 12) and test10.

Reads the scored decodes of tools/run_e20.sh eval_decode / eval_pe (<eval>/score/SET/REC/DECODER/REF/result.json),
adds the two VLM systems run before E20 with the same character map and scoring (V0, transcription from the line
crop alone, frozen at tr_pt; R1·PE, the frozen paper_pt_period correction of R1's greedy text), and writes:
- <out>/grid.json and <out>/grid.md: WER/CER of every system on every set and reference;
- <out>/paired.json: block bootstrap (tools/block_bootstrap.py: block 10, 10k resamples, seed 42) of R1·P6 against
  each comparison system, and of the decomposition pairs.

  python tools/e20_report.py --eval outputs/e20_colab_e20ev_ev/outputs/e20/eval --out records/e20/eval
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from block_bootstrap import block_errors, block_lengths, compare, page_texts  # noqa: E402  (path set above)
from ocr_pipeline.data import discover  # noqa: E402

logger = logging.getLogger(__name__)
SETS = {'heldout': {'pages': [7, 10, 12], 'images': 'data/images',
                    'refs': {'oficial': 'data/ground_truth', 'gemini': 'data/ground_truth_gemini'}},
        'test10': {'pages': [18, 22, 23, 24, 25, 31, 32, 34, 44, 55], 'images': 'data/extra/teste/imagens',
                   'refs': {'gemini': 'data/extra/teste/referencias'}}}
EXTRA = {('v0', 'v0'): 'records/e20/v0/{s}_tr_pt/lines.json',
         ('r1_st_c1m', 'pe'): 'outputs/baselines/search/vlm/final_{s}_paper_pt_period/lines.json'}
OURS = ('r1_st_c1m', 'p6')
VERSUS = [('r4_traprinq', 'pa'), ('r4_traprinq', 'p6'), ('r1_st_c1m', 'pa'), ('r4_traprinq', 'pe'),
          ('r1_st_c1m', 'pe'), ('v0', 'v0'), ('r5_mccatmus', 'pa'), ('r0_medium', 'p6')]
DECOMPOSE = [(('r4_traprinq', 'p6'), ('r4_traprinq', 'pa')), (('r1_st_c1m', 'pa'), ('r4_traprinq', 'pa')),
             (('r1_st_c1m', 'p0'), ('r4_traprinq', 'p0'))]


def tool(*args: str) -> None:
    subprocess.run([sys.executable, '-I', *args], check=True, stdout=subprocess.DEVNULL)


def score_extra(ev: Path) -> None:
    """Character map + text rules + score for the VLM systems decoded before E20 (same as eval_pe)."""
    for (rec, dec), pattern in EXTRA.items():
        for s, spec in SETS.items():
            mapped = ev / 'lines' / s / rec / f'{dec}.json'
            tool('tools/charmap.py', '--input', pattern.format(s=s), '--out', str(mapped))
            for ref, path in spec['refs'].items():
                out = ev / 'score' / s / rec / dec / ref
                tool('tools/score_lines.py', '--lines', str(mapped), '--config', 'configs/final_v6.json',
                     '--images', spec['images'], '--references', path, '--out', str(out))


def grid(ev: Path) -> dict:
    rows = {}
    for p in sorted(ev.glob('score/*/*/*/*/result.json')):
        s, rec, dec, ref = p.parts[-5:-1]
        c = json.loads(p.read_text())['corpus']
        rows.setdefault(f'{rec}·{dec}', {})[f'{s}/{ref}'] = {'wer': round(100 * c['wer_norm'], 2),
                                                              'cer': round(100 * c['cer_norm'], 2)}
    return rows


def markdown(rows: dict) -> str:
    cols = [f'{s}/{r}' for s, spec in SETS.items() for r in spec['refs']]
    head = '| system | ' + ' | '.join(f'{c} WER | {c} CER' for c in cols) + ' |\n|---|' + '---|' * 2 * len(cols)
    body = [f'| {name} | ' + ' | '.join(f"{v.get(c, {}).get('wer', '-')} | {v.get(c, {}).get('cer', '-')}"
                                        for c in cols) + ' |' for name, v in rows.items()]
    return '\n'.join([head, *body]) + '\n'


def paired(ev: Path, a: tuple, b: tuple, s: str, ref: str) -> dict:
    spec = SETS[s]
    pages = {p.number: p for p in discover(Path(spec['images']), Path(spec['refs'][ref]))}
    run = lambda sys_: ev / 'score' / s / sys_[0] / sys_[1] / ref  # noqa: E731
    ta, tb = page_texts(run(a)), page_texts(run(b))
    ea, eb, ln, wins = [], [], [], [0, 0, 0]
    for n in spec['pages']:
        text = pages[n].reference_text()
        x, y = block_errors(text, ta[n], 10), block_errors(text, tb[n], 10)
        ea.append(x), eb.append(y), ln.append(block_lengths(text, 10))
        wins[0 if x.sum() < y.sum() else 1 if x.sum() == y.sum() else 2] += 1
    result = compare(np.concatenate(ea), np.concatenate(eb), np.concatenate(ln), 10000, 42)
    return {**{k: round(v, 2) if isinstance(v, float) else v for k, v in result.items()},
            'a': '·'.join(a), 'b': '·'.join(b), 'set': s, 'ref': ref, 'pages_won_tied_lost': wins}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, required=True, help='OUT/eval of tools/run_e20.sh')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    score_extra(args.eval)
    rows = grid(args.eval)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'grid.json').write_text(json.dumps(rows, indent=1))
    (args.out / 'grid.md').write_text(markdown(rows))
    pairs = [(OURS, b) for b in VERSUS] + DECOMPOSE
    results = [paired(args.eval, a, b, s, ref) for a, b in pairs for s, spec in SETS.items() for ref in spec['refs']
               if f'{a[0]}·{a[1]}' in rows and f'{b[0]}·{b[1]}' in rows]
    (args.out / 'paired.json').write_text(json.dumps(results, indent=1))
    for r in results:
        logger.info('%-22s vs %-18s %-7s %-7s dWER %+6.2f CI95 %s p %.4f pages %s', r['a'], r['b'], r['set'], r['ref'],
                    r['dwer'], r['ci95'], r['p_signflip'], r['pages_won_tied_lost'])


if __name__ == '__main__':
    main()
