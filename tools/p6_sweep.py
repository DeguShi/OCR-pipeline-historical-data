"""E20 rule 4 for P6: the E16 v6 sweep (records/colab_jobs/sweep_clean.py), run from a recognizer's tuning posteriors.

Each post-correction variant is decoded with tools/postcorr_from_posteriors.py on the 9 tuning pages, mapped with
tools/charmap.py and scored with tools/score_lines.py --no-rules against both tuning references. Same rule as E16:
a variant is accepted if it lowers the WER against the Gemini reference by >= 0.3 points without raising the
official WER; accepted variants are combined greedily (largest gain first, each kept if it gains >= 0.15 more).
Left out of the E16 list: pad16/pad32/no_deskew change the image (rule 3 fixes preprocessing and segmentation), and
lm_w4 needs models/lexicon/pt_hist_legal_w4.ngrams.npz, which is not in the repository or the model repo.

Resumable: every scored run is one JSON line in <out>/runs.jsonl (its files in <out>/runs/NAME). Output: <out>/final_overrides.json and
<out>/config.json (the frozen configuration).

  python tools/p6_sweep.py --config configs/e20/r2_traprinq.json --posteriors outputs/e20/tune/post/r2_traprinq \
      --out outputs/e20/select/r2_traprinq/p6 --workers 2
"""
import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)
TUNING = '5,6,8,9,11,13,14,15,16'
REFS = {'oficial': 'data/ground_truth', 'gemini': 'data/ground_truth_gemini'}
ACCEPT, KEEP = 0.3, 0.15  # WER points, as E16
VARIANTS = {
    'alpha8': {'ctc_alpha': 8}, 'alpha24': {'ctc_alpha': 24},
    'beta05': {'context_beta': 0.5}, 'beta15': {'context_beta': 1.5},
    'ratio04': {'lexicon_max_ratio': 0.4}, 'ratio06': {'lexicon_max_ratio': 0.6},
    'short40': {'context_short': 40}, 'short80': {'context_short': 80},
    'cand20': {'ctc_candidates': 20}, 'cand50': {'ctc_candidates': 50},
    'nbest3': {'context_nbest': 3}, 'nbest8': {'context_nbest': 8},
    'beam0': {'beam': 0}, 'beam128': {'beam': 128},
    'minlen3': {'lexicon_min_len': 3}, 'fold': {'fold_accents': True},
}


def with_post(config: dict, overrides: dict) -> dict:
    return {**config, 'post': {**config['post'], **overrides}}


def tool(name: str, *args: str) -> None:
    subprocess.run([sys.executable, '-I', f'tools/{name}', *args], check=True, stdout=subprocess.DEVNULL)


def score_run(name: str, overrides: dict, config: dict, posteriors: str, work: Path) -> dict:
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'config.json').write_text(json.dumps(with_post(config, overrides), indent=1))
    tool('postcorr_from_posteriors.py', '--config', str(d / 'config.json'), '--posteriors', posteriors,
         '--pages', TUNING, '--out', str(d / 'p6'))
    tool('charmap.py', '--input', str(d / 'p6' / 'lines.json'), '--out', str(d / 'lines_mapped.json'))
    row = {'name': name, 'overrides': overrides}
    for ref, path in REFS.items():
        tool('score_lines.py', '--lines', str(d / 'lines_mapped.json'), '--config', str(d / 'config.json'),
             '--no-rules', '--references', path, '--out', str(d / f'score_{ref}'))
        c = json.loads((d / f'score_{ref}' / 'result.json').read_text())['corpus']
        row[ref] = {'wer': round(100 * c['wer_norm'], 2), 'cer': round(100 * c['cer_norm'], 2)}
    return row


class Sweep:
    def __init__(self, config: dict, posteriors: str, out: Path, workers: int) -> None:
        self.config, self.posteriors, self.out, self.workers = config, posteriors, out, workers
        self.log = out / 'runs.jsonl'
        self.done = {r['name']: r for r in map(json.loads, self.log.read_text().splitlines())} \
            if self.log.exists() else {}

    def run(self, jobs: dict[str, dict]) -> dict[str, dict]:
        todo = {n: o for n, o in jobs.items() if n not in self.done}
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {n: pool.submit(score_run, n, o, self.config, self.posteriors, self.out / "runs") for n, o in todo.items()}
            for name, future in futures.items():
                row = future.result()
                with self.log.open('a') as f:
                    f.write(json.dumps(row) + '\n')
                self.done[name] = row
                logger.info('%s gemini %.2f oficial %.2f', name, row['gemini']['wer'], row['oficial']['wer'])
        return {n: self.done[n] for n in jobs}


def combine(sweep: Sweep, results: dict) -> dict:
    base = results['base']
    gains = {n: round(base['gemini']['wer'] - r['gemini']['wer'], 2) for n, r in results.items() if n != 'base'
             and base['gemini']['wer'] - r['gemini']['wer'] >= ACCEPT and r['oficial']['wer'] <= base['oficial']['wer']}
    current, chosen, best = {}, [], base
    for name in sorted(gains, key=gains.get, reverse=True):
        trial = {**current, **VARIANTS[name]}
        result = sweep.run({f'combo_{"+".join([*chosen, name])}': trial}).popitem()[1] if chosen else results[name]
        if best['gemini']['wer'] - result['gemini']['wer'] >= KEEP and result['oficial']['wer'] <= best['oficial']['wer']:
            current, best = trial, result
            chosen.append(name)
    return {'accepted': gains, 'chosen': chosen, 'overrides': current, 'result': best, 'base': base}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--posteriors', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--variants', default=None, help='comma list, a subset of VARIANTS (smoke test only)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    sweep = Sweep(config, args.posteriors, args.out, args.workers)
    variants = {n: VARIANTS[n] for n in args.variants.split(',')} if args.variants else VARIANTS
    final = combine(sweep, sweep.run({'base': {}, **variants}))
    final['config_source'] = str(args.config)
    (args.out / 'final_overrides.json').write_text(json.dumps(final, indent=1))
    (args.out / 'config.json').write_text(json.dumps(with_post(config, final['overrides']), indent=1))
    logger.info('P6 chosen %s: gemini %.2f oficial %.2f', final['chosen'], final['result']['gemini']['wer'],
                final['result']['oficial']['wer'])


if __name__ == '__main__':
    main()
