"""Resumable hyperparameter search for the external baselines (A: pyctcdecode+KenLM, B: Word Beam Search) on the
tuning pages only. Stdlib only; each configuration runs the baseline script in a subprocess.

Configurations are a fixed function of --kind and --seed, so every machine sees the same list:
- id 0 is the configuration chosen by the first, smaller search (the reproducibility check: its output must match
  the one decoded before);
- A: random search over LM, alpha, beta, OOV offset, the two pruning thresholds and the beam width;
- B: the full grid of mode, smoothing, corpus and beam (the space is small).
--shard i/k runs every k-th configuration starting at i.

Every finished configuration is appended as one JSON line (params, page lines, seconds) to <out>/results.jsonl,
written with fsync. A configuration is skipped if its id is already there, so a restarted job resumes where it stopped.
Scoring needs the references and runs elsewhere (tools/score_lines.py).

  python tools/baseline_search.py --kind A --python /content/venv/bin/python --root /content/work --shard 0/1 \
      --out /content/work/search_A
"""
import argparse
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)
TUNING = '5,6,8,9,11,13,14,15,16'
LMS = {'kn3_w1': ('outputs/lm/kn3_w1.bin', 'outputs/lm/period_w1.txt'),
       'kn3_w4b8': ('outputs/lm/kn3_w4b8.bin', 'outputs/lm/period_w4_livro78.txt')}
CORPORA = {'w1': 'outputs/lm/period_w1.txt', 'w4b8': 'outputs/lm/period_w4_livro78.txt'}


def configs_a(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = [{'lm': 'kn3_w1', 'alpha': 0.8, 'beta': 1.0, 'unk': -10.0, 'beam_prune_logp': -10.0,
            'token_min_logp': -5.0, 'beam': 64}]
    for _ in range(n):
        out.append({'lm': rng.choice(sorted(LMS)), 'alpha': round(rng.uniform(0.2, 2.0), 3),
                    'beta': round(rng.uniform(-1.0, 4.0), 3), 'unk': round(rng.uniform(-20.0, -2.0), 2),
                    'beam_prune_logp': round(rng.uniform(-20.0, -5.0), 2),
                    'token_min_logp': round(rng.uniform(-10.0, -2.0), 2), 'beam': rng.choice([32, 64, 128])})
    return [dict(c, id=i) for i, c in enumerate(out)]


def configs_b() -> list[dict]:
    out = [{'mode': 'NGrams', 'smoothing': 0.003, 'corpus': 'w1', 'beam': 64}]
    for corpus in ('w1', 'w4b8'):
        for smoothing in (0.001, 0.003, 0.01):
            for beam in (64, 128):
                c = {'mode': 'NGrams', 'smoothing': smoothing, 'corpus': corpus, 'beam': beam}
                if c != out[0]:
                    out.append(c)
    out.append({'mode': 'NGramsForecast', 'smoothing': 0.003, 'corpus': 'w4b8', 'beam': 64})
    return [dict(c, id=i) for i, c in enumerate(out)]


def command(kind: str, c: dict, python: str, out: Path) -> list[str]:
    common = ['--posteriors', 'outputs/posteriors/v6_gt12', '--pages', TUNING, '--workers', '2', '--out', str(out)]
    if kind == 'A':
        lm, corpus = LMS[c['lm']]
        return [python, '-I', 'tools/baseline_pyctcdecode.py', *common, '--lm', lm, '--corpus', corpus,
                '--alpha', str(c['alpha']), '--beta', str(c['beta']), '--unk', str(c['unk']), '--beam', str(c['beam']),
                '--beam-prune-logp', str(c['beam_prune_logp']), '--token-min-logp', str(c['token_min_logp'])]
    return [python, '-I', 'tools/baseline_wordbeam.py', *common, '--corpus', CORPORA[c['corpus']],
            '--mode', c['mode'], '--smoothing', str(c['smoothing']), '--beam', str(c['beam'])]


def done_ids(results: Path) -> set[int]:
    if not results.exists():
        return set()
    return {json.loads(line)['id'] for line in results.read_text().splitlines() if line.strip()}


def run_one(kind: str, c: dict, python: str, root: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        start = time.time()
        subprocess.run(command(kind, c, python, Path(tmp)), cwd=root, check=True)
        (lines_file,) = Path(tmp).glob('*/lines.json')
        return {'id': c['id'], 'kind': kind, 'params': c, 'seconds': round(time.time() - start, 1),
                'lines': json.loads(lines_file.read_text())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', choices=['A', 'B'], required=True)
    parser.add_argument('--n', type=int, default=120, help='random configurations for A (besides id 0)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shard', default='0/1', help='i/k')
    parser.add_argument('--python', required=True, help='interpreter of the decoder environment')
    parser.add_argument('--root', type=Path, required=True, help='repository root (tools/ and outputs/)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--list', action='store_true', help='print the configurations and exit')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    configs = configs_a(args.n, args.seed) if args.kind == 'A' else configs_b()
    i, k = (int(x) for x in args.shard.split('/'))
    configs = configs[i::k]
    if args.list:
        print('\n'.join(json.dumps(c) for c in configs))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / 'results.jsonl'
    done = done_ids(results)
    todo = [c for c in configs if c['id'] not in done]
    logger.info('kind %s shard %s: %d configurations, %d done, %d to run', args.kind, args.shard, len(configs),
                len(configs) - len(todo), len(todo))
    for c in todo:
        row = run_one(args.kind, c, args.python, args.root)
        with results.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        logger.info('id %d done in %.0fs (%d/%d)', c['id'], row['seconds'], len(done_ids(results)), len(configs))
    logger.info('ALL DONE')


if __name__ == '__main__':
    main()
