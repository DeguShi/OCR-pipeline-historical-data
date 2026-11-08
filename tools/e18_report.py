"""E18 report: per synthetic shift, held-out WER of the default v6 and of the chain each objective chose on the one
sample page, the share of the shift's damage it recovers, and how well each label-free objective ranks the candidates
(Spearman against greedy CER on the sample page; GT used for this analysis only).

  python tools/e18_report.py --root outputs/e18_colab/e18 --out records/e18/report
"""
import argparse
import json
from pathlib import Path

import numpy as np

OBJECTIVES = ('conf', 'lex', 'conflex', 'stat', 'oracle')
SIGN = {'conf': -1, 'lex': -1, 'conflex': -1, 'stat': 1}  # sign that makes "better" = lower, like CER


def ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind='stable')
    r = np.empty(len(values))
    r[order] = np.arange(len(values))
    for v in set(values):  # average ranks of ties
        idx = [i for i, x in enumerate(values) if x == v]
        r[idx] = r[idx].mean()
    return r


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float('nan')
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else float('nan')


def heldout_wer(shift_dir: Path) -> dict[str, float]:
    same = {}
    if (shift_dir / 'same_as.txt').is_file():
        for line in (shift_dir / 'same_as.txt').read_text().splitlines():
            left, right = line.split(' = ')
            same[left.split('/')[1]] = right
    out = {}
    for name in ('default', *OBJECTIVES):
        src = same.get(name, name)
        path = shift_dir / 'runs' / src / 'result.json'
        if path.is_file():
            out[name] = 100 * json.loads(path.read_text())['corpus']['wer_norm']
    return out


def proxy_quality(shift_dir: Path) -> dict[str, float]:
    rows = [json.loads(l) for l in (shift_dir / 'adapt/candidates.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['n_lines'] > 0 and 'cer_norm' in r]
    cer = [r['cer_norm'] for r in rows]
    return {o: spearman([SIGN[o] * r[o] for r in rows], cer) for o in SIGN} | {'n_candidates': len(rows)}


def chosen(shift_dir: Path) -> dict[str, str]:
    out = {}
    for o in OBJECTIVES:
        p = shift_dir / 'adapt' / f'chosen_{o}.json'
        if p.is_file():
            c = json.loads(p.read_text())
            out[o] = f"up{c['params']['upscale_factor']:g} seg[{'+'.join(c['seg_ops'])}] rec[{'+'.join(c['rec_ops'])}]"
    return out


def shift_rows(root: Path, clean: float) -> list[dict]:
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / 'adapt/candidates.jsonl').is_file()):
        wer = heldout_wer(d)
        loss = wer.get('default', float('nan')) - clean
        recovery = {o: (wer['default'] - wer[o]) / loss if o in wer and loss > 0 else float('nan') for o in OBJECTIVES}
        rows.append({'shift': d.name, 'wer': wer, 'loss': loss, 'recovery': recovery, 'spearman': proxy_quality(d),
                     'chosen': chosen(d)})
    return rows


def markdown(rows: list[dict], clean: float) -> str:
    head = '| shift | default | ' + ' | '.join(OBJECTIVES) + ' |\n|---|---|' + '---|' * len(OBJECTIVES) + '\n'
    body = ''.join(f"| {r['shift']} | {r['wer'].get('default', float('nan')):.2f} | " + ' | '.join(
        f"{r['wer'][o]:.2f} ({100 * r['recovery'][o]:+.0f}%)" if o in r['wer'] else '-' for o in OBJECTIVES) + ' |\n'
        for r in rows)
    corr = '| shift | n | ' + ' | '.join(SIGN) + ' |\n|---|---|' + '---|' * len(SIGN) + '\n' + ''.join(
        f"| {r['shift']} | {r['spearman']['n_candidates']} | " + ' | '.join(f"{r['spearman'][o]:.2f}" for o in SIGN)
        + ' |\n' for r in rows)
    picks = ''.join(f"- {r['shift']}: " + '; '.join(f'{o} {c}' for o, c in r['chosen'].items()) + '\n' for r in rows)
    return (f'Held-out WER (%) on 7, 10, 12; clean default {clean:.2f}. In brackets: share of the shift\'s damage '
            f'recovered (default - objective) / (default - clean).\n\n{head}{body}\n'
            f'Spearman(objective, greedy CER) over candidates on the sample page (1 = perfect ranking):\n\n{corr}\n'
            f'Chosen chains:\n\n{picks}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    clean = 100 * json.loads((args.root / 'clean/runs/default/result.json').read_text())['corpus']['wer_norm']
    rows = shift_rows(args.root, clean)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'e18_report.json').write_text(json.dumps({'clean': clean, 'shifts': rows}, indent=1))
    (args.out / 'e18_report.md').write_text(markdown(rows, clean))
    print(markdown(rows, clean))


if __name__ == '__main__':
    main()
