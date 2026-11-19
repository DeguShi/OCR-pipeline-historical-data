"""E19 summary from the pulled outputs of tools/run_e19.sh (both VMs): runs, search choices, weak-GT segmentation,
crop checks and the book-1 auto vs hand crop runs. Writes one JSON and prints a markdown digest.

  python tools/e19_report.py --a outputs/e19_colab_e19a/e19 --b outputs/e19_colab_e19b/e19 --out records/e19/e19_report.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_pipeline.metrics import corpus, score  # noqa: E402  (path set above)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def runs_summary(runs: Path) -> dict[str, Any]:
    out = {}
    for result in sorted(runs.glob('*/result.json')):
        r = load(result)
        out[result.parent.name] = {'wer_norm': r['corpus']['wer_norm'], 'cer_norm': r['corpus']['cer_norm'],
                                   'wer_strict': r['corpus']['wer_strict'], 'pages': len(r['pages']),
                                   'per_page': {p['page']: round(100 * p['wer_norm'], 2) for p in r['pages']}}
    return out


def chain_name(cfg: dict[str, Any]) -> str:
    p = cfg['params']
    extra = ''.join(f"[{k}={p[k]}]" for k in ('gamma', 'channel') if any(k in o for o in cfg['rec_ops']))
    return f"up{p['upscale_factor']:g}|seg {'+'.join(cfg['seg_ops']) or 'none'}|rec {'+'.join(cfg['rec_ops']) or 'none'}{extra}"


def search_summary(adapt: Path) -> dict[str, Any]:
    rows = [json.loads(l) for l in (adapt / 'candidates.jsonl').read_text().splitlines()]
    chosen = {f.stem.removeprefix('chosen_'): chain_name(load(f)) for f in sorted(adapt.glob('chosen_*.json'))}
    table = [{k: r.get(k) for k in ('stage', 'seg_name', 'name', 'n_lines', 'conf', 'lex', 'conflex', 'stat',
                                    'wer_norm', 'cer_norm')} for r in rows]
    return {'chosen': chosen, 'candidates': table}


def seg_summary(path: Path) -> dict[str, Any]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        a = agg[r['candidate']]
        for k in ('n_ref', 'n_seg', 'found', 'split', 'merged', 'missed', 'extra', 'norm_we', 'norm_rw', 'norm_ce',
                  'norm_rc'):
            a[k] += r[k]
        a['pages'] += 1
    return {c: {**{k: int(v) for k, v in a.items()}, 'recall': a['found'] / a['n_ref'],
                'greedy_wer': a['norm_we'] / a['norm_rw'], 'greedy_cer': a['norm_ce'] / a['norm_rc']}
            for c, a in agg.items()}


def rescore(run: Path, references: Path, images: Path) -> dict[str, float]:
    """Corpus scores of a run's page texts against another reference folder (book 1: Gemini references)."""
    rows = []
    for image in images.glob('*.jpg'):
        n = int(image.name.split('.')[0])
        text = run / f'page_{n:02d}.txt'
        ref = references / f'{image.stem}.txt'
        if text.is_file() and ref.is_file():
            rows.append(score(ref.read_text(encoding='utf-8'), text.read_text(encoding='utf-8')))
    return {**corpus(rows), 'pages': len(rows)} if rows else {}


def crop_summary(crop: Path) -> dict[str, Any]:
    out = {}
    if (f := crop / 'crop_book2.json').is_file():
        r = load(f)
        out['book2'] = {'photos': len(r), 'lines': sum(x['n_lines'] for x in r), 'inside': sum(x['inside'] for x in r),
                        'min_share': min((x['min_share'] for x in r if x['min_share'] is not None), default=None),
                        'transcript_lines': sum(x['transcript_lines'] or 0 for x in r),
                        'outside_photos': [x['photo'][:12] for x in r if x['inside'] < x['n_lines']]}
    if (f := crop / 'crop_book1.json').is_file():
        r = load(f)
        out['book1'] = {'pages': len(r), 'iou': [x['iou'] for x in r],
                        'auto_inside': sum(x['auto_lines']['inside'] for x in r),
                        'hand_inside': sum(x['hand_lines']['inside'] for x in r),
                        'lines': sum(x['auto_lines']['n_lines'] for x in r)}
    if (f := crop / 'crop_pagexml.json').is_file():
        r = load(f)
        out['pagexml'] = {'pages': len(r), 'lines': sum(x['n_lines'] for x in r), 'inside': sum(x['inside'] for x in r),
                          'watermarks': sum(x['watermark'] for x in r),
                          'max_wm_score': max((x['watermark_score'] for x in r), default=None)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', type=Path, required=True, help='VM A output (crop, seg)')
    parser.add_argument('--b', type=Path, required=True, help='VM B output (adapt, runs)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rep: dict[str, Any] = {}
    if (args.b / 'runs').is_dir():
        rep['runs'] = runs_summary(args.b / 'runs')
    if (args.b / 'adapt' / 'candidates.jsonl').is_file():
        rep['search'] = search_summary(args.b / 'adapt')
    if (args.a / 'seg_weak' / 'seg_weak.jsonl').is_file():
        rep['seg_weak'] = seg_summary(args.a / 'seg_weak' / 'seg_weak.jsonl')
    rep['crop'] = crop_summary(args.a / 'crop')
    if (args.a / 'book1_runs').is_dir():
        b1 = runs_summary(args.a / 'book1_runs')
        for name in b1:
            b1[name]['vs_gemini'] = rescore(args.a / 'book1_runs' / name, Path('data/ground_truth_gemini'),
                                            Path('data/images'))
        rep['book1_runs'] = b1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding='utf-8')
    for name, r in rep.get('runs', {}).items():
        print(f"run {name:8s} WER {100 * r['wer_norm']:.2f} CER {100 * r['cer_norm']:.2f} ({r['pages']} pages)")
    for obj, c in rep.get('search', {}).get('chosen', {}).items():
        print(f'chosen {obj:8s} {c}')
    for c, s in sorted(rep.get('seg_weak', {}).items(), key=lambda kv: kv[1]['greedy_cer']):
        print(f"seg {c:40s} recall {100 * s['recall']:.1f} found {s['found']}/{s['n_ref']} extra {s['extra']} "
              f"merged {s['merged']} split {s['split']} missed {s['missed']} greedy WER {100 * s['greedy_wer']:.2f} "
              f"CER {100 * s['greedy_cer']:.2f}")
    print(json.dumps(rep['crop'], ensure_ascii=False))
    for name, r in rep.get('book1_runs', {}).items():
        g = r['vs_gemini']
        print(f"book1 {name}: official WER {100 * r['wer_norm']:.2f} CER {100 * r['cer_norm']:.2f}; Gemini WER "
              f"{100 * g.get('wer_norm', float('nan')):.2f} CER {100 * g.get('cer_norm', float('nan')):.2f}")


if __name__ == '__main__':
    main()
