"""Baseline D: masked-LM rescoring of our own candidates with PortOldBERT (Osorio & Lopes Cardoso, EACL 2026), a
historical-Portuguese DeBERTa encoder. The masked LM adds one term to v6's choice; everything else is v6.

v6 builds, per line, one slot per word (candidate strings with CTC log-likelihood + lexicon prior) and picks one
candidate per slot with a trigram Viterbi (ocr_pipeline.context_lm.decode_line). Here:
1. dump: run v6 from the posterior export (single process), recording every decode_line call (slots and v6's
   choice), in call order per page;
2. score: per slot with alternatives, the pseudo-log-likelihood (PLL; Salazar et al. 2020) of each of its top-k
   candidates (by v6's slot score, plus v6's choice) in the line, the other words fixed to v6's choice, with the
   previous and next lines' choices as extra context. PLL = sum over the candidate's subword tokens of log p(token)
   with that token masked. Candidates outside the top-k get the slot's lowest PLL;
3. decode: rerun v6 with every slot score s replaced by s + lambda * PLL (same Viterbi, same trigram, same text rules
   and lexicon edit step), writing lines.json for tools/score_lines.py --no-rules. lambda is chosen on the tuning
   pages only.

  python tools/postcorr_mlm.py dump --config configs/final_v6.json --posteriors outputs/posteriors/v6_gt12 \
      --out outputs/baselines/mlm/slots_gt12.json
  python tools/postcorr_mlm.py score --slots outputs/baselines/mlm/slots_gt12.json --out outputs/baselines/mlm/pll_gt12.json
  python tools/postcorr_mlm.py decode --config configs/final_v6.json --posteriors outputs/posteriors/v6_gt12 \
      --pll outputs/baselines/mlm/pll_gt12.json --lam 0.5 --out outputs/baselines/mlm/gt12_lam0.5
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_pipeline import ctc_rescore  # noqa: E402  (path set above)
from ocr_pipeline.config import load_config  # noqa: E402
from postcorr_from_posteriors import PosteriorCodec, decode_page, load_page  # noqa: E402

logger = logging.getLogger(__name__)
MODEL = 'LIACC/PortOldBERT'
MODEL_REVISION = '9641bba86b86173f91a0e3936f28fa76b1c94121'
TOKENIZER = 'PORTULAN/albertina-100m-portuguese-ptpt-encoder'
REAL_DECODE = ctc_rescore.decode_line


def page_paths(posteriors: Path) -> list[tuple[str, Path]]:
    return [(str(int(p.stem.split('_')[1])), p) for p in sorted(posteriors.glob('page_*.npz'))]


def run_pages(config_path: Path, posteriors: Path, hook) -> dict[str, list[str]]:
    """v6 on every exported page, single process, with decode_line replaced by hook(page, slots, lm, beta, order)."""
    config = load_config(config_path)
    codec = PosteriorCodec(json.loads((posteriors / 'alphabet.json').read_text()))
    lines = {}
    for page, path in page_paths(posteriors):
        ctc_rescore.decode_line = lambda slots, lm, beta, order=2, _p=page: hook(_p, slots, lm, beta, order)
        lines[page] = decode_page(codec, load_page(path), config)
    ctc_rescore.decode_line = REAL_DECODE
    return lines


def dump(args: argparse.Namespace) -> None:
    calls: dict[str, list[dict]] = {}

    def hook(page, slots, lm, beta, order):
        chosen = REAL_DECODE(slots, lm, beta, order)
        calls.setdefault(page, []).append({'slots': slots, 'chosen': chosen})
        return chosen

    lines = run_pages(args.config, args.posteriors, hook)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'calls': calls, 'lines': lines}, ensure_ascii=False), encoding='utf-8')
    logger.info('dumped %d calls on %d pages', sum(map(len, calls.values())), len(calls))


def masked_texts(words: list[str], i: int, cand: str, left: str, right: str, tok) -> tuple[list[list[int]], list[int]]:
    """Token ids of the context with words[i] = cand, once per candidate token with that token masked."""
    pre = ' '.join(([left] if left else []) + words[:i])
    post = ' '.join(words[i + 1:] + ([right] if right else []))
    ids_pre = tok(pre, add_special_tokens=False)['input_ids'] if pre else []
    ids_cand = tok((' ' if pre else '') + cand, add_special_tokens=False)['input_ids']
    ids_post = tok(' ' + post, add_special_tokens=False)['input_ids'] if post else []
    ids_pre, ids_post = ids_pre[-200:], ids_post[:200]  # well inside the 512-token window
    base = [tok.cls_token_id] + ids_pre + ids_cand + ids_post + [tok.sep_token_id]
    start = 1 + len(ids_pre)
    seqs = [base[:start + j] + [tok.mask_token_id] + base[start + j + 1:] for j in range(len(ids_cand))]
    return seqs, ids_cand


def pll_batch(model, tok, items: list[tuple[list[list[int]], list[int], int]], device: str) -> list[float]:
    """items: (masked sequences, candidate token ids, candidate start); one PLL per item."""
    import torch
    flat = [(n, j, s) for n, (seqs, _, _) in enumerate(items) for j, s in enumerate(seqs)]
    width = max(len(s) for _, _, s in flat)
    ids = torch.full((len(flat), width), tok.pad_token_id)
    for r, (_, _, s) in enumerate(flat):
        ids[r, :len(s)] = torch.tensor(s)
    with torch.inference_mode():
        logits = model(input_ids=ids.to(device), attention_mask=(ids != tok.pad_token_id).to(device)).logits
    logp = torch.log_softmax(logits.float(), -1).cpu()
    out = [0.0] * len(items)
    for r, (n, j, _) in enumerate(flat):
        _, cand_ids, start = items[n]
        out[n] += float(logp[r, start + j, cand_ids[j]])
    return out


def slot_jobs(calls: dict[str, list[dict]], k: int) -> list[tuple[str, int, int, list[str]]]:
    """(page, call, slot, candidates to score) for every slot with alternatives."""
    jobs = []
    for page, page_calls in calls.items():
        for c, call in enumerate(page_calls):
            for s, slot in enumerate(call['slots']):
                if len(slot) < 2:
                    continue
                top = [w for w, _ in sorted(slot, key=lambda x: -x[1])[:k]]
                jobs.append((page, c, s, list(dict.fromkeys(top + [call['chosen'][s]]))))
    return jobs


def context(call_list: list[dict], c: int, side: int) -> str:
    j = c + side
    return ' '.join(call_list[j]['chosen']) if 0 <= j < len(call_list) else ''


def score(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, revision=MODEL_REVISION).to(device).eval()
    calls = json.loads(args.slots.read_text(encoding='utf-8'))['calls']
    jobs = slot_jobs(calls, args.k)
    t0, items, keys, pll = time.time(), [], [], {}
    for page, c, s, cands in jobs:
        call = calls[page][c]
        left, right = context(calls[page], c, -1), context(calls[page], c, 1)
        for cand in cands:
            seqs, ids = masked_texts(call['chosen'], s, cand, left, right, tok)
            start = seqs[0].index(tok.mask_token_id)
            items.append((seqs, ids, start)); keys.append((page, c, s, cand))
    logger.info('%d slots, %d candidates, %d forward rows on %s', len(jobs), len(items),
                sum(len(i[0]) for i in items), device)
    for b in range(0, len(items), args.batch):
        for key, value in zip(keys[b:b + args.batch], pll_batch(model, tok, items[b:b + args.batch], device)):
            pll.setdefault(key[0], {}).setdefault(str(key[1]), {}).setdefault(str(key[2]), {})[key[3]] = value
    meta = {'model': MODEL, 'revision': MODEL_REVISION, 'tokenizer': TOKENIZER, 'k': args.k, 'device': device,
            'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None, 'seconds': round(time.time() - t0, 1),
            'slots': len(jobs), 'candidates': len(items), 'torch': torch.__version__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'meta': meta, 'pll': pll}, ensure_ascii=False), encoding='utf-8')
    logger.info('ALL DONE: %s', meta)


def decode(args: argparse.Namespace) -> None:
    pll = json.loads(args.pll.read_text(encoding='utf-8'))['pll']
    counter: dict[str, int] = {}

    def hook(page, slots, lm, beta, order):
        c = counter.get(page, 0); counter[page] = c + 1
        table = pll.get(page, {}).get(str(c), {})
        new = []
        for s, slot in enumerate(slots):
            scores = table.get(str(s))
            if not scores:
                new.append(slot); continue
            floor = min(scores.values())
            new.append([(w, v + args.lam * scores.get(w, floor)) for w, v in slot])
        return REAL_DECODE(new, lm, beta, order)

    lines = run_pages(args.config, args.posteriors, hook)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'lines.json').write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding='utf-8')
    (args.out / 'params.json').write_text(json.dumps({'config': str(args.config), 'pll': str(args.pll),
                                                      'lam': args.lam}, indent=1), encoding='utf-8')
    logger.info('ALL DONE: lam %s, %d pages', args.lam, len(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump'); d.add_argument('--config', type=Path, required=True)
    d.add_argument('--posteriors', type=Path, required=True); d.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('score'); s.add_argument('--slots', type=Path, required=True)
    s.add_argument('--k', type=int, default=10); s.add_argument('--batch', type=int, default=16)
    s.add_argument('--seed', type=int, default=42); s.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('decode'); r.add_argument('--config', type=Path, required=True)
    r.add_argument('--posteriors', type=Path, required=True); r.add_argument('--pll', type=Path, required=True)
    r.add_argument('--lam', type=float, required=True); r.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    {'dump': dump, 'score': score, 'decode': decode}[args.cmd](args)


if __name__ == '__main__':
    main()
