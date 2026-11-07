"""Baseline C: a ByT5-small post-OCR corrector trained on synthetic noise (never on evaluation pages).

1. noise: character confusion statistics of OUR recognizer, from its raw greedy output vs the reference on the tuning
   pages only (one optimal Levenshtein path per page): per reference character, P(kept), P(substituted by c),
   P(deleted); and P(insertion of c) per position. Spaces are ordinary characters, so word splits and merges are in.
2. data: period-corpus lines (the corpus our lexicon and n-grams come from), corrupted with that model -> (noisy,
   clean) pairs; a held-back slice of the pairs is the validation set (synthetic too).
3. train: google/byt5-small, seq2seq, fp32 (T5 overflows in fp16), fixed seed.
4. predict: every line of the posterior export (its greedy text, the input every other decoder reads) -> corrected
   line; lines.json for tools/score_lines.py (text rules applied, like every external baseline).

  python tools/baseline_byt5.py noise --posteriors outputs/posteriors/v6_gt12 --pages 5,6,8,9,11,13,14,15,16 \
      --out outputs/baselines/byt5/noise.json
  python tools/baseline_byt5.py data --noise outputs/baselines/byt5/noise.json --corpus outputs/lm/period_w4_livro78.txt \
      --n 200000 --out outputs/baselines/byt5/pairs.jsonl
  python tools/baseline_byt5.py train --pairs outputs/baselines/byt5/pairs.jsonl --out outputs/baselines/byt5/model
  python tools/baseline_byt5.py predict --model outputs/baselines/byt5/model --posteriors outputs/posteriors/v6_test10 \
      --out outputs/baselines/byt5/test10
"""
import argparse
import json
import logging
import random
import sys
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)
MODEL = 'google/byt5-small'
DATA_DIR, IMAGES = Path('data/ground_truth'), Path('data/images')


PLAIN = re.compile(r"[A-Za-zÀ-ÿ0-9 ,.;:!?()\-'\"º°ª§/]")


def nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def plain(line: str) -> bool:
    """Ordinary text: at least 90% letters, digits, spaces and common punctuation (drops binary junk)."""
    return sum(bool(PLAIN.match(c)) for c in line) >= 0.9 * len(line)


def greedy_pages(posteriors: Path) -> dict[int, list[str]]:
    pages = {}
    for path in sorted(posteriors.glob('page_*.npz')):
        data = np.load(path)
        pages[int(path.stem.split('_')[1])] = [str(data[f'greedy_{i}']) for i in range(int(data['n_lines']))]
    return pages


def confusions(ref: str, hyp: str, sub: dict, ins: Counter, slots: list[int]) -> None:
    """Accumulate per-reference-character outcomes and insertions along one optimal path."""
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp).as_list():
        if tag == 'equal':
            for ch in ref[i1:i2]:
                sub[ch][ch] += 1
        elif tag == 'delete':
            for ch in ref[i1:i2]:
                sub[ch][''] += 1
        elif tag == 'insert':
            ins.update(hyp[j1:j2])
        else:
            for k, ch in enumerate(ref[i1:i2]):
                sub[ch][hyp[j1 + k] if j1 + k < j2 else ''] += 1
            ins.update(hyp[j1 + (i2 - i1):j2])
    slots[0] += len(ref) + 1


def noise(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ocr_pipeline.data import discover
    keep = {int(p) for p in args.pages.split(',')}
    refs = {p.number: p.reference.read_text(encoding='utf-8') for p in discover(IMAGES, DATA_DIR) if p.number in keep}
    hyps = greedy_pages(args.posteriors)
    sub: dict = defaultdict(Counter); ins: Counter = Counter(); slots = [0]
    for page in sorted(keep):
        confusions(nfc(' '.join(refs[page].split())), nfc(' '.join(' '.join(hyps[page]).split())), sub, ins, slots)
    model = {'pages': sorted(keep), 'sub': {c: dict(v) for c, v in sub.items()}, 'ins': dict(ins),
             'ins_rate': sum(ins.values()) / slots[0]}
    errors = sum(n for c, v in sub.items() for o, n in v.items() if o != c) + sum(ins.values())
    model['cer'] = errors / sum(sum(v.values()) for v in sub.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding='utf-8')
    logger.info('noise model from %d pages: CER %.3f, insertion rate %.4f', len(keep), model['cer'], model['ins_rate'])


def corrupt(line: str, model: dict, rng: random.Random) -> str:
    """Sample one recognizer-like reading of a clean line from the confusion model."""
    ins_chars, ins_w = list(model['ins']), list(model['ins'].values())
    out = []
    for ch in line + '\0':
        if ins_chars and rng.random() < model['ins_rate']:
            out.append(rng.choices(ins_chars, ins_w)[0])
        if ch == '\0':
            break
        table = model['sub'].get(ch)
        out.append(rng.choices(list(table), list(table.values()))[0] if table else ch)
    return nfc(''.join(out))


def chunks(text: str, width: int) -> list[str]:
    """Cut running text into manuscript-like lines of about `width` characters at word boundaries."""
    lines, cur = [], []
    for word in text.split():
        if cur and len(' '.join(cur + [word])) > width:
            lines.append(' '.join(cur)); cur = []
        cur.append(word)
    return lines + ([' '.join(cur)] if cur else [])


def clean_lines(book: Path, laws: Path, width: int) -> tuple[list[str], list[str]]:
    book_lines = [nfc(' '.join(l.split())) for f in sorted(book.glob('*.txt'))
                  for l in f.read_text(encoding='utf-8').splitlines() if len(l.split()) >= 2]
    law_lines = [l for f in sorted(laws.glob('*.txt')) for l in chunks(nfc(f.read_text(encoding='utf-8')), width)
                 if len(l) >= 15 and plain(l)]
    return book_lines, law_lines


def data(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    model = json.loads(args.noise.read_text(encoding='utf-8'))
    book, laws = clean_lines(args.book, args.laws, args.width)
    pool = book * args.book_repeat + rng.sample(laws, min(len(laws), max(0, args.n - len(book) * args.book_repeat)))
    rng.shuffle(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as f:
        for clean in pool:
            f.write(json.dumps({'src': corrupt(clean, model, rng), 'tgt': clean}, ensure_ascii=False) + '\n')
    logger.info('%d pairs (%d book lines x%d, %d law lines)', len(pool), len(book), args.book_repeat,
                len(pool) - len(book) * args.book_repeat)


def record_env(out: Path) -> dict:
    """pip freeze and nvidia-smi into out/, library versions returned for params."""
    import subprocess
    import torch
    import transformers
    out.mkdir(parents=True, exist_ok=True)
    (out / 'env.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                                                text=True).stdout, encoding='utf-8')
    smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True) if torch.cuda.is_available() else None
    (out / 'nvidia_smi.txt').write_text(smi.stdout if smi else 'no gpu', encoding='utf-8')
    return {'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version.split()[0]}


def train(args: argparse.Namespace) -> None:
    import torch
    from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              T5ForConditionalGeneration, set_seed)
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MODEL)
    rows = [json.loads(l) for l in args.pairs.read_text(encoding='utf-8').splitlines()][:args.limit or None]
    n_val = min(2000, len(rows) // 20)

    def encode(r: dict) -> dict:
        enc = tok(r['src'], max_length=args.max_len, truncation=True)
        enc['labels'] = tok(text_target=r['tgt'], max_length=args.max_len, truncation=True)['input_ids']
        return enc

    train_set, val_set = [encode(r) for r in rows[n_val:]], [encode(r) for r in rows[:n_val]]
    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.out / 'ckpt'), per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_ratio=0.05, weight_decay=0.01, seed=args.seed,
        eval_strategy='steps', eval_steps=args.eval_steps, save_strategy='steps', save_steps=args.eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', logging_steps=100,
        fp16=False, report_to=[], dataloader_num_workers=2)
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_set, eval_dataset=val_set,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model))
    t0 = time.time(); trainer.train()
    trainer.save_model(str(args.out)); tok.save_pretrained(str(args.out))
    meta = {'model': MODEL, 'pairs': str(args.pairs), 'n_train': len(train_set), 'n_val': n_val, 'epochs': args.epochs,
            'batch': args.batch, 'accum': args.accum, 'limit': args.limit, 'lr': args.lr, 'max_len': args.max_len, 'seed': args.seed,
            'train_seconds': round(time.time() - t0, 1), 'best_eval_loss': trainer.state.best_metric,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, **record_env(args.out)}
    (args.out / 'train_params.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    logger.info('ALL DONE: %s', meta)


def predict(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(str(args.model))
    model = T5ForConditionalGeneration.from_pretrained(str(args.model)).to(device).eval()
    pages, out, t0 = greedy_pages(args.posteriors), {}, time.time()
    for page, lines in pages.items():
        out[str(page)] = [''] * len(lines)
        todo = [i for i, l in enumerate(lines) if l.strip()]
        for b in range(0, len(todo), args.batch):
            idx = todo[b:b + args.batch]
            enc = tok([nfc(lines[i]) for i in idx], return_tensors='pt', padding=True, truncation=True,
                      max_length=args.max_len).to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_length=args.max_len, num_beams=args.beams, do_sample=False)
            for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[str(page)][i] = text
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'lines.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    meta = {'model': str(args.model), 'posteriors': str(args.posteriors), 'beams': args.beams, 'device': device,
            'seconds': round(time.time() - t0, 1), 'lines': sum(map(len, out.values())), **record_env(args.out)}
    (args.out / 'params.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    logger.info('ALL DONE: %s', meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    n = sub.add_parser('noise'); n.add_argument('--posteriors', type=Path, required=True)
    n.add_argument('--pages', required=True); n.add_argument('--out', type=Path, required=True)
    d = sub.add_parser('data'); d.add_argument('--noise', type=Path, required=True)
    d.add_argument('--book', type=Path, default=Path('data/extra/transcricoes/treino'))
    d.add_argument('--laws', type=Path, required=True); d.add_argument('--n', type=int, default=200000)
    d.add_argument('--book-repeat', type=int, default=10); d.add_argument('--width', type=int, default=45)
    d.add_argument('--seed', type=int, default=42); d.add_argument('--out', type=Path, required=True)
    t = sub.add_parser('train'); t.add_argument('--pairs', type=Path, required=True)
    t.add_argument('--epochs', type=float, default=1.0); t.add_argument('--batch', type=int, default=32)
    t.add_argument('--lr', type=float, default=5e-4); t.add_argument('--max-len', type=int, default=192)
    t.add_argument('--accum', type=int, default=1); t.add_argument('--limit', type=int, default=0)
    t.add_argument('--eval-steps', type=int, default=1000); t.add_argument('--seed', type=int, default=42)
    t.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('predict'); p.add_argument('--model', type=Path, required=True)
    p.add_argument('--posteriors', type=Path, required=True); p.add_argument('--beams', type=int, default=4)
    p.add_argument('--batch', type=int, default=32); p.add_argument('--max-len', type=int, default=192)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    random.seed(42); np.random.seed(42)
    {'noise': noise, 'data': data, 'train': train, 'predict': predict}[args.cmd](args)


if __name__ == '__main__':
    main()
