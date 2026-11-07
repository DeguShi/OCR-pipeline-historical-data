"""External baseline E: image-informed post-OCR correction with a vision-language model, following Thomas et al.,
"Image-Informed Post-OCR Correction with Vision-Language Models" (Findings of EMNLP 2026), out-of-the-box prompting
setting: one-shot prompt of their Figure 7, sampling disabled. It is a labeled external baseline, not part of our
pipeline.

Input per line: the line crop and the recognizer's greedy text (tools/export_line_crops.py manifest), the same
recognizer output every other decoder reads. Lines whose OCR text is empty are left empty (correction, not re-OCR).
Deviations from the paper, stated: line instead of sentence units; 4-bit NF4 weights on a T4 instead of an A100.

Prompts (--prompt):
- paper_en: Figure 7 verbatim, with the paper's English example;
- paper_pt: the same instruction with one Portuguese example (tuning page 6, line 3, corrected by hand from the
  image). Which one is used on held-out and test is chosen on the tuning pages.

Every line is appended to <out>/generations.jsonl with fsync; a restarted run skips the lines already there. At the
end <out>/lines.json ({page: [lines]}, for tools/score_lines.py) and <out>/params.json are written.

  python tools/baseline_vlm.py --manifest outputs/line_crops/gt12/manifest.json --pages 5,6,8,9,11,13,14,15,16 \
      --prompt paper_en --out outputs/baselines/vlm/tuning_paper_en
"""
import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
INSTRUCTION = ('I will provide you with OCR output and source image. Please fix the errors in the provided text. '
               'Return only the corrected text. Do not include any additional commentary or explanations. '
               'If you understand, please reply with OK.')
EXAMPLES = {'paper_en': ('__RO BBERY AT A BARONET’S. V', 'ROBBERY AT A BARONETS.'),
            'paper_pt': ('Reginento interro da Camara Memicipal', 'Regimento interno da Camara Municipal')}


def messages(prompt: str, image: Image.Image, ocr: str) -> list[dict]:
    example_ocr, example_fixed = EXAMPLES[prompt]
    text = lambda t: [{'type': 'text', 'text': t}]  # noqa: E731
    return [{'role': 'user', 'content': text(INSTRUCTION)}, {'role': 'assistant', 'content': text('OK')},
            {'role': 'user', 'content': text(example_ocr)}, {'role': 'assistant', 'content': text(example_fixed)},
            {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': ocr}]}]


def load(quant: str):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    kwargs = {'torch_dtype': torch.float16, 'device_map': 'cuda', 'revision': REVISION}
    if quant == '4bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                           bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, **kwargs).eval()
    return model, AutoProcessor.from_pretrained(MODEL, revision=REVISION)


@torch.inference_mode()
def correct(model, processor, prompt: str, image: Image.Image, ocr: str, max_new: int) -> tuple[str, int]:
    from qwen_vl_utils import process_vision_info
    msgs = messages(prompt, image, ocr)
    chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(msgs)
    inputs = processor(text=[chat], images=images, return_tensors='pt').to('cuda')
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new = out[0, inputs['input_ids'].shape[1]:]
    return processor.decode(new, skip_special_tokens=True).strip(), int(new.shape[0])


def done_keys(path: Path) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return {(r['page'], r['line']): r for r in rows}


def run(args: argparse.Namespace) -> dict[tuple[int, int], dict]:
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))['pages']
    pages = [int(p) for p in args.pages.split(',')] if args.pages != 'all' else sorted(int(p) for p in manifest)
    log = args.out / 'generations.jsonl'
    done = done_keys(log)
    todo = [(p, r) for p in pages for r in manifest[str(p)] if (p, r['line']) not in done]
    logger.info('%d pages, %d lines to run, %d already done', len(pages), len(todo), len(done))
    model = processor = None
    for n, (page, row) in enumerate(todo, 1):
        start, text, tokens = time.time(), '', 0
        if row['ocr'].strip() and row['file']:
            if model is None:
                model, processor = load(args.quant)
            with Image.open(args.manifest.parent / row['file']) as img:
                text, tokens = correct(model, processor, args.prompt, img.convert('RGB'), row['ocr'], args.max_new)
        rec = {'page': page, 'line': row['line'], 'ocr': row['ocr'], 'out': text, 'new_tokens': tokens,
               'seconds': round(time.time() - start, 3)}
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        done[(page, row['line'])] = rec
        if n % 25 == 0:
            logger.info('%d/%d lines', n, len(todo))
    return {k: v for k, v in done.items() if k[0] in pages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pages', default='all')
    parser.add_argument('--prompt', choices=sorted(EXAMPLES), required=True)
    parser.add_argument('--quant', choices=['4bit', 'fp16'], default='4bit')
    parser.add_argument('--max-new', type=int, default=160)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = run(args)
    lines: dict[str, list[str]] = {}
    for (page, line), rec in sorted(rows.items()):
        lines.setdefault(str(page), []).append(rec['out'])
    (args.out / 'lines.json').write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding='utf-8')
    import transformers
    import bitsandbytes
    params = {'model': MODEL, 'revision': REVISION, 'prompt': args.prompt, 'example': EXAMPLES[args.prompt],
              'quant': args.quant, 'max_new': args.max_new, 'seed': args.seed, 'do_sample': False,
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'bitsandbytes': bitsandbytes.__version__, 'gpu': torch.cuda.get_device_name(0),
              'python': platform.python_version(), 'gpu_seconds': round(sum(r['seconds'] for r in rows.values()), 1),
              'lines': len(rows)}
    (args.out / 'params.json').write_text(json.dumps(params, indent=1, ensure_ascii=False), encoding='utf-8')
    logger.info('ALL DONE: %d lines, %.0f s', len(rows), params['gpu_seconds'])


if __name__ == '__main__':
    main()
