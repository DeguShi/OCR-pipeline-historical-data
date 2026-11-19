"""E19: turn a folder of archive photos + one-text-line-per-page-line transcripts into a pipeline-ready book.

Output (same layout as tools/traprinq_prepare.py, so ocr_pipeline.run and tools/run_e18_books.sh read it):
  images/<i>.<stem>.jpg      automatic crop (ocr_pipeline.auto_crop) of the photo; boxes in images/crops.json
  references/<i>.<stem>.txt  transcript with line-end hyphenation joined (scoring reference)
  lines/<i>.<stem>.txt       transcript exactly as given, one text line per page line (weak segmentation GT)
  manifest.json              {'cases': [{'case', 'pages', 'sample', 'eval'}]} plus the photo -> i map
i is the rank of the photo when sorted by its leading number, then name (two photos may share a number).
The sample/eval split comes from --split (a JSON with 'sample' and 'eval' lists of photo stems, drawn beforehand).

Hyphenation: a line ending in a letter + '-' or '=' followed by a line starting with a lowercase letter is joined
(the word moves up, as in the book-1 references, which carry no line-end hyphens); other dashes are kept.

  python tools/book_prepare.py --images PHOTOS --transcripts TXT --split E19_split.json --out outputs/e19/book
"""
import argparse
import json
import re
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_pipeline.auto_crop import auto_crop  # noqa: E402  (path set above)

def join_hyphens(text: str) -> tuple[str, int]:
    """'pala-\nvra de' -> 'palavra\nde' when the next line starts with a lowercase letter."""
    lines = [line.rstrip() for line in text.rstrip('\n').split('\n')]
    joined = 0
    for i in range(len(lines) - 1):
        following = lines[i + 1].lstrip()
        if re.search(r'[^\W\d_][-=]$', lines[i]) and re.match(r'[a-zà-ÿ]', following):
            head, _, rest = following.partition(' ')
            lines[i], lines[i + 1], joined = lines[i][:-1] + head, rest, joined + 1
    return '\n'.join(line for line in lines if line.strip()) + '\n', joined


def order(photos: list[Path]) -> list[Path]:
    return sorted(photos, key=lambda p: (int(re.match(r'\d+', p.name).group()), p.name))


def prepare(args: argparse.Namespace) -> dict:
    photos = order([p for p in args.images.glob('*.jpg')])
    split = json.loads(args.split.read_text(encoding='utf-8'))
    for sub in ('images', 'references', 'lines'):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    index, crops, joined = {}, {}, 0
    for i, photo in enumerate(photos, 1):
        text = (args.transcripts / f'{photo.stem}.txt').read_text(encoding='utf-8')
        name = f'{i}.{photo.stem}'
        with Image.open(photo) as image:
            cropped, result = auto_crop(image)
        cropped.save(args.out / 'images' / f'{name}.jpg', quality=95)
        crops[name] = result.to_dict()
        (args.out / 'lines' / f'{name}.txt').write_text(text, encoding='utf-8')
        ref, n = join_hyphens(text)
        joined += n
        (args.out / 'references' / f'{name}.txt').write_text(ref, encoding='utf-8')
        index[photo.stem] = i
    (args.out / 'images' / 'crops.json').write_text(json.dumps(crops, indent=1), encoding='utf-8')
    case = {'case': args.case, 'pages': len(photos), 'sample': [index[s] for s in split['sample']],
            'eval': [index[s] for s in split['eval']]}
    manifest = {'cases': [case], 'photo_index': index, 'hyphens_joined': joined, 'split_file': str(args.split)}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--transcripts', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', default='livro2')
    args = parser.parse_args()
    m = prepare(args)
    c = m['cases'][0]
    print(f"{c['pages']} pages, sample {c['sample']}, eval {c['eval']}, {m['hyphens_joined']} hyphens joined")


if __name__ == '__main__':
    main()
