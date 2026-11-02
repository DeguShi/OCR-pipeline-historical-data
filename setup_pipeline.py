"""Setup (free-tier Colab or local); run from the repository root.

  python setup_pipeline.py                          # installs Kraken + text deps (never torch), fetches and checks
                                                    # the files configs/final_v6.json needs
  python setup_pipeline.py --config configs/final_v5.json
  python setup_pipeline.py --skip-install           # only fetches and checks the files

The lexicon and context LM of the final configuration ship with the repository. The fine-tuned recognizers are
hosted on the Hugging Face Hub and the public base models on Zenodo; every download is hash-checked.
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
KRAKEN = 'kraken @ git+https://github.com/mittagessen/kraken.git@bb487219e7d8289274443d1ad0989efb4924dc2a'
PACKAGES = [KRAKEN, 'transformers==4.57.1', 'rapidfuzz==3.14.1']
CONFIG = Path('configs/final_v6.json')
HUB = 'https://huggingface.co/KobayashiShiroe/OCR-pipeline-historical-data/resolve/main'
RESOURCE_KEYS = ('lexicon', 'period_lexicon', 'context_lm', 'edit_costs')
DOWNLOADS = {  # path (as engine.MODELS / configs expect) -> (url, algorithm, digest)
    Path('models/finetuned/st_c1m.safetensors'):  # final recognizer (v6)
        (f'{HUB}/st_c1m.safetensors', 'sha256', '8b1488d66916357ab9ad1770e5e53777573295f5d2bbc987fb25c717a75ad15a'),
    Path('models/finetuned/st_g3m.safetensors'):  # v5 recognizer
        (f'{HUB}/st_g3m.safetensors', 'sha256', 'd41ac347a293d5d7a3588a7a0b771aa7a9332f3a540b13f7ded18f11fd1eee4d'),
    Path('models/10.5281_zenodo.21788410/medium.safetensors'):
        ('https://zenodo.org/records/21788410/files/medium.safetensors?download=1', 'md5',
         'e3411a453ce3b9e9efae3f8631d85762'),
    Path('models/lexicon/pt_br_50k.txt'):
        ('https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/pt_br/pt_br_50k.txt',
         'sha256', 'a61d6f2ede97c5daad5fb3907b72a228f0aff12668be6d24da590d20804fa611'),
}


def digest(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, path.read_bytes(), usedforsecurity=False).hexdigest()


def fetch(path: Path, url: str, algorithm: str, expected: str) -> None:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info('downloading %s', path)
        urllib.request.urlretrieve(url, path)
    actual = digest(path, algorithm)
    if actual != expected:
        raise ValueError(f'{path}: {algorithm} {actual} != {expected}; delete it and rerun')
    logger.info('ok %s', path)


def needed(config: Path) -> list[Path]:
    """Recognizer weights and lexicon/LM files the configuration names."""
    from ocr_pipeline.engine import MODELS

    data = json.loads(config.read_text(encoding='utf-8'))
    return [MODELS[data['rec']['model']], *(Path(data['post'][k]) for k in RESOURCE_KEYS if data['post'].get(k))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--skip-install', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if not Path('ocr_pipeline/run.py').is_file():
        raise SystemExit('run from the repository root')
    if not args.skip_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGES], check=True)
    for path in needed(args.config):
        if path in DOWNLOADS:
            fetch(path, *DOWNLOADS[path])
        elif path.is_file():
            logger.info('ok %s', path)
        else:
            raise SystemExit(f'{path} is missing: rebuild it with tools/build_period_lexicon.py (see records/README.md)')


if __name__ == '__main__':
    main()
