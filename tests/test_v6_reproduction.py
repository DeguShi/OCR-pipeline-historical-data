"""Slow regression test (about 10 CPU min): v6 from the committed posteriors reproduces the committed final lines.
Run with OCR_SLOW=1 pytest tests/test_v6_reproduction.py"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get('OCR_SLOW') != '1', reason='slow; set OCR_SLOW=1')
def test_v6_reproduces_committed_lines() -> None:
    res = subprocess.run([sys.executable, str(ROOT / 'tools/check_v6_reproduction.py')], cwd=ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
