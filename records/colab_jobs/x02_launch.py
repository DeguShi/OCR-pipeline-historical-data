"""X02 Colab launcher, run on the VM with `colab exec -s x02 -f records/colab_jobs/x02_launch.py` (status:
x02_tail.py). Before it: upload code.tgz (`git archive HEAD`, in 20 MB parts joined on the VM), post.tgz (the E20
posteriors, from x02_remote.tar.gz on Drive OCR-backup-20261118) and x02_sha.txt / x02_mode.txt."""
import subprocess
import tarfile
from pathlib import Path

# X02 on a free Colab CPU VM. Uploaded: /content/code.tgz (git archive of the commit in /content/x02_sha.txt),
# /content/post.tgz (E20 tuning + eval posteriors of R1/R4/R5), /content/x02_mode.txt (smoke | full).
# Idempotent: never relaunches a running job of the same mode.
MODE = Path('/content/x02_mode.txt').read_text().strip()
W = Path('/content/x02')
LOG = W / f'outputs/job_{MODE}.log'
if LOG.exists() and 'JOB_' not in LOG.read_text(errors='replace'):
    print(f'x02 {MODE} already running, not relaunched')
    raise SystemExit(0)
if not (W / 'tools/run_x02.sh').exists():
    W.mkdir(exist_ok=True)
    tarfile.open('/content/code.tgz').extractall(W, filter='data')
    (W / 'outputs/x02').mkdir(parents=True, exist_ok=True)
    tarfile.open('/content/post.tgz').extractall(W / 'outputs/x02', filter='data')
SHA = Path('/content/x02_sha.txt').read_text().strip()
JOB = r'''#!/bin/bash
set -euo pipefail
cd /content/x02
SECONDS=0
if [ ! -f /content/setup_done ]; then
  (command -v /usr/bin/time >/dev/null || (apt-get -qq update && apt-get -qq install -y time)) >/dev/null 2>&1
  env -u PYTHONPATH -u UV_SYSTEM_PYTHON bash -c '(command -v uv || pip install -q uv) >/dev/null && uv venv -q --python 3.12.12 --python-preference only-managed /content/venv && uv pip install -q --python /content/venv/bin/python -r requirements.txt && uv venv -q --python 3.12.12 --python-preference only-managed /content/dvenv && uv pip install -q --python /content/dvenv/bin/python -r tools/requirements-decoders.txt'
  for c in configs/final_v6.json configs/e20/r4_traprinq.json configs/e20/r5_mccatmus.json; do
    /content/venv/bin/python setup_pipeline.py --skip-install --config $c; done
  mkdir -p outputs/lm; H=https://huggingface.co/KobayashiShiroe/OCR-pipeline-historical-data/resolve/main/lm
  for f in kn3_w1.bin kn3_w4b8.bin period_w1.txt period_w4_livro78.txt; do curl -sfL -o outputs/lm/$f $H/$f; done
  cat > /content/lm.sha <<EOF
81cf1f5366ff8354e93b336207fc2bf8f213e5c96bd9d529ef29e93136b7777e  outputs/lm/period_w1.txt
9f87ee1bdc32c4c127d72e4dc9292ba19bca653dc5e3715921a059a96c79a125  outputs/lm/period_w4_livro78.txt
7612318f361a387b9df26354008f744fcd2b5b441732d271160138085a913959  outputs/lm/kn3_w1.bin
782260bbcec940dfb5193cc97b76dbd9686ed3f76c51d6d0b329d7b19c8b080b  outputs/lm/kn3_w4b8.bin
EOF
  sha256sum -c /content/lm.sha
  touch /content/setup_done
fi
echo "PART setup ${SECONDS}s"
export PY=/content/venv/bin/python DPY=/content/dvenv/bin/python
if [ MODE_HERE = smoke ]; then
  O=outputs/x02_smoke; rm -rf $O; mkdir -p $O; ln -s ../x02/tune $O/tune; mkdir -p $O/eval; ln -s ../../x02/eval/post $O/eval/post
  export X02_RECS=r4_traprinq X02_N=2 X02_N2=1
else
  O=outputs/x02; export X02_SHARDS=${X02_SHARDS:-1}
fi
echo SHA_HERE > $O/git_sha.txt
sha256sum tools/run_x02.sh tools/x02_report.py tools/baseline_search.py tools/baseline_pyctcdecode.py tools/charmap.py tools/score_search.py tools/score_lines.py > $O/sha_tools.txt
(uv pip freeze -p /content/venv/bin/python > $O/env_pipeline.txt; uv pip freeze -p /content/dvenv/bin/python > $O/env_decoders.txt; nproc; free -g; lscpu | grep 'Model name'; hostname) > $O/machine.txt 2>&1 || true
SECONDS=0; /usr/bin/time -v bash tools/run_x02.sh select $O 2> $O/time_select.txt; echo "PART select ${SECONDS}s"
grep -E 'Elapsed|Maximum resident' $O/time_select.txt | sed 's/^/  select /'
SECONDS=0; /usr/bin/time -v bash tools/run_x02.sh eval $O 2> $O/time_eval.txt; echo "PART eval ${SECONDS}s"
grep -E 'Elapsed|Maximum resident' $O/time_eval.txt | sed 's/^/  eval /'
if [ MODE_HERE = full ]; then SECONDS=0; bash tools/run_x02.sh report $O; echo "PART report ${SECONDS}s"; fi
'''.replace('MODE_HERE', MODE).replace('SHA_HERE', SHA)
open(f'/content/x02_{MODE}.sh', 'w').write(JOB)
(W / 'outputs').mkdir(exist_ok=True)
subprocess.Popen(['bash', '-c', f'(bash /content/x02_{MODE}.sh && echo JOB_OK || echo JOB_FAILED) > {LOG} 2>&1'],
                 start_new_session=True, cwd=str(W))
print(f'x02 {MODE} job started', subprocess.run('nproc', capture_output=True, text=True).stdout.strip())
