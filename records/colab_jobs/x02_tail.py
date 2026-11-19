import subprocess
from pathlib import Path
print('VM_ALIVE')
for log in sorted(Path('/content/x02/outputs').glob('job_*.log')):
    lines = log.read_text(errors='replace').splitlines()
    print(log.name)
    print('\n'.join(l for l in lines if l.startswith(('PART', 'STAGE', 'REPRO', 'FROZEN', 'DECODED', 'JOB_', 'GRID', 'PAIR', '  '))))
    print('\n'.join(lines[-4:]))
print(subprocess.run('uptime', shell=True, capture_output=True, text=True).stdout)
