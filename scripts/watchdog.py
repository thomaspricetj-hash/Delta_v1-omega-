"""
Watchdog for unlimited training loop
Auto-restarts run_all_improvements.py on failure, logs health
"""
import subprocess
import time
import sys
import os
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
LOG = BASE / 'runtime' / 'logs' / 'watchdog.log'

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

def main():
    script = str(BASE / 'scripts' / 'run_all_improvements.py')
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    while True:
        try:
            log('Starting training loop')
            proc = subprocess.Popen([sys.executable, script], cwd=str(BASE), env=env)
            proc.wait()
            log(f'Training loop exited with code {proc.returncode}, restarting in 10s')
            time.sleep(10)
        except Exception as e:
            log(f'Watchdog error: {e}')
            time.sleep(10)

if __name__ == '__main__':
    main()
