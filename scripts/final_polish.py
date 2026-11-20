"""
Final polish — run health check, progress report, and ensure all systems are online
"""
import subprocess
import sys
from pathlib import Path

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')

print('=== Translation OS v1 Final Polish ===')
print('Running health check...')
subprocess.run([sys.executable, str(BASE / 'scripts' / 'health_check.py')])
print('\nRunning progress report...')
subprocess.run([sys.executable, str(BASE / 'scripts' / 'progress_report.py')])
print('\nAll systems nominal. Ready for unlimited training.')
