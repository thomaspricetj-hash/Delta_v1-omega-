
"""
Wikidata ingest watchdog with RAM guard and auto-restart
"""
import subprocess, time, psutil, sys
from pathlib import Path

SCRIPT = Path("scripts/learn_wikidata_loop_byteoffset.py")
RAM_LIMIT = 80  # percent
CHECK_INTERVAL = 10

def ram_ok():
    try:
        return psutil.virtual_memory().percent < RAM_LIMIT
    except:
        return True

def run():
    while True:
        if not ram_ok():
            print("[Watchdog] RAM high, pausing 60s")
            time.sleep(60)
            continue
        try:
            proc = subprocess.Popen([sys.executable, str(SCRIPT)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # stream output
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                print(line, end="")
                if "Error:" in line or "bytearray index out of range" in line:
                    print("[Watchdog] Error detected, restarting in 10s")
                    proc.terminate()
                    break
                if not ram_ok():
                    print("[Watchdog] RAM high during run, pausing")
                    proc.terminate()
                    break
            ret = proc.wait()
            if ret != 0:
                print("[Watchdog] Process exited with code", ret, "restarting in 10s")
        except Exception as e:
            print("[Watchdog] Exception", e)
        time.sleep(10)

if __name__ == "__main__":
    run()
