"""
Verify hybrid activated modules actually import and run basic health checks
"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "orchestrator"))
sys.path.insert(0, str(BASE / "infrastructure"))
sys.path.insert(0, str(BASE / "scripts"))

import importlib.util
import json

def try_import(path):
    try:
        spec = importlib.util.spec_from_file_location("mod", path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return True, None
    except Exception as e:
        return False, str(e)
    return False, "no spec"

results = []
# Infrastructure dirs
infra_root = BASE / "infrastructure"
for entry in sorted(infra_root.iterdir()):
    if entry.is_dir():
        # Try __init__.py
        init = entry / "__init__.py"
        if init.exists():
            ok, err = try_import(init)
            results.append({"type":"infrastructure_pkg","name":entry.name,"ok":ok,"error":err})
        # Try main file
        py_files = list(entry.glob("*.py"))
        for pf in py_files[:3]:
            ok, err = try_import(pf)
            results.append({"type":"infrastructure_file","name":f"{entry.name}/{pf.name}","ok":ok,"error":err})

# Scripts
scripts_root = BASE / "scripts"
for f in sorted(scripts_root.glob("*.py")):
    if f.name.startswith("__"):
        continue
    ok, err = try_import(f)
    results.append({"type":"script","name":f.name,"ok":ok,"error":err})

ok_count = sum(1 for r in results if r["ok"])
fail_count = len(results) - ok_count
print(f"Checked {len(results)} modules: {ok_count} OK, {fail_count} FAIL")
# Save report
out_path = BASE / "runtime" / "logs" / "hybrid_verification.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r)+"\n")
print(f"Report written to {out_path}")
# Print failures
for r in results:
    if not r["ok"]:
        print(f"FAIL {r['type']} {r['name']}: {r['error'][:200]}")
