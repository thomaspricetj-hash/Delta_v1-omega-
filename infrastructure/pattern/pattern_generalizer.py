"""
Pattern Generalizer V1
Extracts templated patterns with variables for generalisation.
"""
import re
from pathlib import Path
import json

TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "runtime" / "logs" / "pattern_templates.json"

TEMPLATES = [
    {
        "name": "derivative_power",
        "regex": r"(derivative of|differentiate|d/dx of)\s+x\^?(\d+)",
        "answer_template": lambda m: f"{m.group(2)}x^{int(m.group(2))-1}",
        "type": "math"
    },
    {
        "name": "derivative_power_alt",
        "regex": r"differentiate\s+x to the (\d+)\w*\s+power",
        "answer_template": lambda m: f"{m.group(1)}x^{int(m.group(1))-1}",
        "type": "math"
    },
    {
        "name": "derivative_ln_power",
        "regex": r"(derivative of|find d/dx of)\s+ln\(x\^(\d+)\)",
        "answer_template": lambda m: f"{m.group(2)}/x",
        "type": "math"
    },
    {
        "name": "derivative_ln_power_alt",
        "regex": r"find d/dx of\s+ln\(x squared\)",
        "answer_template": lambda m: "2/x",
        "type": "math"
    },
    {
        "name": "integral_power",
        "regex": r"integral of\s+(\d+)x\^(\d+)\s+dx",
        "answer_template": lambda m: (
            f"x^{int(m.group(2))+1}" if int(m.group(1)) == int(m.group(2))+1 else
            f"{(int(m.group(1))/(int(m.group(2))+1)):.0f}x^{int(m.group(2))+1}" if int(m.group(1)) % (int(m.group(2))+1) == 0 else
            f"{int(m.group(1))/(int(m.group(2))+1)}x^{int(m.group(2))+1}"
        ),
        "type": "math"
    },
    {
        "name": "integral_power_cubed",
        "regex": r"integral of\s+(\d+)\s*x\s+cubed\s+dx",
        "answer_template": lambda m: (
            f"x^{4}" if int(m.group(1)) == 4 else
            f"{(int(m.group(1))/4):.0f}x^4" if int(m.group(1)) % 4 == 0 else
            f"{int(m.group(1))/4}x^4"
        ),
        "type": "math"
    },
    {
        "name": "percent_of",
        "regex": r"(what is|\d+ percent of|percent of)\s+(\d+)%\s+of\s+(\d+)",
        "answer_template": lambda m: str(int(m.group(2))*int(m.group(3))//100) if len(m.groups())>=3 else str(int(m.group(1))*int(m.group(2))//100),
        "type": "math"
    },
    {
        "name": "percent_of_alt",
        "regex": r"(\d+) percent of (\d+)\s+equals?",
        "answer_template": lambda m: str(int(m.group(1))*int(m.group(2))//100),
        "type": "math"
    },
    {
        "name": "distance_speed_time",
        "regex": r"car (travels|drives)\s+(\d+)\s*km/?h\s+for\s+(\d+)\s*hours",
        "answer_template": lambda m: str(int(m.group(2))*int(m.group(3))),
        "type": "math"
    },
    {
        "name": "distance_speed_time_alt",
        "regex": r"car drives\s+(\d+)\s+km per hour for\s+(\d+)\s*hours",
        "answer_template": lambda m: str(int(m.group(1))*int(m.group(2))),
        "type": "math"
    },
    {
        "name": "distance_speed_time_vehicle",
        "regex": r"(vehicle|car|truck) travels\s+(\d+)\s*km/?h\s+for\s+(\d+)\s*hours",
        "answer_template": lambda m: str(int(m.group(2))*int(m.group(3))),
        "type": "math"
    },
    {
        "name": "power",
        "regex": r"(what is|\d+ raised to)\s+(\d+)\s*(\^|raised to)\s*(\d+)",
        "answer_template": lambda m: str(int(m.group(2))**int(m.group(4))) if m.group(1)=='what is' else str(int(m.group(1))**int(m.group(4))),
        "type": "math"
    },
    {
        "name": "power_simple",
        "regex": r"(\d+) raised to (\d+)",
        "answer_template": lambda m: str(int(m.group(1))**int(m.group(2))),
        "type": "math"
    },
    {
        "name": "sqrt",
        "regex": r"(square root of|sqrt)\s+(\d+)",
        "answer_template": lambda m: str(int(float(int(m.group(2))**0.5))) if float(int(m.group(2))**0.5).is_integer() else str(float(int(m.group(2))**0.5)),
        "type": "math"
    },
]

def _load_templates():
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("templates", [])
    return []

def _save_templates(templates):
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"templates": templates}, f, indent=2)

def generalize(query):
    q = query.lower()
    for t in TEMPLATES + _load_templates():
        m = re.search(t["regex"], q)
        if m:
            try:
                ans = t["answer_template"](m)
                return {"template": t["name"], "type": t["type"], "answer": ans, "match": m.group(0)}
            except Exception:
                continue
    return None

def learn_template(query, answer):
    # Simple heuristic to add new template if query matches existing pattern family
    # For now just store pair for later manual review
    data = {"query": query, "answer": answer, "timestamp": __import__("time").time()}
    # Append to learning log
    log_path = Path(__file__).resolve().parent.parent.parent / "runtime" / "logs" / "pattern_learning_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data)+"\n")
    return True

def get_templates():
    return TEMPLATES + _load_templates()
