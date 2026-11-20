"""
Fact KB lookup for capitals, chemical symbols, atomic numbers, painters
"""
import json
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent / "fact_kb.json"

def _load():
    if KB_PATH.exists():
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"capitals": {}, "chemical_symbols": {}, "atomic_numbers": {}, "painters": {}}

def lookup(query):
    import re
    q = query.lower()
    kb = _load()
    # capital
    if "capital of" in q or "capital city of" in q:
        # extract country
        m = re.search(r"capital (?:of|city of)\s+(.+)", q)
        if m:
            country = m.group(1).strip().rstrip("?")
            # remove trailing words
            country = re.sub(r"\s+and.*", "", country)
            # simple lookup
            cap = kb.get("capitals", {}).get(country)
            if cap:
                return {"source": "fact_kb_capital", "answer": cap}
    # chemical symbol
    if "chemical symbol for" in q or "symbol for" in q:
        m = re.search(r"(?:chemical symbol for|symbol for)\s+(.+)", q)
        if m:
            elem = m.group(1).strip().rstrip("?")
            sym = kb.get("chemical_symbols", {}).get(elem)
            if sym:
                return {"source": "fact_kb_chemical", "answer": sym}
    # atomic number
    if "atomic number of" in q:
        m = re.search(r"atomic number of\s+(.+)", q)
        if m:
            elem = m.group(1).strip().rstrip("?")
            num = kb.get("atomic_numbers", {}).get(elem)
            if num:
                return {"source": "fact_kb_atomic", "answer": num}
    # painter
    if "who painted" in q or "painter of" in q:
        m = re.search(r"(?:who painted|painter of)\s+(.+)", q)
        if m:
            art = m.group(1).strip().rstrip("?")
            painter = kb.get("painters", {}).get(art)
            if painter:
                return {"source": "fact_kb_painter", "answer": painter}
    return None
