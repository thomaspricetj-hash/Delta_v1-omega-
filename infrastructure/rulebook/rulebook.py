"""
Rule Book – explicit, fast, deterministic rules for the orchestrator
Supports pattern matching, validation, and auto-promotion from emergent patterns.
"""
import json, re
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent
STATE_PATH = BASE / "runtime" / "logs" / "rulebook.json"
DEFAULT_RULES = [
    {
        "id": "capital_p36",
        "pattern": r"what is the capital of (.+)",
        "action": "wikidata_lookup",
        "prop": "P36",
        "priority": 100,
        "description": "Capital lookup via Wikidata"
    },
    {
        "id": "favorite_color_persist",
        "pattern": r"my favorite color is (\w+)",
        "action": "store_preference",
        "key": "favorite_color",
        "priority": 90,
        "description": "Persist user favorite color"
    },
    {
        "id": "color_query",
        "pattern": r"what is my favorite color",
        "action": "recall_preference",
        "key": "favorite_color",
        "priority": 95,
        "description": "Recall favorite color"
    },
    {
        "id": "arithmetic_sanity",
        "pattern": r"2 \+ 2",
        "action": "return_literal",
        "value": "4",
        "priority": 1000,
        "description": "Prevent 2+2=5"
    }
]

def load_rules():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except:
            pass
    return DEFAULT_RULES

def save_rules(rules):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(rules))

def match_rule(query):
    q = query.lower()
    rules = load_rules()
    matches = []
    for r in rules:
        try:
            m = re.search(r["pattern"], q, re.I)
            if m:
                matches.append((r["priority"], r, m))
        except re.error:
            continue
    if not matches:
        return None
    matches.sort(key=lambda x: x[0], reverse=True)
    _, rule, m = matches[0]
    return {"rule": rule, "match": m}

def apply_rule(rule_hit, query, orchestrator_state=None):
    rule = rule_hit["rule"]
    action = rule.get("action")
    if action == "return_literal":
        return {"response": rule.get("value"), "source": "rulebook"}
    if action == "store_preference":
        key = rule.get("key")
        val = rule_hit["match"].group(1)
        # store in user_preferences via north_star
        try:
            sys_path = str(BASE / "infrastructure" / "north_star")
            import sys
            sys.path.insert(0, sys_path)
            from north_star import update_preferences
            update_preferences({key: val})
        except:
            pass
        return {"response": f"Got it, I'll remember {key} = {val}", "source": "rulebook"}
    if action == "recall_preference":
        key = rule.get("key")
        try:
            sys_path = str(BASE / "infrastructure" / "north_star")
            import sys
            sys.path.insert(0, sys_path)
            from north_star import load_state
            state = load_state()
            val = state.get("user_preferences",{}).get(key)
            if val:
                return {"response": val, "source": "rulebook"}
        except:
            pass
        return {"response": None, "source": "rulebook"}
    if action == "wikidata_lookup":
        # Defer to orchestrator – just signal rule matched
        return {"response": None, "source": "rulebook_hint", "prop": rule.get("prop")}
    return None

def validate_answer(query, answer):
    # Simple sanity checks
    rules = load_rules()
    for r in rules:
        if r.get("action") == "return_literal":
            if re.search(r["pattern"], query, re.I):
                if answer.strip() != r.get("value"):
                    return False, "Rule violation: expected "+r.get("value")
    return True, ""

def promote_pattern(query, answer, confidence=0.9):
    # Auto-promote stable patterns to rules
    rules = load_rules()
    # Simple heuristic: if confidence high and query contains "is"
    if confidence > 0.85 and " is " in query.lower():
        new_rule = {
            "id": f"auto_{abs(hash(query))%100000}",
            "pattern": re.escape(query).replace(r"\ ", r"\s+"),
            "action": "return_literal",
            "value": answer,
            "priority": 50,
            "description": "Auto-promoted from emergent pattern"
        }
        # avoid duplicates
        if not any(r["pattern"] == new_rule["pattern"] for r in rules):
            rules.append(new_rule)
            save_rules(rules)
    return True
