"""
Skim layer for Translation OS
Speed-read style rapid scan: entities, intent, keywords, summary
"""
import re
from collections import Counter

STOPWORDS = set("""
a an the and or but if then else for with without in on at to from by is are was were be been being
of to the it its that this these those
""".split())

def _extract_entities(text):
    # Simple heuristic: capitalized words, QIDs, numbers
    words = re.findall(r"[A-Z][a-z]+|[A-Z]{2,}|Q\d+|[\d\.]+", text)
    # filter stop-like
    ents = [w for w in words if w.lower() not in STOPWORDS]
    return list(dict.fromkeys(ents))[:20]

def _extract_keywords(text, top_k=10):
    words = re.findall(r"\b\w+\b", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    cnt = Counter(words)
    return [w for w,_ in cnt.most_common(top_k)]

def _classify_intent(text):
    t = text.lower()
    if t.startswith("what is") or t.startswith("explain"):
        return "definition"
    if "what do" in t and "have in common" in t:
        return "comparison"
    if "why" in t or "how" in t:
        return "causal"
    if "summarize" in t or "summar" in t:
        return "summarize"
    if "would" in t and "likely" in t:
        return "generalization"
    if "what would happen if" in t:
        return "novel_combination"
    return "general"

def _summarize(text, max_words=20):
    words = text.split()
    if len(words) <= max_words:
        return text
    # Simple leading summary
    return " ".join(words[:max_words]) + " ..."

def skim(text: str):
    entities = _extract_entities(text)
    keywords = _extract_keywords(text)
    intent = _classify_intent(text)
    summary = _summarize(text)
    return {
        "entities": entities,
        "keywords": keywords,
        "intent": intent,
        "summary": summary,
        "length_words": len(text.split())
    }

if __name__ == "__main__":
    print(skim("What is a dog? Dogs are mammals and can be pets."))
