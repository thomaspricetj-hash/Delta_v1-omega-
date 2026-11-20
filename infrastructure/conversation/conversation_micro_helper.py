"""
Conversation Micro Helper V0
Fast, lightweight responses for natural dialogue.
Rewards/penalizes turns and updates maps.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
RUNTIME = BASE / "runtime"

def _load_maps():
    # Minimal in-memory stubs; real maps are updated via runtime imports
    return {}

def detect_intent(q):
    ql = q.lower().strip()
    greetings = {"hi","hello","hey","yo","good morning","good afternoon","good evening"}
    thanks = {"thanks","thank you","thank u","thx"}
    yes_no = {"yes","no","yep","nope","sure","ok","okay"}
    if ql in greetings:
        return "greeting"
    if ql in thanks:
        return "thanks"
    if ql in yes_no:
        return "acknowledge"
    if ql.startswith("what are") or ql.startswith("what is") or ql.endswith("?"):
        return "question"
    if any(t in ql for t in ["tell me","can you","could you","explain","how do","why do"]):
        return "request"
    return "chat"

def respond(q, history=None):
    # Try learned response first
    try:
        # import semantic map predictor
        import sys
        from pathlib import Path
        base = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(base / "runtime"))
        from semantic_map import predict_semantic
        sem = predict_semantic(q, top_k=1, threshold=0, min_words=1)
        if sem and sem.get("candidates"):
            top = sem["candidates"][0]
            if top.get("score",0) >= 5.0:
                return {"intent": "learned", "answer": top.get("answer"), "source": "conversation_micro_helper_learned"}
    except Exception:
        pass
    intent = detect_intent(q)
    responses = {
        "greeting": "Hey! How can I help you today?",
        "thanks": "You’re welcome! Anything else you’d like to explore?",
        "acknowledge": "Got it. Let me know if you want to dive deeper.",
        "question": "Good question. Let me think about that…",
        "request": "Sure, I can help with that. What specifically would you like to focus on?",
        "chat": "Interesting. Tell me more about that."
    }
    answer = responses.get(intent, "Tell me more.")
    return {"intent": intent, "answer": answer, "source": "conversation_micro_helper"} 

def reward_turn(query, answer, user_feedback=None):
    # Simple reward logging
    reward = 1.0 if user_feedback in [None, "positive", "good"] else -0.5
    log_path = BASE / "reports" / "conversation_rewards.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "query": query,
        "answer": answer,
        "reward": reward,
        "feedback": user_feedback
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    # Update maps if runtime available
    try:
        sys_path = [str(RUNTIME)]
        import sys
        sys.path.insert(0, str(RUNTIME))
        from semantic_map import update_map
        from pattern_mapper import update as pattern_update
        from next_word_map import update as next_word_update
        from statistical_word_relations import update as stat_rel_update
        from word_meaning_mapper import learn_from_conversation
        update_map(query, answer, weight=reward)
        pattern_update(query, answer, reward=reward)
        next_word_update(query, answer, reward=reward)
        stat_rel_update(query, answer)
        learn_from_conversation(query, answer)
    except Exception:
        pass
    return {"reward": reward}

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:])
    print(json.dumps(respond(q), indent=2))
