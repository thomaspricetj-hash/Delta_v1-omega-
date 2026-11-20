"""
Wikidata -> Q/A generator with paraphrasing for model-like training.
"""
import random

TEMPLATES = {
    "P31": [  # instance of
        "What is {subject} an instance of?",
        "What type of thing is {subject}?",
        "Which class does {subject} belong to?",
    ],
    "P17": [  # country
        "Which country is {subject} in?",
        "What country is {subject} part of?",
    ],
    "P1082": [  # population
        "What is the population of {subject}?",
        "How many people live in {subject}?",
    ],
    "P625": [  # coordinates
        "Where is {subject} located?",
    ],
    "P1376": [  # capital
        "What is the capital of {subject}?",
        "Which city is the capital of {subject}?",
    ],
}

def generate_qa(entity_label, prop_id, value_label, negative=False):
    subject = entity_label
    value = str(value_label)
    templates = TEMPLATES.get(prop_id, [
        "What is {subject} {prop}?",
        "Tell me about {subject} {prop}.",
    ])
    # Diverse paraphrasing
    q = random.choice(templates).format(subject=subject, prop=prop_id)
    # Add slight variation
    if random.random() < 0.3:
        q = q.replace("What is", "Can you tell me what is").replace("What", "Which")
    if negative:
        # Generate plausible wrong answer
        # Simple negative: swap with a random common value
        wrong_pool = ["Unknown", "N/A", "None", "Other"]
        value = random.choice(wrong_pool)
    return q, value

def process_wikidata_item(item):
    """
    item: dict with 'label', 'claims': {prop_id: [values]}
    Returns list of (q,a) pairs
    """
    label = item.get('label','')
    pairs = []
    claims = item.get('claims', {})
    for prop, vals in claims.items():
        for v in vals[:2]:
            q,a = generate_qa(label, prop, v)
            pairs.append((q,a))
    return pairs
