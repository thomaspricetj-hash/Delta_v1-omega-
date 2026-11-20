def extract_structured_facts(query, answer):
    facts = []
    if not answer:
        return facts
    # Pipe separated fields
    if '|' in answer:
        parts = [p.strip() for p in answer.split('|')]
        for p in parts:
            if p:
                facts.append((query, p))
    # Key:Value lines
    if '\n' in answer:
        for line in answer.split('\n'):
            if ':' in line and len(line) < 500:
                k,v = line.split(':',1)
                q2 = f'What is {k.strip()} for {query}?'
                facts.append((q2, v.strip()))
    # Simple list bullets
    if answer.startswith('-') or '\n- ' in answer:
        items = [l.strip('- ').strip() for l in answer.split('\n') if l.strip().startswith('-')]
        for it in items:
            facts.append((f'List item for {query}', it))
    return facts
