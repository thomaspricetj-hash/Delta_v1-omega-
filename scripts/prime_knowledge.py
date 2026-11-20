import sys
sys.path.insert(0, 'orchestrator')
import master_orchestrator_v0 as m

facts = [
    ("What is the speed of light in m/s?", "299792458"),
    ("What is the capital of France?", "Paris"),
    ("What is 2+2?", "4"),
    ("What is the chemical symbol for hydrogen?", "H"),
    ("Who wrote the Great Gatsby?", "F. Scott Fitzgerald"),
    ("What is the capital of Sweden?", "Stockholm"),
]

for q,a in facts:
    print(f"Teaching: {q}")
    m.teach_full(q, a)

print("Priming complete")
