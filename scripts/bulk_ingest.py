import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'orchestrator'))
sys.path.insert(0, str(BASE))
import master_orchestrator_v0 as m
from infrastructure.task_manager.task_manager import get_task_manager

tm = get_task_manager()

facts = []

for i in range(1, 5001):
    facts.append((f"What is {i} plus {i}?", str(i + i)))
    facts.append((f"What is {i} times 2?", str(i * 2)))

capitals = [
    ("France","Paris"), ("Germany","Berlin"), ("Italy","Rome"), ("Spain","Madrid"),
    ("United Kingdom","London"), ("Canada","Ottawa"), ("Australia","Canberra"),
    ("Japan","Tokyo"), ("China","Beijing"), ("India","New Delhi"),
    ("Brazil","Brasilia"), ("Russia","Moscow"), ("South Africa","Pretoria"),
    ("Mexico","Mexico City"), ("United States","Washington DC")
]
for c,cap in capitals:
    facts.append((f"What is the capital of {c}?", cap))

science = [
    ("What is the chemical symbol for hydrogen?","H"),
    ("What is the chemical symbol for oxygen?","O"),
    ("What is the chemical symbol for carbon?","C"),
    ("What is the speed of light in m/s?","299792458"),
    ("What is the acceleration due to gravity on Earth in m/s^2?","9.8"),
    ("What is the atomic number of helium?","2"),
]
facts.extend(science)

history = [
    ("Who discovered penicillin?","Alexander Fleming"),
    ("In which year did World War II end?","1945"),
    ("Who was the first person to walk on the moon?","Neil Armstrong"),
]
facts.extend(history)

while len(facts) < 20000:
    i = len(facts)
    facts.append((f"What is {i} mod 10?", str(i % 10)))

import time
CHUNK_SIZE = 500
print(f"Teaching {len(facts)} facts in chunks of {CHUNK_SIZE}...")

def teach_one(idx, q, a):
    m.teach_full(q, a)

for chunk_start in range(0, len(facts), CHUNK_SIZE):
    chunk = facts[chunk_start:chunk_start+CHUNK_SIZE]
    print(f"Submitting chunk {chunk_start//CHUNK_SIZE + 1} with {len(chunk)} facts")
    for idx,(q,a) in enumerate(chunk, chunk_start+1):
        tm.submit(teach_one, idx, q, a)
    # Pause to let workers drain and caches offload
    time.sleep(2)
    # Force a small GC hint by sleeping longer every 5 chunks
    if (chunk_start // CHUNK_SIZE + 1) % 5 == 0:
        time.sleep(5)
    print(f"Chunk {chunk_start//CHUNK_SIZE + 1} submitted")

print("Bulk ingest completed")
