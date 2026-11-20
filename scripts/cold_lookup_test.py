import json, time, sys
sys.path.insert(0,'C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/infrastructure/embeddings')
from embeddings import embed
index_path='C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/runtime/embeddings/embed_store_index.json'
with open(index_path) as f:
    idx=json.load(f)
keys=list(idx.keys())
if not keys:
    print('no keys')
    sys.exit()
key=keys[0]
print('Testing cold lookup for key:', key[:80])
t0=time.perf_counter()
vec=embed(key)
t1=time.perf_counter()
print('elapsed ms', (t1-t0)*1000)
print('vec shape', len(vec))
