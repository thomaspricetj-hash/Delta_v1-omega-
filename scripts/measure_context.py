import sys
sys.path.insert(0,'C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/infrastructure/chunk_retriever')
from chunk_retriever import build_layered_context
# Use a sample corpus - let's use a small sample from the embedding store
corpus = "This is a sample corpus text for testing the layered context builder with multilayered weave lattice and sliding window expansion. " * 1000
query = "What is the capital of France?"
ctx, metrics = build_layered_context(query, corpus, layers=4)
print('Metrics:')
for k,v in metrics.items():
    print(f'{k}: {v}')
print('Context length chars:', len(ctx))
