import os
path='C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/runtime/embeddings/embed_store.jsonl.bak'
size=os.path.getsize(path)
total_bytes=0
count=0
with open(path,'r',encoding='utf-8',errors='ignore') as f:
    for i,line in enumerate(f):
        if i>=1000:
            break
        total_bytes+=len(line)
        count+=1
avg=total_bytes/count if count else 1
est_lines=size/avg
print(f'avg line bytes {avg:.1f}')
print(f'estimated total lines {est_lines:.0f}')
# count shards
shard_dir='C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/runtime/embeddings/shards'
import glob
shards=glob.glob(shard_dir+'/embed_store_shard_*.bd2.bd2')
print(f'shards created {len(shards)}')
processed_entries=len(shards)*5000
print(f'processed entries approx {processed_entries}')
pct=processed_entries/est_lines*100 if est_lines else 0
print(f'approx % complete {pct:.2f}%')