import sys
sys.path.insert(0,'C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
from pathlib import Path
import json, pickle, torch
base=Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
m=base/'runtime'/'cache'/'texts_chunk_0.manifest'
man=json.load(open(m))
parts=man['parts']
texts=[]
for p in parts[:1]:
    texts.extend(pickle.load(open(base/'runtime'/'cache'/p,'rb')))
print('loaded', len(texts))
from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
tok=get_codebook_tokenizer()
n=1000
tokenized=[]
for i in range(n):
    ids=tok.encode(texts[i])
    tokenized.append(ids)
    if i%100==0:
        print(i)
torch.save(tokenized, base/'runtime'/'cache'/'tokens_sample_1k.pt')
print('saved')
