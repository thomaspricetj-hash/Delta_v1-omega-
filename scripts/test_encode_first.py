import sys
sys.path.insert(0,'C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
from pathlib import Path
import json, pickle, time
base=Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
m=base/'runtime'/'cache'/'texts_chunk_0.manifest'
man=json.load(open(m))
parts=man['parts']
texts=pickle.load(open(base/'runtime'/'cache'/parts[0],'rb'))
from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
tok=get_codebook_tokenizer()
t0=time.time()
ids=tok.encode(texts[0])
print('len', len(ids), 'elapsed', time.time()-t0)
print('words', len(texts[0].split()))
