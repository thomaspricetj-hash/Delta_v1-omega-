import sys
sys.path.insert(0,'C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
import time
tok=get_codebook_tokenizer()
t0=time.time()
ids=tok.encode('The quick brown fox jumps over the lazy dog')
print('len', len(ids), 'elapsed', time.time()-t0)
