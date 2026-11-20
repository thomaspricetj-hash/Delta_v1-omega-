p='C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/runtime/embeddings/embed_store.bd2.bd2'
c=0
with open(p,'r',encoding='utf-8') as f:
    for line in f:
        c+=1
print(c)