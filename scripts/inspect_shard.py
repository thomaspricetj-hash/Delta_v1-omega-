import json
p="C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1/runtime/embeddings/embed_store.bd2.bd2"
with open(p) as f:
  for i,line in enumerate(f):
    if i<3:
      obj=json.loads(line)
      print(obj.keys())
      print(obj["data"].keys())
    else:
      break