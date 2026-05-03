import faiss
import numpy as np
def search(query:str,model,index,chunks,k:int):
    vectors=model.encode([query])
    vectors=vectors.astype("float32")
    D,I=index.search(vectors,k)
    indices=I[0]#Rename later
    return [chunks[i] for i in indices]