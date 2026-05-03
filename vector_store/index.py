import faiss
import numpy as np
def build_faiss_index(chunks:list[dict]):
    embeddings=[chunk["embedding"] for chunk in chunks]

    embeddings_np=np.array(embeddings).astype("float32")
    d=embeddings_np.shape[1]
    index=faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    id_map=[chunk["chunk_id"] for chunk in chunks]
    return index,id_map