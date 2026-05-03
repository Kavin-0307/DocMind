from vector_store.search import search
from vector_store.reranker import rerank

def retrieve(query, model, index, chunks, k=5):
    # Stage 1: FAISS (candidate generation)
    # Cap k_candidates at index.ntotal to avoid faiss.Exception on small documents.
    # IndexFlatL2 errors (does not pad) if k > number of indexed vectors.
    k_candidates = min(50, index.ntotal)
    if k_candidates == 0:
        return []
    candidates = search(query, model, index, chunks, k=k_candidates)

    # Stage 2: Reranking
    results = rerank(query, candidates, top_k=k)
    return results