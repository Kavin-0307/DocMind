from vector_store.search import search
from vector_store.reranker import rerank

def retrieve(query, model, index, chunks, k=5):
    # Stage 1: FAISS (candidate generation)
    candidates = search(query, model, index, chunks, k=50)

    # Stage 2: Reranking
    results = rerank(query, candidates, top_k=k)
    print("FAISS results:", candidates)
    print("RERANKED:", results)
    return results