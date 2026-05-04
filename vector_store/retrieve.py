from vector_store.search import search
from vector_store.reranker import rerank
import logging
logger=logging.getLogger(__name__)
def retrieve(query, model, index, chunks, k=5):
    # Stage 1: FAISS (candidate generation)
    # Cap k_candidates at index.ntotal to avoid faiss.Exception on small documents.
    # IndexFlatL2 errors (does not pad) if k > number of indexed vectors.
    k_candidates = min(50, index.ntotal)
    if k_candidates == 0:
        logger.warning("FAISS index is empty-nothing to retrieve")
        return []
    k=min(k,k_candidates)
    candidates=search(query,model,index,chunks,k=k_candidates)
    logger.debug("FAISS returned %d candidates for query: %s", len(candidates), query)

    # Stage 2: Reranking
    results = rerank(query, candidates, top_k=k)
    logger.debug("Reranker returned %d results", len(results))

    return results