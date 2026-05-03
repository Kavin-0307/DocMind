from sentence_transformers import CrossEncoder

# load once (like your embedding model)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidate_chunks: list[dict], top_k: int = 5):
    """
    Stage 2: Cross-encoder reranking

    Args:
        query: user query
        candidate_chunks: output from FAISS (list of chunk dicts)
        top_k: number of final results

    Returns:
        top_k reranked chunks
    """

    # 1. create (query, chunk_text) pairs
    pairs = [(query, chunk["text"]) for chunk in candidate_chunks]

    # 2. get relevance scores
    scores = reranker_model.predict(pairs)

    # 3. attach scores to chunks
    for i, chunk in enumerate(candidate_chunks):
        chunk["score"] = float(scores[i])

    # 4. sort by score (descending)
    reranked = sorted(candidate_chunks, key=lambda x: x["score"], reverse=True)

    # 5. return top_k
    return reranked[:top_k]