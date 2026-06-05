import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate(sentences: list[str], model, threshold: float = 0.85) -> list[str]:
    if not sentences:
        return []
    embeddings = model.encode(sentences, convert_to_numpy=True).astype("float32")
    kept_indices = []
    kept_embeddings = []
    for i, emb in enumerate(embeddings):
        if not kept_embeddings:
            kept_indices.append(i); kept_embeddings.append(emb); continue
        sims = cosine_similarity([emb], kept_embeddings)[0]
        if sims.max() < threshold:
            kept_indices.append(i); kept_embeddings.append(emb)
    return [sentences[i] for i in kept_indices]