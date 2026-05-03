def generate_embeddings(chunks: list[dict], model=None) -> list[dict]:
    if model is None:
        raise ValueError("A SentenceTransformer model instance must be passed explicitly.")

    texts=[chunk["text"] for chunk in chunks]
    embeddings=model.encode(texts)
    embeddings=embeddings.astype("float32")
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb
    return chunks
