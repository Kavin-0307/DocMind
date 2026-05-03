from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(chunks: list[dict]) -> list[dict]:
    texts=[chunk["text"] for chunk in chunks]
    embeddings=model.encode(texts)
    embeddings=embeddings.astype("float32")
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb
    return chunks
