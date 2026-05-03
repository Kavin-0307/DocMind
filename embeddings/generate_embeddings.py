from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(chunks:list[dict])->list[dict]:
    text=[chunk["text"] for chunk in chunks]
    embeddings=model.encode(text)
    embeddings=embeddings.astype("float32")
    for i,embedding in enumerate(embeddings):
        chunks[i]["embeddings"]=embedding
    return chunks
