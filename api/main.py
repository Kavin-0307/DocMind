from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# your modules
from similarity.semantic_chunk import semantic_chunk_text
from similarity.structure_chunks import structure_chunks
from embeddings.generate_embeddings import generate_embeddings
from faiss.index import build_faiss_index
from faiss.search import search

app = FastAPI()

model = None
index = None
chunks = None


# ✅ FIX 1: load model at startup
@app.on_event("startup")
def load_model():
    global model
    model = SentenceTransformer("all-MiniLM-L6-v2")


# ✅ FIX 2: use POST body instead of GET params
class QueryRequest(BaseModel):
    question: str
    k: int = 3


# ✅ FIX 3: full semantic pipeline
@app.post("/upload")
async def build_system(file: UploadFile):
    global index, chunks

    content = await file.read()
    text = content.decode("utf-8")

    # basic sentence split (can improve later)
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]

    raw_chunks = semantic_chunk_text(sentences, threshold=0.7)
    structured = structure_chunks(raw_chunks)
    embedded = generate_embeddings(structured)

    index, _ = build_faiss_index(embedded)
    chunks = embedded

    return {"status": "Success", "chunks_indexed": len(chunks)}



@app.post("/query")
async def query_system(req: QueryRequest):
    global index, chunks, model

    if index is None or chunks is None:
        raise HTTPException(status_code=400, detail="Upload a document first")

    # clean float32 handling
    query_vec = model.encode([req.question]).astype("float32")

    D, I = index.search(query_vec, req.k)

    results = []
    for idx in I[0]:
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx]["text"])  

    return {"query": req.question, "results": results}