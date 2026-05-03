from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# your modules
from vector_store.retrieve import retrieve
from chunking.semantic_chunker import semantic_chunk_text
from similarity.structure_chunks import structure_chunks
from embeddings.generate_embeddings import generate_embeddings
from vector_store.index import build_faiss_index


app = FastAPI()

model = None
index = None
chunks = None


 
@app.on_event("startup")
def load_model():
    global model
    model = SentenceTransformer("all-MiniLM-L6-v2")

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
    results = retrieve(
        query=req.question,
        model=model,
        index=index,
     chunks=chunks,
     k=req.k
    )

    return {
    "query": req.question,
    "results": [r["text"] for r in results]
}