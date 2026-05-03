from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

from keywords.rake_extractor import extract_keywords
from chunking.semantic_chunker import semantic_chunk_text
from vector_store.retrieve import retrieve
from vector_store.index import build_faiss_index
from embeddings.generate_embeddings import generate_embeddings
from similarity.structure_chunks import structure_chunks

app = FastAPI(title="DocMind AI Engine")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Per-session storage instead of global state
indexes: dict[str, dict] = {}

class AIRequest(BaseModel):
    text: str

class AIResponse(BaseModel):
    keywords: List[str]
    summary: str
    important_points: List[str]
    revision_sheet: str

class IndexRequest(BaseModel):
    text: str
    session_id: str

class QueryRequest(BaseModel):
    question: str
    session_id: str
    k: int = 3

@app.post("/api/process", response_model=AIResponse)
async def process_lecture(req: AIRequest):
    if not req.text or len(req.text) < 20:
        raise HTTPException(status_code=400, detail="Text too short for analysis")

    try:
        keywords = extract_keywords(req.text, top_n=8)

        sentences = [s.strip() for s in req.text.split('.') if len(s.strip()) > 5]
        raw_chunks = semantic_chunk_text(sentences, threshold=0.7, model=model)

        summary = _extract_summary(raw_chunks)
        important_points = raw_chunks[:5] if len(raw_chunks) > 5 else raw_chunks

        revision_sheet = f"# Revision Roadmap\n\n### 🔴 CORE CONCEPTS\n"
        revision_sheet += "\n".join([f"- {p}" for p in important_points[:3]])
        revision_sheet += f"\n\n### 📝 SUMMARY\n{summary}"

        return AIResponse(
            keywords=keywords,
            summary=summary,
            important_points=important_points,
            revision_sheet=revision_sheet
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")

def _extract_summary(chunks: list[str]) -> str:
    """Extract a summary from chunks by joining and taking first few sentences"""
    import re
    full_text = " ".join(chunks)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    return " ".join(sentences[:3]) if len(sentences) >= 3 else full_text

@app.post("/api/index")
async def index_document(req: IndexRequest):
    """Pre-processes and indexes document for real-time querying"""
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        sentences = [s.strip() for s in req.text.split('.') if len(s.strip()) > 10]
        raw_chunks = semantic_chunk_text(sentences, threshold=0.7, model=model)
        structured = structure_chunks(raw_chunks)
        embedded = generate_embeddings(structured, model=model)

        index, _ = build_faiss_index(embedded)
        indexes[req.session_id] = {
            "index": index,
            "chunks": embedded
        }

        return {
            "status": "Successfully Indexed",
            "chunks": len(embedded),
            "session_id": req.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing Error: {str(e)}")

@app.post("/api/query")
async def query_doc(req: QueryRequest):
    """Answers specific user questions using retrieval"""
    if not req.session_id or req.session_id not in indexes:
        raise HTTPException(
            status_code=400,
            detail="Please upload/index a document first or provide a valid session_id"
        )

    try:
        import time
        start = time.perf_counter()

        session = indexes[req.session_id]
        results = retrieve(
            query=req.question,
            model=model,
            index=session["index"],
            chunks=session["chunks"],
            k=req.k
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return {
            "query": req.question,
            "session_id": req.session_id,
            "top_k": req.k,
            "results": [
                {"text": r["text"], "score": float(r["score"]), "rank": i + 1}
                for i, r in enumerate(results)
            ],
            "metadata": {
                "total_chunks": len(session["chunks"]),
                "retrieval_method": "faiss + cross-encoder",
                "processing_time_ms": elapsed_ms
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Error: {str(e)}")

@app.get("/")
async def health():
    return {
        "status": "DocMind ML Layer Active",
        "engines": ["Processing", "Retrieval"]
    }