from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

# 📥 Importing your custom modular components
from keywords.rake_extractor import extract_keywords
from chunking.semantic_chunker import semantic_chunk_text
from vector_store.retrieve import retrieve
from vector_store.index import build_faiss_index
from embeddings.generate_embeddings import generate_embeddings
from similarity.structure_chunks import structure_chunks

# Initialize App and Model
app = FastAPI(title="DocMind AI Engine")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Global state for System 2 (Retrieval Engine)
active_index = None
active_chunks = None

# --- DATA SCHEMAS (Synced with SOS Backend) ---

class AIRequest(BaseModel):
    text: str

class AIResponse(BaseModel):
    keywords: List[str]
    summary: str
    important_points: List[str]
    revision_sheet: str

class QueryRequest(BaseModel):
    question: str
    k: int = 3

# --- SYSTEM 1: LECTURE PROCESSING (Summarization & Material Gen) ---

@app.post("/api/process", response_model=AIResponse)
async def process_lecture(req: AIRequest):
    if not req.text or len(req.text) < 20:
        raise HTTPException(status_code=400, detail="Text too short for analysis")

    try:
        # 1. Keywords (System 1)
        keywords = extract_keywords(req.text, top_n=8)

        # 2. Semantic Chunking (For Summary & Important Points)
        sentences = [s.strip() for s in req.text.split('.') if len(s.strip()) > 5]
        raw_chunks = semantic_chunk_text(sentences, threshold=0.7)
        
        # Logic: First chunk is summary, next 5 are important points
        summary = raw_chunks[0] if raw_chunks else "No summary available"
        important_points = raw_chunks[1:6] if len(raw_chunks) > 1 else raw_chunks

        # 3. Revision Sheet Generation
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

# --- SYSTEM 2: QUERY ENGINE (FAISS Retrieval for Chat) ---

@app.post("/api/index")
async def index_document(req: AIRequest):
    """Pre-processes and indexes document for real-time querying"""
    global active_index, active_chunks
    
    try:
        sentences = [s.strip() for s in req.text.split('.') if len(s.strip()) > 10]
        raw_chunks = semantic_chunk_text(sentences, threshold=0.7)
        structured = structure_chunks(raw_chunks)
        embedded = generate_embeddings(structured)
        
        active_index, _ = build_faiss_index(embedded)
        active_chunks = embedded
        return {"status": "Successfully Indexed", "chunks": len(embedded)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing Error: {str(e)}")

@app.post("/api/query")
async def query_doc(req: QueryRequest):
    """Answers specific user questions using retrieval"""
    global active_index, active_chunks
    
    if active_index is None:
        raise HTTPException(status_code=400, detail="Please upload/index a document first")

    try:
        results = retrieve(
            query=req.question,
            model=model,
            index=active_index,
            chunks=active_chunks,
            k=req.k
        )
        return {"results": [r["text"] for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Error: {str(e)}")

@app.get("/")
async def health():
    return {"status": "DocMind ML Layer Active", "engines": ["Processing", "Retrieval"]}
