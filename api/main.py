from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
from labeling.label_generator import generate_labels
logger = logging.getLogger(__name__)
from keywords.rake_extractor import extract_keywords
from chunking.semantic_chunker import semantic_chunk_text
from vector_store.retrieve import retrieve
from vector_store.index import build_faiss_index
from embeddings.generate_embeddings import generate_embeddings
from similarity.structure_chunks import structure_chunks
INDEXES_DIR = Path("./indexes")
INDEXES_DIR.mkdir(parents=True, exist_ok=True)
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
        summary = _extract_summary(raw_chunks, model)

        # split chunks → sentences
        all_sentences = []
        for chunk in raw_chunks:
            sents = [s.strip() for s in chunk.split('.') if len(s.strip()) > 10]
            all_sentences.extend(sents)

        important_points = _select_important_points(all_sentences, top_n=5)
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

def _extract_summary(chunks: list[str], model, n_sentences: int = 3) -> str:
    if not chunks:
        return ""

    import re
    all_sentences = []

    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 10])

    if len(all_sentences) <= n_sentences:
        return " ".join(all_sentences)

    embeddings = model.encode(all_sentences, convert_to_numpy=True).astype("float32")

    centroid = embeddings.mean(axis=0, keepdims=True)
    scores = cosine_similarity(embeddings, centroid).flatten()

    top_indices = sorted(np.argsort(scores)[-n_sentences:])
    return " ".join(all_sentences[i] for i in top_indices)
def _select_important_points(sentences: list[str], top_n: int = 5) -> list[str]:
    if not sentences:
        return []

    if len(sentences) <= top_n:
        return sentences

    result = generate_labels(sentences)
    scores = result["scores"]

    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    )

    return [sentences[i] for i in top_indices]
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
            "chunks": embedded,
            "created_at": datetime.now().isoformat()
        }

        # SAVE TO DISK
        save_index(req.session_id, indexes[req.session_id])
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
@app.on_event("startup")
async def load_indexes():
    logger.info("Loading indexes from disk...")

    for file in INDEXES_DIR.glob("*.pkl"):
        try:
            with open(file, "rb") as f:
                session_id = file.stem
                indexes[session_id] = pickle.load(f)
                logger.info(f"Loaded index: {session_id}")
        except Exception as e:
            logger.error(f"Failed loading {file}: {e}")
def save_index(session_id: str, data: dict):
    try:
        file_path = INDEXES_DIR / f"{session_id}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved index: {session_id}")
    except Exception as e:
        logger.error(f"Failed saving index {session_id}: {e}")
@app.get("/")
async def health():
    return {
        "status": "DocMind ML Layer Active",
        "engines": ["Processing", "Retrieval"]
    }