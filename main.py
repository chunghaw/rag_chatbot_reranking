# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import vector_service

# force testing fallback if TESTING=true
if os.getenv("TESTING", "").lower() == "true":
    vector_service.initialize_milvus()
    vector_service.connect_to_milvus()
    # vector_service.use_fallback = True  # not needed here: search/ingest are in‐mem

app = FastAPI(title="Week2 RAG API", version="1.0.0")

class IngestRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)

class AskResponse(BaseModel):
    answer: str
    sources: list

@app.on_event("startup")
def startup():
    vector_service.initialize_milvus()
    vector_service.connect_to_milvus()
    # no real collections needed in testing

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ingest", status_code=204)
def ingest(req: IngestRequest):
    try:
        # write into both “sparse” and “dense” in real mode, but here same store
        vector_service.ingest_milvus(req.text, collection_name="sparse")
        vector_service.ingest_milvus(req.text, collection_name="dense")
        return Response(status_code=204)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        # empty → validation will normally catch, but just in case:
        raise HTTPException(status_code=400, detail="Empty question")
    # fetch top‐10 from each
    sparse = vector_service.search_milvus(q, "sparse", top_k=10)
    dense  = vector_service.search_milvus(q, "dense",  top_k=10)
    # merge & dedupe
    merged = {}
    for txt,score in sparse + dense:
        merged[txt] = max(merged.get(txt,0.0), score)
    candidates = sorted(merged.items(), key=lambda kv:kv[1], reverse=True)
    if not candidates:
        # no match
        return AskResponse(answer="I don't know.", sources=[])
    # take the top 5
    top5 = candidates[:5]
    answer = top5[0][0]
    sources = [{"text":txt,"score":sc} for txt,sc in top5]
    return AskResponse(answer=answer, sources=sources)