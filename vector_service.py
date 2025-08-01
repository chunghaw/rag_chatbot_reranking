# vector_service.py
from dotenv import load_dotenv
load_dotenv()                # MUST be first

import os
import uuid
import numpy as np
from typing import List, Dict
from openai import OpenAI
from pymilvus import (
    connections,
    Collection,
    utility,
    CollectionSchema,
    FieldSchema,
    DataType,
)

# ─────────── Configuration ───────────
MILVUS_URI        = os.getenv("MILVUS_URI")
MILVUS_USER       = os.getenv("MILVUS_USER")
MILVUS_PASSWORD   = os.getenv("MILVUS_PASSWORD")
COLLECTION_NAME   = "ai_chat_knowledge_base"
DIMENSION         = 1536  # text-embedding-ada-002

# In‐memory fallback if Milvus is unreachable
fallback_embeddings: Dict[str, List[float]] = {}
fallback_texts:      Dict[str, str]         = {}
use_fallback = False

# ─────────── Core Helpers ───────────
def connect_to_milvus():
    """Try to connect to Milvus; flip fallback flag on failure."""
    global use_fallback
    try:
        connections.connect(
            alias="default",
            uri = os.getenv("MILVUS_URI"),
            user = os.getenv("MILVUS_USER"),
            password = os.getenv("MILVUS_PASSWORD"),
            secure = True
        )
        print("✅ Connected to Milvus")
        use_fallback = False
    except Exception as e:
        print(f"⚠️ Milvus unreachable: {e}")
        print("→ Falling back to in-memory storage")
        use_fallback = True

def create_collection_if_not_exists():
    """Create the RAG collection (id, text, embedding) if missing."""
    if utility.has_collection(COLLECTION_NAME):
        return
    # Define schema
    schema = CollectionSchema(
        fields=[
            FieldSchema("primary_key",        DataType.VARCHAR,      is_primary=True, max_length=36),
            FieldSchema("text",      DataType.VARCHAR,      max_length=65535),
            FieldSchema("embedding", DataType.FLOAT_VECTOR,  dim=DIMENSION),
        ],
        description="AI Chat Knowledge Base",
    )
    coll = Collection(name=COLLECTION_NAME, schema=schema)
    # Create index on vector field
    coll.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type":  "IVF_FLAT",
            "params":      {"nlist": 128},
        }
    )
    coll.load()
    print(f"✅ Created collection `{COLLECTION_NAME}`")

def get_embedding(text: str) -> List[float]:
    """Embed text using OpenAI’s ada-002."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model="text-embedding-ada-002", input=[text])
    return resp.data[0].embedding

# ─────────── Public API ───────────
def initialize_milvus():
    """Call on startup to ensure collection exists (or fallback)."""
    connect_to_milvus()
    if not use_fallback:
        create_collection_if_not_exists()

def ingest_milvus(text: str) -> None:
    """Insert a new (text, embedding) into Milvus or fallback store."""
    # 1. Embed the text
    emb = get_embedding(text)
    # Always generate an in-memory ID for fallback
    rec_id = str(uuid.uuid4())

    # 2. Fallback path
    if use_fallback:
        fallback_embeddings[rec_id] = emb
        fallback_texts[rec_id]      = text
        print(f"📝 Ingested to fallback: {text[:50]}…")
        return

    # 3. Real Milvus path
    connect_to_milvus()           # ensure connection
    create_collection_if_not_exists()
    coll = Collection(COLLECTION_NAME)

    # **Only** send the non-PK fields; Milvus will auto-generate the primary key
    coll.insert([
        {
          "text":      text,
          "embedding": emb
        }
    ])
    coll.flush()
    print(f"📝 Ingested to Milvus: {text[:50]}…")

def search_local(query: str, top_k: int = 3) -> str:
    """Brute-force cosine search over in-memory embeddings."""
    if not fallback_embeddings:
        return "No data yet. Please add some text first."
    q_emb = np.array(get_embedding(query))
    sims = []
    for rid, emb in fallback_embeddings.items():
        e = np.array(emb)
        cos = float(np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e)))
        sims.append((rid, cos))
    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:top_k]
    return "\n\n".join(fallback_texts[rid] for rid, _ in top)

def search_milvus(query: str, top_k: int = 3) -> str:
    """Retrieve top-k most similar texts from Milvus or fallback."""
    if use_fallback:
        return search_local(query, top_k)

    try:
        connect_to_milvus()
        coll = Collection(COLLECTION_NAME)
        coll.load()
        q_emb = get_embedding(query)
        results = coll.search(
            data=[q_emb],
            anns_field="embedding",
            param={"metric_type":"COSINE", "params":{"nprobe":10}},
            limit=top_k,
            output_fields=["text"]
        )
        texts = []
        for hits in results:
            for hit in hits:
                texts.append(hit.entity.get("text", ""))
        if not texts:
            return "No relevant info found. Please add more to the knowledge base."
        return "\n\n".join(texts)
    except Exception as e:
        print(f"⚠️ Milvus search failed: {e}")
        return search_local(query, top_k)