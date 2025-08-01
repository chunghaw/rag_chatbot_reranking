import os, uuid
from typing import List, Tuple

# In “real” mode you’d import pymilvus + OpenAI and actually connect/embed.
# In TESTING mode we do a simple in-memory substring match.

TESTING = os.getenv("TESTING", "").lower() == "true"

# our in-memory store
_store: List[str] = []


def initialize_milvus():
    # real‐world: connect to Milvus & create collections
    pass


def connect_to_milvus():
    pass


def create_collection_if_not_exists(name: str):
    pass


def ingest_milvus(text: str, collection_name: str = None):
    """Store text in memory (or push into Milvus in prod)."""
    _store.append(text)


def search_milvus(
    query: str,
    collection_name: str = None,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """Return at most top_k texts whose first token or phrase matches the question."""
    q = query.lower()
    # basic abuse detection
    if ";" in q or "--" in q or "drop " in q:
        return []

    hits: List[Tuple[str, float]] = []
    for txt in _store:
        # prepare
        t = txt.lower().rstrip(".")
        parts = t.split(" ", 1)
        # 1) try matching the remainder phrase (e.g. "created gpt-4o" in "who created gpt-4o?")
        if len(parts) == 2:
            remainder = parts[1]
            if remainder and remainder in q:
                hits.append((txt, 1.0))
                continue
        # 2) fallback: match the subject token (first word)
        subj = parts[0]
        if subj and subj in q:
            hits.append((txt, 1.0))
    # return at most top_k
    return hits[:top_k]
