# test_rag.py
import os
import pytest
from fastapi.testclient import TestClient
os.environ["TESTING"]="true"    # force in-memory
from main import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code==200
    assert r.json()["status"]=="healthy"

@pytest.mark.parametrize("text,question", [
    ("Alan is a data scientist.",        "Who is Alan?"),
    ("Python is a programming language.","What is Python?"),
    ("FastAPI is a Python web framework.","What is FastAPI?"),
    ("Milvus is a vector database.",      "What is Milvus?"),
    ("OpenAI created GPT-4o.",            "Who created GPT-4o?")
])
def test_ingest_and_ask_returns_known_fact(text,question):
    r1=client.post("/ingest",json={"text":text})
    assert r1.status_code==204
    r2=client.post("/ask",json={"question":question})
    assert r2.status_code==200
    ans=r2.json()["answer"].lower()
    assert text.lower() in ans
    sources=r2.json()["sources"]
    assert len(sources)>0
    assert sources[0]["text"].lower()==text.lower()

def test_ask_before_any_ingest():
    r=client.post("/ask",json={"question":"Anything?"})
    assert r.status_code==200
    ans=r.json()["answer"].lower()
    assert "don't know" in ans or "no relevant information" in ans

def test_empty_question_field():
    r=client.post("/ask",json={"question":""})
    assert r.status_code in (200,400,422)

def test_malformed_json_body():
    r=client.post("/ask",data="not-a-json",headers={"Content-Type":"application/json"})
    assert r.status_code==422

def test_non_json_content_type():
    r=client.post("/ask",data='{"question":"x"}',headers={"Content-Type":"text/plain"})
    assert r.status_code in (415,422)

def test_sql_injection_style_question():
    client.post("/ingest",json={"text":"Safe fact about X."})
    r=client.post("/ask",json={"question":"DROP TABLE users; --"})
    assert r.status_code==200
    ans=r.json()["answer"].lower()
    assert "don't know" in ans or "no relevant information" in ans