#!/usr/bin/env python3
import argparse
import tempfile
import shutil
import subprocess
import os
import re
from vector_service import ingest_milvus

def clone_repo(repo_url: str, dest: str):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", repo_url, dest], check=True)

def discover_files(root_dir: str, exts=(".py", ".md", ".txt")):
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(exts):
                yield os.path.join(dp, fn)

def chunk_by_file(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [f.read()]

def chunk_by_function(path: str):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    # Simple split: every top-level "def "
    parts = re.split(r"\n(?=def )", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_by_window(path: str, window_size=200, overlap=50):
    words = open(path, "r", encoding="utf-8", errors="ignore").read().split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + window_size]
        chunks.append(" ".join(chunk))
        i += window_size - overlap
    return chunks

def main():
    p = argparse.ArgumentParser(
        description="Bulk-load a GitHub repo into your sparse Milvus index"
    )
    p.add_argument("--repo",      required=True, help="GitHub repo URL")
    p.add_argument(
        "--strategy",
        choices=["file", "function", "window"],
        default="function",
        help="Chunking strategy",
    )
    p.add_argument(
        "--window-size", type=int, default=200,
        help="Window size (words) when using 'window' strategy",
    )
    p.add_argument(
        "--overlap", type=int, default=50,
        help="Overlap size (words) when using 'window' strategy",
    )
    args = p.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="load_data_")
    try:
        print(f"Cloning {args.repo} → {tmpdir}")
        clone_repo(args.repo, tmpdir)

        files = list(discover_files(tmpdir))
        print(f"Discovered {len(files)} files to index.")

        total = 0
        for path in files:
            if args.strategy == "file":
                chunks = chunk_by_file(path)
            elif args.strategy == "function":
                chunks = chunk_by_function(path)
            else:
                chunks = chunk_by_window(
                    path, args.window_size, args.overlap
                )

            for chunk in chunks:
                ingest_milvus(chunk)
                total += 1

        print(f"✅ Ingested {total} chunks with '{args.strategy}' strategy.")
    finally:
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()