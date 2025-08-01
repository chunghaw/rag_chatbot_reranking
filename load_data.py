#!/usr/bin/env python3
# load_data.py
import os, sys, argparse, tempfile
from git import Repo
from vector_service import ingest_milvus

def chunk_by_file(repo_dir, ingest_fn):
    for root,_,files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root,f)
                with open(path) as r:
                    text = r.read()
                ingest_fn(text)

def chunk_by_function(repo_dir, ingest_fn):
    import re
    func_re = re.compile(r"^(def .+?:)", re.MULTILINE)
    for root,_,files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root,f)
                text = open(path).read()
                parts = func_re.split(text)
                # parts = [pre, 'def foo...', body, 'def bar...', body...]
                for i in range(1,len(parts),2):
                    chunk = parts[i] + parts[i+1]
                    ingest_fn(chunk)

def chunk_sliding_window(repo_dir, ingest_fn, size=200, overlap=50):
    for root,_,files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".py"):
                text = open(os.path.join(root,f)).read().split()
                i=0
                while i < len(text):
                    chunk = " ".join(text[i:i+size])
                    ingest_fn(chunk)
                    i += size - overlap

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--strategy", choices=["file","function","window"], default="file")
    p.add_argument("--window-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=50)
    args = p.parse_args()

    tmp = tempfile.mkdtemp()
    Repo.clone_from(args.repo, tmp)

    if args.strategy=="file":
        chunk_by_file(tmp, ingest_milvus)
    elif args.strategy=="function":
        chunk_by_function(tmp, ingest_milvus)
    else:
        chunk_sliding_window(tmp, ingest_milvus, args.window_size, args.overlap)

if __name__=="__main__":
    main()