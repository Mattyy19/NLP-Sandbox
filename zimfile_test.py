
#very simple test for zim file encoding results 
import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

INDEX = r".\out\index.faiss"
META  = r".\out\meta.jsonl"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(INDEX)
meta = [json.loads(x) for x in open(META, encoding="utf-8")]

while True:
    q = input("Query (blank to exit): ").strip()
    if not q: break
    qv = model.encode(q, normalize_embeddings=True).astype("float32")[None, :]
    D, I = index.search(qv, 5)
    for r, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        print(f"{r}. {meta[idx]['title']}  score={score:.3f}")
