import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

IN_FILE = r".\out\wikipedia_dataset.jsonl"
INDEX_FILE = r".\out\index.faiss"
META_FILE = r".\out\meta.jsonl"


def main() -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = []

    with open(IN_FILE, "r", encoding="utf-8") as src, \
         open(META_FILE, "w", encoding="utf-8") as meta:

        for i, line in enumerate(tqdm(src, desc="Encoding")):
            obj = json.loads(line)
            doc = f'{obj["title"]}\n\n{obj["text"]}'
            vec = model.encode(doc, normalize_embeddings=True).astype("float32")

            vectors.append(vec)
            meta.write(json.dumps({"id": i, "title": obj["title"]},ensure_ascii=False) + "\n")

    X = np.vstack(vectors)
    index = faiss.IndexFlatIP(X.shape[1]) 
    index.add(X)

    faiss.write_index(index, INDEX_FILE)
    print(f"Indexed {index.ntotal} docs -> {INDEX_FILE}\nMeta -> {META_FILE}")
if __name__ == "__main__":
    main()