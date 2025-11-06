import json
import os
import argparse
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE = r".\out"
DATA_FILE = os.path.join(BASE, "wikipedia_dataset.jsonl")
META_FILE = os.path.join(BASE, "meta.jsonl")
INDEX_FILE = os.path.join(BASE, "index.faiss")
NPY_OFFSETS = os.path.join(BASE, "wikipedia_dataset.offsets.npy")

def load_titles(path: str) -> List[str]:
    with open(path,"r", encoding = "utf-8") as f:
        return [json.loads(line)["title"] for line in f]
    

def clip(text: str, n:int) -> str:
    if n<=0:
        return text
    
    txt = text[:n].replace("\n", " ")
    return txt + ("..." if len(text) > n else "")

def read_line_at(jsonl_path: str, btye_pos: int) -> dict:
    with open(jsonl_path, "r", encoding = "utf-8") as f:
        f.seek(byte_pos)
        return json.loads(f.readline())


def build_load_offsets(jsonl_path: str, npy_path: str) -> np.ndarray:
    if os.path.exists(npy_path):
        return np.load(npy_path)
    
    offsets =  []
    pos=0

    with open(jsonl_path, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    

    arr = np.asarray(offsets, dtype=np.int64)
    np.save(npy_path, arr)
    return arr

def Main():
    