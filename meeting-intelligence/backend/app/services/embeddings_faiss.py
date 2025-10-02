import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  
INDEX_DIR = os.path.join(os.getcwd(), "data", "faiss_indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()

def create_faiss_index(meeting_id: int, embedding_dim: int, vectors: np.ndarray, mapping: List[int]) -> Tuple[str, str]: 
    """
    Create a FAISS index file for a meeting and save mapping (faiss_id -> segment_id)
    vectors: np.ndarray shape (N, dim)
    mapping: list of segment_id length N
    """
    index = faiss.IndexFlatIP(embedding_dim)  # Using Inner Product for cosine similarity

    # normalize vectors to unit length for cosine similarity
    faiss.normalize_L2(vectors)
    index.add(vectors)
    index_path = os.path.join(INDEX_DIR, f"meeting_{meeting_id}.index")
    mapping_path = os.path.join(INDEX_DIR, f"meeting_{meeting_id}_map.json")
    faiss.write_index(index, index_path)

    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    return index_path, mapping_path

def load_faiss_index(meeting_id: int) -> Tuple[faiss.Index, List[int]]:
    index_path = os.path.join(INDEX_DIR, f"meeting_{meeting_id}.index")
    mapping_path = os.path.join(INDEX_DIR, f"meeting_{meeting_id}_map.json")
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return index, mapping

def query_index(meeting_id: int, query_vector: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Query the FAISS index for a meeting with a text query.
    Returns list of tuples (segment_id, score)
    """
    index, mapping = load_faiss_index(meeting_id)
    if index is None:
        return []

    query_vec = np.array(query_vector).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        segment_id = mapping[idx]
        results.append({"segment_id": segment_id, "score": float(score)})
    return results
    