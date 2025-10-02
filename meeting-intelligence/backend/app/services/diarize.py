import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

encoder = None

def ensure_encoder():
    global encoder
    if encoder is None:
        encoder = VoiceEncoder()
    return encoder

def embed_audio_segment(wav_path: str) -> np.ndarray:
    """
    Generate an embedding vector for a given audio segment using Resemblyzer.
    """
    enc = ensure_encoder()
    wav = preprocess_wav(wav_path)
    embedding = enc.embed_utterance(wav)
    return embedding

def diarize_segments(segment_paths: List[str], min_speakers: int = 1, max_speakers: int = 6) -> List[int]:
    """
    Given a list of paths to audio segments (short WAV files), compute embeddings and cluster them.
    Returns a list of cluster labels (speaker ids) corresponding to each input segment.
    """
    embeddings = []
    for path in segment_paths:
        try:
            emb = embed_audio_segment(path)
            embeddings.append(emb)
        except Exception as e:
            logger.exception(f"Error embedding {path}: {e}")
            embeddings.append(np.zeros((256,)))  # Fallback to zero vector

    X = np.vstack(embeddings)
    n_clusters = min(max_speakers, max(min_speakers, len(segment_paths)))
    if n_clusters < 1:
        return [0] * len(segment_paths)  # All segments assigned to speaker 0
    
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    labels = clustering.labels_.tolist()
    return labels