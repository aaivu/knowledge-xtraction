from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EmbedderConfig:
    model_name: str = "paraphrase-MPNet-base-v2"
    batch_size: int = 64
    device: str | None = None  # "cpu" or "cuda", None = auto


class TextEmbedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype("float32")

    @staticmethod
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        # a,b are 1D normalized vectors
        return float(np.dot(a, b))

    @staticmethod
    def cos_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return cosine_similarity(A, B)


def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS inner-product index for normalized embeddings.
    Returns index or None if FAISS not available.
    """
    if not _HAS_FAISS:
        return None

    import faiss  # type: ignore

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index
