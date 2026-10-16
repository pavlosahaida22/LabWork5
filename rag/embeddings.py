"""5.2.6 Векторні представлення текстів."""
import hashlib
import re
import numpy as np
from . import config


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.clip(n, 1e-12, None)).astype("float32")


class Embedder:
    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = backend or config.EMBEDDING_BACKEND
        self.model_name = model or config.EMBEDDING_MODEL
        self._model = None
        if self.backend == "fastembed":
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
            self.dim = 384
        elif self.backend == "hash":
            self.dim = 384
        else:
            raise ValueError(f"Невідомий backend: {self.backend}")

    @property
    def label(self) -> str:
        return self.model_name if self.backend == "fastembed" else "hash-demo"

    def _hash(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok[:5].encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 40) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self.backend == "hash":
            return _normalize(np.vstack([self._hash(t) for t in texts]))
        return _normalize(np.array(list(self._model.embed(texts)), dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
