"""5.2.6–5.2.7 Векторне сховище (FAISS) зі збереженням у локальній директорії."""
import json
import time
from pathlib import Path

import faiss
import numpy as np

from .chunking import embedding_text


class VectorStore:
    def __init__(self, index, chunks: list[dict], meta: dict):
        self.index, self.chunks, self.meta = index, chunks, meta
        self.vectors = index.reconstruct_n(0, index.ntotal) if index.ntotal else np.zeros((0, index.d), "float32")
        self._bm25 = None

    # ----------------------------------------------------------- створення / збереження
    @classmethod
    def build(cls, chunks, embedder, params: dict | None = None):
        vecs = embedder.embed_documents([embedding_text(c) for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])      # косинусна подібність (вектори нормалізовані)
        index.add(vecs)
        meta = {"embedding_backend": embedder.backend, "embedding_model": embedder.label,
                "n_chunks": len(chunks),
                "n_documents": len({c["metadata"]["source"] for c in chunks}),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), **(params or {})}
        return cls(index, chunks, meta)

    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        (d / "index.faiss").write_bytes(faiss.serialize_index(self.index).tobytes())
        (d / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        index = faiss.deserialize_index(np.frombuffer((d / "index.faiss").read_bytes(), dtype="uint8"))
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return cls(index, chunks, meta)

    @staticmethod
    def exists(directory) -> bool:
        d = Path(directory)
        return all((d / f).exists() for f in ("index.faiss", "chunks.json", "meta.json"))

    # -------------------------------------------------------------------------- пошук
    @property
    def sources(self) -> list[str]:
        return sorted({c["metadata"]["source"] for c in self.chunks})

    def allowed_mask(self, sources):
        if not sources:
            return np.ones(len(self.chunks), dtype=bool)
        sset = set(sources)
        return np.array([c["metadata"]["source"] in sset for c in self.chunks])

    def vector_scores(self, qvec: np.ndarray) -> np.ndarray:
        return self.vectors @ qvec
