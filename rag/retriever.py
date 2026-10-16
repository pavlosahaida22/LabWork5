"""5.2.7 Пошуковий компонент: similarity, MMR, BM25, hybrid (RRF) + фільтрація за метаданими."""
import re
from dataclasses import dataclass

import numpy as np

MODES = {
    "similarity": "Векторний (cosine)",
    "mmr": "MMR (різноманітність)",
    "bm25": "BM25 (ключові слова)",
    "hybrid": "Гібридний (вектори + BM25, RRF)",
}


@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int

    @property
    def meta(self):
        return self.chunk["metadata"]


def _tokens(text: str) -> list[str]:
    # спрощений «стемінг» для української: перші 5 літер слова
    return [t[:5] for t in re.findall(r"\w+", text.lower())]


class Retriever:
    def __init__(self, store, embedder, mode="similarity", k=4, fetch_k=20,
                 lambda_mult=0.6, sources=None):
        self.store, self.embedder = store, embedder
        self.mode, self.k, self.fetch_k = mode, k, fetch_k
        self.lambda_mult, self.sources = lambda_mult, sources

    # --- допоміжні
    def _bm25_scores(self, query: str) -> np.ndarray:
        if self.store._bm25 is None:
            from rank_bm25 import BM25Okapi
            self.store._bm25 = BM25Okapi([_tokens(c["text"] + " " + c["metadata"].get("section", ""))
                                          for c in self.store.chunks])
        return np.array(self.store._bm25.get_scores(_tokens(query)), dtype="float32")

    @staticmethod
    def _rank(scores: np.ndarray, mask: np.ndarray) -> list[int]:
        idx = np.where(mask)[0]
        return [int(i) for i in idx[np.argsort(-scores[idx])]]

    # --- основний метод
    def retrieve(self, query: str) -> list[Hit]:
        st, k = self.store, self.k
        mask = st.allowed_mask(self.sources)
        if not mask.any():
            return []

        if self.mode == "bm25":
            s = self._bm25_scores(query)
            order = [i for i in self._rank(s, mask)[:k] if s[i] > 0]
            return [Hit(st.chunks[i], float(s[i]), r + 1) for r, i in enumerate(order)]

        q = self.embedder.embed_query(query)
        sims = st.vector_scores(q)

        if self.mode == "similarity":
            order = self._rank(sims, mask)[:k]
            return [Hit(st.chunks[i], float(sims[i]), r + 1) for r, i in enumerate(order)]

        if self.mode == "mmr":
            cand = self._rank(sims, mask)[:max(self.fetch_k, k)]
            chosen = []
            while cand and len(chosen) < k:
                best, best_val = None, -1e9
                for i in cand:
                    div = max((float(st.vectors[i] @ st.vectors[j]) for j in chosen), default=0.0)
                    val = self.lambda_mult * float(sims[i]) - (1 - self.lambda_mult) * div
                    if val > best_val:
                        best, best_val = i, val
                chosen.append(best); cand.remove(best)
            return [Hit(st.chunks[i], float(sims[i]), r + 1) for r, i in enumerate(chosen)]

        if self.mode == "hybrid":
            bm = self._bm25_scores(query)
            rv = {i: r for r, i in enumerate(self._rank(sims, mask)[:self.fetch_k], start=1)}
            rb = {i: r for r, i in enumerate(self._rank(bm, mask)[:self.fetch_k], start=1) if bm[i] > 0}
            fused = {}
            for i in set(rv) | set(rb):
                fused[i] = (1 / (60 + rv[i]) if i in rv else 0) + (1 / (60 + rb[i]) if i in rb else 0)
            order = sorted(fused, key=fused.get, reverse=True)[:k]
            return [Hit(st.chunks[i], float(fused[i]), r + 1) for r, i in enumerate(order)]

        raise ValueError(f"Невідомий режим пошуку: {self.mode}")
