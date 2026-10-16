"""Швидка перевірка без мережі та API-ключів (використовує hash-ембединги).
Запуск:  EMBEDDING_BACKEND=hash python tests/smoke_test.py
"""
import os, sys
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from rag.embeddings import Embedder
from rag.indexing import build_index
from rag.store import VectorStore
from rag.retriever import Retriever, MODES
from rag.pipeline import answer_question, NO_INFO

emb = Embedder("hash")
with tempfile.TemporaryDirectory() as tmp:
    store = build_index(index_dir=tmp, embedder=emb)
    store = VectorStore.load(tmp)                      # перевірка збереження/завантаження
assert store.meta["n_chunks"] > 10
assert all("source" in c["metadata"] and "section" in c["metadata"] for c in store.chunks)

for mode in MODES:
    hits = Retriever(store, emb, mode=mode, k=3).retrieve("Який запас ходу в режимі Eco?")
    assert hits and len(hits) <= 3, mode
filtered = Retriever(store, emb, k=3, sources=["03_garantiya.md"]).retrieve("запас ходу")
assert {h.meta["source"] for h in filtered} == {"03_garantiya.md"}

class StubLLM:
    def generate(self, system, user, max_tokens=0):
        assert "Контекст:" in user and "ВИКЛЮЧНО" in system
        return "Тестова відповідь [1]"
res = answer_question("Скільки гарантії на раму?", Retriever(store, emb, k=3), StubLLM())
assert res.answer.startswith("Тестова") and res.sources
print("OK: усі перевірки пройдено; чанків:", store.meta["n_chunks"])
