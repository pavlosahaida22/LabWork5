"""5.2.4–5.2.6 Побудова векторного сховища з документів і збереження його локально.

Приклад:  python build_index.py --chunk-size 400 --chunk-overlap 60
"""
import argparse

from rag import config
from rag.embeddings import Embedder
from rag.indexing import build_index
from rag.loader import load_documents

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs-dir", default=str(config.DOCS_DIR))
    ap.add_argument("--index-dir", default=str(config.INDEX_DIR))
    ap.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=config.DEFAULT_CHUNK_OVERLAP)
    ap.add_argument("--backend", default=config.EMBEDDING_BACKEND, choices=["fastembed", "hash"])
    a = ap.parse_args()

    docs = load_documents(a.docs_dir)
    print(f"Завантажено записів-документів: {len(docs)}")
    for d in docs:
        m = d["metadata"]
        print(f"  • {m['source']:34s} | {m['title']} | {len(d['text'])} символів" + (f" | стор. {m['page']}" if m['page'] else ""))

    store = build_index(a.docs_dir, a.index_dir, a.chunk_size, a.chunk_overlap, Embedder(a.backend))
    print(f"\nСтворено фрагментів: {store.meta['n_chunks']} (розмір {a.chunk_size}, перекриття {a.chunk_overlap})")
    print(f"Модель ембедингів: {store.meta['embedding_model']}")
    print(f"Сховище збережено у: {a.index_dir}")
    print("\nПриклад фрагмента:", store.chunks[0]["metadata"])
    print(store.chunks[0]["text"][:200], "…")
