"""5.2.7, 5.2.11–5.2.12 Оцінка пошуку та експерименти з параметрами RAG.

  python evaluate.py                                   # порівняння режимів пошуку та k
  python evaluate.py --chunk-sizes 200 400 800         # + вплив розміру чанка
  python evaluate.py --generate --provider gemini      # + відповіді LLM (потрібен API-ключ у змінній середовища)
Метрики пошуку (за очікуваними джерелами у eval/test_queries.json):
  hit_rate    – частка запитів, де серед результатів є хоча б одне очікуване джерело;
  full_recall – частка запитів, де знайдено ВСІ очікувані джерела (важливо для multi-запитів).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from rag import config
from rag.chunking import split_documents
from rag.embeddings import Embedder
from rag.llm import LLMClient, get_api_key
from rag.loader import load_documents
from rag.pipeline import answer_question
from rag.retriever import Retriever
from rag.store import VectorStore


def evaluate_retrieval(store, embedder, queries, mode, k):
    r = Retriever(store, embedder, mode=mode, k=k)
    rows = []
    for q in queries:
        hits = r.retrieve(q["question"])
        found = {h.meta["source"] for h in hits}
        exp = set(q["expected_sources"])
        rows.append({"type": q["type"],
                     "hit": bool(found & exp) if exp else None,
                     "full": exp <= found if exp else None,
                     "top_score": hits[0].score if hits else 0.0})
    df = pd.DataFrame(rows)
    ans = df[df["type"] != "absent"]
    return {"hit_rate": ans["hit"].mean(), "full_recall": ans["full"].mean(),
            "multi_full_recall": df[df["type"] == "multi"]["full"].mean(),
            "top_score_absent": df[df["type"] == "absent"]["top_score"].mean()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default="eval/test_queries.json")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[config.DEFAULT_CHUNK_SIZE])
    ap.add_argument("--overlap-ratio", type=float, default=0.15)
    ap.add_argument("--modes", nargs="+", default=["similarity", "mmr", "bm25", "hybrid"])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--backend", default=config.EMBEDDING_BACKEND)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    queries = json.loads(Path(a.queries).read_text(encoding="utf-8"))
    embedder = Embedder(a.backend)
    docs = load_documents(config.DOCS_DIR)

    results, stores = [], {}
    for cs in a.chunk_sizes:
        chunks = split_documents(docs, cs, int(cs * a.overlap_ratio))
        store = stores[cs] = VectorStore.build(chunks, embedder)
        for mode in a.modes:
            for k in a.ks:
                results.append({"chunk_size": cs, "n_chunks": len(chunks), "mode": mode, "k": k,
                                **evaluate_retrieval(store, embedder, queries, mode, k)})
    df = pd.DataFrame(results).round(3)
    print(df.to_string(index=False))
    Path("eval").mkdir(exist_ok=True)
    df.to_csv("eval/results.csv", index=False)
    print("\nРезультати збережено в eval/results.csv")

    if a.generate:
        key = get_api_key(a.provider)
        if not key:
            raise SystemExit("Не знайдено API-ключ у змінних середовища.")
        llm = LLMClient(a.provider, key, a.model)
        best = df.sort_values(["hit_rate", "full_recall"], ascending=False).iloc[0]
        print(f"\nГенерація відповідей з конфігурацією: chunk={int(best.chunk_size)}, mode={best['mode']}, k={int(best.k)}")
        retr = Retriever(stores[int(best.chunk_size)], embedder, mode=best["mode"], k=int(best.k))
        for q in queries:
            res = answer_question(q["question"], retr, llm)
            print(f"\n[{q['id']}] ({q['type']}) {q['question']}\n  Еталон: {q['reference']}\n  Відповідь: {res.answer or res.error}\n"
                  f"  Джерела: {sorted({s['source'] for s in res.sources})}")
