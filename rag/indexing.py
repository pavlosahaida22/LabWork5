from . import config
from .chunking import split_documents
from .embeddings import Embedder
from .loader import load_documents
from .store import VectorStore


def build_index(docs_dir=config.DOCS_DIR, index_dir=config.INDEX_DIR,
                chunk_size=config.DEFAULT_CHUNK_SIZE, chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
                embedder: Embedder | None = None, save=True) -> VectorStore:
    embedder = embedder or Embedder()
    docs = load_documents(docs_dir)
    if not docs:
        raise FileNotFoundError(f"У {docs_dir} немає документів (.txt, .md, .pdf)")
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    store = VectorStore.build(chunks, embedder, {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
    if save:
        store.save(index_dir)
    return store


def load_or_build(embedder: Embedder, docs_dir=config.DOCS_DIR, index_dir=config.INDEX_DIR,
                  chunk_size=config.DEFAULT_CHUNK_SIZE, chunk_overlap=config.DEFAULT_CHUNK_OVERLAP) -> VectorStore:
    if VectorStore.exists(index_dir):
        store = VectorStore.load(index_dir)
        if (store.meta.get("embedding_backend") == embedder.backend
                and store.meta.get("embedding_model") == embedder.label):
            return store
    return build_index(docs_dir, index_dir, chunk_size, chunk_overlap, embedder)
