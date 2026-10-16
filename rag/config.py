import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "vectorstore"

# Багатомовна (укр. підтримується) легка модель ембедингів, ONNX через fastembed (~220 МБ, 384 вимірів).
# Увага: модель обробляє до ~128 токенів, тому чанки не варто робити надто великими.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# "fastembed" – реальна модель; "hash" – лише для офлайн-тестів без завантаження моделі
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fastembed")

DEFAULT_CHUNK_SIZE = 400      # символів
DEFAULT_CHUNK_OVERLAP = 60    # символів
DEFAULT_K = 4

DEFAULT_MODELS = {            # назви моделей можна змінити в інтерфейсі
    "gemini": "gemini-3.1-flash-lite",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-5-mini",
}
