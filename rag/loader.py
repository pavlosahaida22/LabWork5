"""5.2.4 Завантаження документів із директорії з метаданими про джерело."""
import re
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def _title(text: str, path: Path) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    return m.group(1).strip() if m else path.stem.replace("_", " ")


def load_documents(docs_dir) -> list[dict]:
    """Повертає список {"text": ..., "metadata": {...}}. Для PDF – по одному запису на сторінку."""
    docs = []
    for path in sorted(Path(docs_dir).rglob("*")):
        ext = path.suffix.lower()
        if ext not in SUPPORTED or not path.is_file():
            continue
        if ext in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append({"text": text, "metadata": {
                    "source": path.name, "title": _title(text, path), "page": None, "type": ext[1:]}})
        else:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"text": text, "metadata": {
                        "source": path.name, "title": path.stem.replace("_", " "), "page": i, "type": "pdf"}})
    return docs
