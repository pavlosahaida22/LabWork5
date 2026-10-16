"""5.2.5 Розбиття документів на фрагменти зі збереженням метаданих."""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)


def _split_sections(text: str, default: str):
    """Ділить Markdown за заголовками: [(назва_розділу, текст)]."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(default, text)]
    sections = []
    pre = text[:matches[0].start()].strip()
    if pre:
        sections.append((default, pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            sections.append((m.group(2).strip(), body))
    return sections


def split_documents(docs: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""])
    chunks, per_source = [], {}
    for d in docs:
        meta = d["metadata"]
        sections = (_split_sections(d["text"], meta["title"]) if meta["type"] == "md"
                    else [(meta["title"], d["text"])])
        for section, body in sections:
            for piece in splitter.split_text(body):
                piece = piece.strip()
                if not piece:
                    continue
                idx = per_source.get(meta["source"], 0)
                per_source[meta["source"]] = idx + 1
                chunks.append({"text": piece,
                               "metadata": {**meta, "section": section,
                                            "chunk_id": len(chunks), "chunk_index": idx}})
    return chunks


def embedding_text(chunk: dict) -> str:
    """Текст для векторизації: назва розділу додається до фрагмента (покращує пошук)."""
    section = chunk["metadata"].get("section")
    return f"{section}. {chunk['text']}" if section else chunk["text"]
