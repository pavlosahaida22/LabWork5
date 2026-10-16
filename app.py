"""
Лабораторна робота № 5 – інтерактивна RAG-система (Streamlit).
Пошук по документах (FAISS) + мовна модель (Gemini / Claude / OpenAI) + відображення джерел.
"""
from pathlib import Path

import streamlit as st

from rag import config
from rag.embeddings import Embedder
from rag.indexing import build_index, load_or_build
from rag.llm import PROVIDER_LABELS, LLMClient, get_api_key
from rag.pipeline import SYSTEM_PROMPT, answer_question
from rag.retriever import MODES, Retriever

st.set_page_config(page_title="RAG-асистент по документах", page_icon="📚", layout="wide")

EXAMPLES = [
    "Який запас ходу має VoltRide X в режимі Eco?",
    "Скільки діє гарантія на батарею і як її зберігати взимку?",
    "Мотор перегрівся через перевантаження – що робити і чи буде це гарантійним випадком?",
    "Скільки коштує електровелосипед VoltRide X?",
]


# ----------------------------------------------------------- ресурси (кешуються)
@st.cache_resource(show_spinner="Завантаження моделі ембедингів…")
def get_embedder(backend: str) -> Embedder:
    return Embedder(backend)


@st.cache_resource(show_spinner="Завантаження / побудова векторного сховища…")
def get_store(backend: str, version: int):
    return load_or_build(get_embedder(backend))


def safe_secrets():
    try:
        return st.secrets
    except Exception:
        return None


# ------------------------------------------------------------------------ стан
st.session_state.setdefault("history", [])
st.session_state.setdefault("index_version", 0)
backend = config.EMBEDDING_BACKEND
embedder = get_embedder(backend)
store = get_store(backend, st.session_state["index_version"])

# --------------------------------------------------------------------- сайдбар
with st.sidebar:
    st.header("⚙️ Налаштування")

    st.subheader("🤖 Мовна модель")
    provider = st.selectbox("Провайдер", list(PROVIDER_LABELS), format_func=PROVIDER_LABELS.get)
    llm = None
    if provider != "none":
        model_name = st.text_input("Модель", value=config.DEFAULT_MODELS[provider],
                                   help="Назву можна змінити, якщо модель застаріла")
        api_key = get_api_key(provider, safe_secrets())
        typed = st.text_input("API-ключ", type="password",
                              placeholder="знайдено в секретах/середовищі" if api_key else "вставте ключ",
                              help="Ключ не зберігається. Краще задати його через змінну середовища або Streamlit Secrets.")
        api_key = typed or api_key
        if api_key:
            llm = LLMClient(provider, api_key, model_name)
        else:
            st.warning("Ключ не задано – працюватиме режим «лише пошук».")

    st.subheader("🔎 Пошук")
    mode = st.selectbox("Спосіб пошуку", list(MODES), format_func=MODES.get)
    k = st.slider("Кількість фрагментів (k)", 1, 10, config.DEFAULT_K)
    lambda_mult = st.slider("MMR: баланс релевантність/різноманітність", 0.0, 1.0, 0.6, disabled=mode != "mmr")
    sources = st.multiselect("Фільтр за документами (metadata filtering)", store.sources,
                             placeholder="усі документи")
    rewrite = st.checkbox("Переформулювання запиту (query rewriting)", disabled=llm is None,
                          help="LLM перетворює запитання на пошуковий запит")

    st.subheader("📚 База знань")
    m = store.meta
    st.caption(f"Документів: **{m['n_documents']}** · фрагментів: **{m['n_chunks']}**  \n"
               f"Розмір чанка: {m.get('chunk_size')} / перекриття: {m.get('chunk_overlap')}  \n"
               f"Ембединги: `{m['embedding_model']}`")
    with st.expander("Перебудувати індекс"):
        cs = st.slider("Розмір фрагмента, символів", 150, 1200, int(m.get("chunk_size", config.DEFAULT_CHUNK_SIZE)), 50)
        ov = st.slider("Перекриття, символів", 0, 300, int(m.get("chunk_overlap", config.DEFAULT_CHUNK_OVERLAP)), 10)
        uploads = st.file_uploader("Додати документи (PDF, TXT, MD)", type=["pdf", "txt", "md"],
                                   accept_multiple_files=True)
        if st.button("🔄 Перебудувати", width="stretch"):
            for f in uploads or []:
                (config.DOCS_DIR / f.name).write_bytes(f.getbuffer())
            with st.spinner("Побудова індексу…"):
                build_index(chunk_size=cs, chunk_overlap=min(ov, cs - 1), embedder=embedder)
            st.session_state["index_version"] += 1
            st.rerun()

    if st.button("🗑️ Очистити діалог", width="stretch"):
        st.session_state["history"] = []
        st.rerun()

# ------------------------------------------------------------------- основне вікно
st.title("📚 RAG-асистент по документах")
st.caption("Відповіді формуються лише на основі завантажених документів, із зазначенням джерел. "
           "Демо-база: документація вигаданого електровелосипеда VoltRide X.")


def render_sources(sources_list):
    if not sources_list:
        st.caption("Релевантних фрагментів не знайдено.")
        return
    with st.expander(f"📎 Джерела ({len(sources_list)})"):
        for s in sources_list:
            st.markdown(f"**[{s['n']}] {s['label']}** · score: `{s['score']:.3f}`")
            st.caption(s["text"])


for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("search_query") and msg["search_query"] != msg.get("question"):
                st.caption(f"🔁 Пошуковий запит після переформулювання: _{msg['search_query']}_")
            render_sources(msg.get("sources"))

if not st.session_state["history"]:
    st.markdown("**Спробуйте запитання:**")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, width="stretch"):
            st.session_state["pending"] = ex
            st.rerun()

question = st.chat_input("Поставте запитання до документів…") or st.session_state.pop("pending", None)

if question:
    st.session_state["history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retriever = Retriever(store, embedder, mode=mode, k=k, lambda_mult=lambda_mult, sources=sources or None)
    with st.chat_message("assistant"):
        with st.spinner("Пошук і формування відповіді…"):
            res = answer_question(question, retriever, llm, rewrite=rewrite)

        if llm is None:
            text = ("ℹ️ *Режим «лише пошук» (LLM не підключено).* Нижче – найбільш релевантні фрагменти.\n\n"
                    + ("\n\n".join(f"**[{s['n']}] {s['label']}**\n> {s['text']}" for s in res.sources)
                       or "Релевантних фрагментів не знайдено."))
        elif res.error:
            text = f"⚠️ Помилка виклику LLM: `{res.error}`"
        else:
            text = res.answer
        st.markdown(text)
        if res.search_query != question:
            st.caption(f"🔁 Пошуковий запит після переформулювання: _{res.search_query}_")
        if llm is not None:
            render_sources(res.sources)
            with st.expander("🧾 Промт, надісланий моделі"):
                st.text(SYSTEM_PROMPT)
                st.text(res.prompt)

    st.session_state["history"].append({
        "role": "assistant", "content": text, "question": question, "search_query": res.search_query,
        "sources": res.sources if llm is not None else []})
