import os
import tempfile
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A | Prashik Sawant",
    page_icon="📚",
    layout="wide"
)

# ── API Key ────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Add it in HuggingFace Space Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ── Embedding Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ── Session State ──────────────────────────────────────────
if "chunks"     not in st.session_state: st.session_state.chunks     = []
if "embeddings" not in st.session_state: st.session_state.embeddings = []
if "sources"    not in st.session_state: st.session_state.sources    = []
if "messages"   not in st.session_state: st.session_state.messages   = []
if "uploaded"   not in st.session_state: st.session_state.uploaded   = set()

# ── Helpers ────────────────────────────────────────────────
def extract_text_from_bytes(file_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    reader = PdfReader(path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    os.unlink(path)
    return text

def chunk_text(text, size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap) if words[i:i+size]]

def add_document(text, filename):
    chunks = chunk_text(text)
    embeds = model.encode(chunks, show_progress_bar=False)
    st.session_state.chunks.extend(chunks)
    st.session_state.embeddings.extend(embeds)
    st.session_state.sources.extend([filename] * len(chunks))
    return len(chunks)

def search(query, n=3):
    if not st.session_state.chunks:
        return [], []
    q_emb = model.encode([query])[0]
    all_emb = np.array(st.session_state.embeddings)
    scores = np.dot(all_emb, q_emb) / (
        np.linalg.norm(all_emb, axis=1) * np.linalg.norm(q_emb) + 1e-10
    )
    idx = np.argsort(scores)[::-1][:n]
    return [st.session_state.chunks[i] for i in idx], [st.session_state.sources[i] for i in idx]

def ask(question, history):
    docs, srcs = search(question)
    context = "\n\n".join(f"[{s}]: {d}" for d, s in zip(docs, srcs)) if docs else "No documents uploaded."
    source_text = ", ".join(set(srcs)) if srcs else ""

    messages = [{
        "role": "system",
        "content": f"""You are a helpful AI assistant answering questions from uploaded documents.

CONTEXT:
{context}

Rules:
- Answer ONLY from the context
- If not found, say "I couldn't find that in the uploaded documents"
- Mention which document your answer came from
- Be concise"""
    }]
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": question})

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
        temperature=0.3
    )
    return res.choices[0].message.content, source_text

# ── UI ─────────────────────────────────────────────────────
st.title("📚 RAG Document Q&A")
st.caption("Upload PDFs → Ask questions → Get cited answers | Built by Prashik Sawant")

with st.sidebar:
    st.header("📂 Documents")

    # Load from URL
    DEFAULT_PDFS = {
        "Attention Is All You Need": "https://arxiv.org/pdf/1706.03762",
    }

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        for name, url in DEFAULT_PDFS.items():
            with st.spinner(f"Loading {name}..."):
                try:
                    import urllib.request
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        urllib.request.urlretrieve(url, tmp.name)
                        with open(tmp.name, "rb") as f:
                            text = extract_text_from_bytes(f.read(), name)
                        os.unlink(tmp.name)
                    n = add_document(text, name)
                    st.session_state.uploaded.add(name)
                    st.success(f"✅ {name} loaded!")
                except Exception as e:
                    st.error(f"❌ Failed to load {name}: {e}")
        st.session_state.initialized = True

    st.divider()
    st.metric("Chunks in memory", len(st.session_state.chunks))

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Built by [Prashik Sawant](https://www.linkedin.com/in/prashik-sawant-ds/)**")
    st.markdown("Day 16 of 30 — AI Engineering Bootcamp")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, sources = ask(prompt, st.session_state.messages[:-1])
        st.markdown(answer)
        if sources:
            st.caption(f"📎 Sources: {sources}")
    st.session_state.messages.append({"role": "assistant", "content": answer})