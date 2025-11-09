# ==============================================
# 💊 MedRAG v3.1 – True RAG for Medical PDFs
# ==============================================
# Features:
# ✅ PyMuPDF text extraction (better than pdfplumber)
# ✅ Semantic chunking + embeddings retrieval
# ✅ Answers ONLY from document, quoted
# ✅ "Not found in document" fallback
# ==============================================

import streamlit as st
import fitz  # PyMuPDF
import tiktoken
import re
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------
# Streamlit Page Setup
# ----------------------------------------------
st.set_page_config(page_title="MedRAG", page_icon="💊", layout="centered")
st.title("💊 MedRAG – Medical PDF Q&A")
st.caption("Upload a medical PDF → Ask → Get answers **only from the document.**")

# ----------------------------------------------
# Check API key
# ----------------------------------------------
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("⚠️ Add `OPENROUTER_API_KEY` in `.streamlit/secrets.toml`")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
MODEL = "gpt-4o-mini"
ENCODER = tiktoken.encoding_for_model(MODEL)

# ----------------------------------------------
# Text Extraction (PyMuPDF)
# ----------------------------------------------
def extract_text(pdf_file):
    text = ""
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
        clean_text = re.sub(r"\s+", " ", text.strip())
        return clean_text
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""

# ----------------------------------------------
# Split text into semantic chunks
# ----------------------------------------------
def split_text(text, max_tokens=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current, token_count = [], "", 0

    for sentence in sentences:
        tokens = len(ENCODER.encode(sentence))
        if token_count + tokens > max_tokens:
            chunks.append(current.strip())
            current, token_count = sentence, tokens
        else:
            current += " " + sentence
            token_count += tokens
    if current.strip():
        chunks.append(current.strip())

    return chunks

# ----------------------------------------------
# Create embeddings
# ----------------------------------------------
def get_embedding(text):
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(emb.data[0].embedding)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return np.zeros(1536)

# ----------------------------------------------
# Retrieve relevant chunks
# ----------------------------------------------
def retrieve_chunks(question, chunks, embeddings, top_k=4):
    # Add contextually rich expansion for methods/framework queries
    synonyms = (
        "stage step phase framework process methodology method approach design "
        "Arksey O'Malley Levac scoping review structure workflow outline sequence"
    )
    question_aug = question + " " + synonyms
    q_emb = get_embedding(question_aug)
    sims = cosine_similarity([q_emb], embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return "\n\n---\n\n".join([chunks[i] for i in top_idx])



# ----------------------------------------------
# Ask GPT only from retrieved chunks
# ----------------------------------------------
def ask_from_doc(question, chunks, embeddings):
    context = retrieve_chunks(question, chunks, embeddings)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical research assistant. "
                        "Answer ONLY from the DOCUMENT provided below. "
                        "Always quote the exact text in double quotes. "
                        "If not found, respond exactly: 'Not found in document.'"
                    ),
                },
                {
                    "role": "user",
                    "content": f"DOCUMENT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
                },
            ],
            temperature=0,
            max_tokens=400,
        )
        answer = response.choices[0].message.content.strip()
        if "not found" not in answer.lower() and '"' not in answer:
            return "Not found in document."
        return answer
    except Exception as e:
        return f"⚠️ API Error: {e}"

# ----------------------------------------------
# Streamlit State Init
# ----------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# ----------------------------------------------
# Upload PDF
# ----------------------------------------------
uploaded = st.file_uploader("📄 Upload a medical PDF", type="pdf")

if uploaded:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text(uploaded)
        if text:
            st.session_state.pdf_text = text
            st.session_state.chunks = split_text(text)
            st.session_state.embeddings = np.array([get_embedding(c) for c in st.session_state.chunks])
            st.success(f"✅ Document processed into {len(st.session_state.chunks)} chunks.")
        else:
            st.error("❌ Failed to extract text.")
            st.stop()
else:
    st.info("⬆️ Upload a medical PDF to begin.")
    st.stop()

# ----------------------------------------------
# Ask a question
# ----------------------------------------------
question = st.text_input(
    "💬 Ask about the document:",
    placeholder="e.g., What is the title of the paper?",
)

if st.button("🔎 Search Document") and question.strip():
    with st.spinner("Analyzing document..."):
        answer = ask_from_doc(question, st.session_state.chunks, st.session_state.embeddings)
    st.markdown(f"**Answer:** {answer}")

# ----------------------------------------------
# Debug Search Tool
# ----------------------------------------------
st.markdown("---")
st.subheader("🔬 Quick Word Search (Debug)")

search = st.text_input("Find a term in the document:", key="debug")
if search and st.session_state.pdf_text:
    txt = st.session_state.pdf_text.lower()
    term = search.lower()
    count = txt.count(term)
    if count:
        st.success(f"✅ Found '{search}' {count} time(s).")
        idx = txt.find(term)
        snippet = st.session_state.pdf_text[max(0, idx - 100): idx + 200]
        st.code(snippet.replace(search, f"**{search}**"), language="text")
    else:
        st.warning(f"'{search}' not found.")

# ----------------------------------------------
# Download Extracted Text
# ----------------------------------------------
if st.session_state.pdf_text:
    st.download_button(
        "⬇️ Download Extracted Text",
        data=st.session_state.pdf_text.encode(),
        file_name="extracted_medtext.txt",
        mime="text/plain",
    )

st.caption("🧬 MedRAG v3.1 – Verified Retrieval-Augmented Q&A for Medical PDFs.")
