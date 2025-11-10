# =============================================================
# 🧠 MedText — Medical Research Paper Summarizer & Q&A
# =============================================================
# ▶️ Install dependencies before running:
# pip install streamlit PyPDF2 openai

import streamlit as st
import PyPDF2
from openai import OpenAI
import io

# =============================================================
# 🔐 Load OpenRouter API Key (from Streamlit Secrets)
# =============================================================
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("❌ API key missing! Please add it under Streamlit → Settings → Secrets as:\n\nOPENROUTER_API_KEY = 'sk-or-v1-xxxxxx'")
    st.stop()

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODEL = "gpt-4o-mini"

# =============================================================
# 📘 Function: Extract text from PDF
# =============================================================
def extract_text_from_pdf(uploaded_file):
    """Extract raw text content from uploaded PDF."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip(), len(reader.pages)
    except Exception as e:
        st.error(f"⚠️ Error extracting text: {e}")
        return "", 0

# =============================================================
# 🧠 Function: Summarize Research Paper
# =============================================================
def summarize_paper(pdf_text):
    """Summarize the research paper content."""
    if not pdf_text:
        return "❌ No text extracted from PDF."

    prompt = f"""
You are a **medical research summarizer**.
Summarize the following research paper strictly using its content.

If any section is missing in the paper, write "Not clearly mentioned in the document."

📄 **Research Paper:**
{pdf_text[:12000]}

Provide summary in the format:
1. 🩺 Title/Topic
2. 🎯 Objective
3. ⚗️ Methods
4. 🔑 Key Findings
5. 🧾 Conclusion
"""

    try:
        with st.spinner("🧠 Generating structured summary..."):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise medical paper summarizer. Use only the text from the document."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# =============================================================
# 💬 Function: Answer Questions Strictly from PDF
# =============================================================
def answer_from_pdf(pdf_text, question):
    """Answer questions using only the provided PDF content."""
    if not pdf_text:
        return "❌ Please upload a valid PDF first."

    if not question.strip():
        return "⚠️ Please enter a valid question."

    prompt = f"""
You are a **medical research assistant**.
Your job is to answer ONLY based on the following research paper text.
Do NOT use any outside knowledge.

If the answer cannot be found or inferred directly from the paper, respond:
"❌ Not enough information found in the uploaded paper to answer this question."

📄 **Research Paper Content:**
{pdf_text[:12000]}

💬 **Question:**
{question}

🧠 **Answer (from the document only):**
"""

    try:
        with st.spinner("💭 Searching the document for an answer..."):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Answer questions strictly from the PDF content. Never add external facts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=700,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

# =============================================================
# 🎨 Streamlit Interface
# =============================================================
st.set_page_config(page_title="🧠 MedText Research Summarizer", layout="wide")

st.title("🧠 MedText — Medical Research Paper Summarizer & Q&A")
st.caption("Upload a medical research paper (PDF), get a structured summary, and ask domain-specific questions based ONLY on the document. Powered by OpenRouter GPT-4o-mini.")

uploaded_file = st.file_uploader("📄 Upload your research paper (PDF only)", type=["pdf"])

if uploaded_file:
    st.success(f"✅ Uploaded: `{uploaded_file.name}`")

    pdf_text, total_pages = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        st.info(f"📘 Extracted content from **{total_pages} pages**.")

        # === Summarize Section ===
        st.subheader("🔍 Summarize Paper")
        if st.button("✨ Generate Summary"):
            summary = summarize_paper(pdf_text)
            st.markdown("### 📋 Summary")
            st.write(summary)

            # Download Option
            if summary and not summary.startswith("⚠️"):
                summary_bytes = io.BytesIO(summary.encode("utf-8"))
                st.download_button(
                    label="⬇️ Download Summary (.txt)",
                    data=summary_bytes,
                    file_name="Research_Paper_Summary.txt",
                    mime="text/plain"
                )

        st.markdown("---")
        # === Q&A Section ===
        st.subheader("💬 Ask a Question About This Paper")
        question = st.text_input("Enter your question here:")
        if st.button("🧠 Get Answer from PDF"):
            answer = answer_from_pdf(pdf_text, question)
            st.markdown("### ✅ Answer (from PDF)")
            st.write(answer)
    else:
        st.warning("⚠️ Could not extract text. Ensure the PDF is text-based (not scanned image).")
else:
    st.info("📤 Upload a medical research paper to begin.")

st.markdown("---")
st.caption("🚀 MedText | AI-Powered Medical Research Summarizer (Context-Locked Mode)")
