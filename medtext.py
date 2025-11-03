# ==============================================
# 🧠 Medical Research Paper Summarizer (Streamlit + OpenRouter)
# ==============================================

# ⚙️ STEP 1: Install required packages
# Run once before starting the app:
# !pip install streamlit PyPDF2 openai

import streamlit as st
import PyPDF2
from openai import OpenAI
import io
import os

# ==============================================
# 🔒 Load Hidden API Key (From Streamlit Secrets)
# ==============================================
# Do NOT store your key in this file
# Instead, go to Streamlit Cloud → Settings → Secrets → and paste:
# OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxx"

if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("⚠️ API key not found. Please add it in Streamlit Secrets.")
else:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODEL_NAME = "gpt-4o-mini"

# ==============================================
# 🧾 Extract text from PDF
# ==============================================
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(reader.pages)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip(), total_pages
    except Exception as e:
        st.error(f"⚠️ Error reading PDF: {e}")
        return "", 0

# ==============================================
# 🧠 Summarize Research Paper
# ==============================================
def summarize_research_paper(pdf_text):
    if not pdf_text:
        return "❌ No text to summarize."

    try:
        with st.spinner("🧠 Summarizing the paper... please wait..."):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical research expert assistant that provides "
                            "structured, concise summaries of research papers."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
Summarize this medical research paper in a structured format:

**Research Paper Content:**
{pdf_text[:12000]}

**Provide a summary with these sections:**
1. **Title/Topic**
2. **Objective**
3. **Methods**
4. **Key Findings**
5. **Conclusion**

Keep it concise but informative.
""",
                    },
                ],
                temperature=0.3,
                max_tokens=1000,
            )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# ==============================================
# 💬 Ask Questions About the Paper
# ==============================================
def ask_research_question(pdf_text, question):
    if not pdf_text:
        return "❌ Please upload and process a PDF first."

    try:
        with st.spinner("💭 Thinking..."):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical research expert who answers questions based on provided research papers.",
                    },
                    {
                        "role": "user",
                        "content": f"""
Based on this research paper, answer the following question:

**Research Paper Context:**
{pdf_text[:12000]}

**Question:** {question}

**Answer:**
""",
                    },
                ],
                temperature=0.3,
                max_tokens=600,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

# ==============================================
# 🧩 Streamlit UI
# ==============================================
st.set_page_config(page_title="🧠 Medical Paper Summarizer", layout="wide")

st.title("🧠 Medical Research Paper Summarizer")
st.caption("Summarize and analyze medical research papers instantly using AI (powered by OpenRouter GPT-4o-mini).")

uploaded_file = st.file_uploader("📄 Upload your research paper (PDF only)", type=["pdf"])

summary_output = None  # To store summary for download

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    pdf_text, total_pages = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        st.info(f"📄 Successfully extracted text from {total_pages} pages.")

        if st.button("🔍 Summarize Paper"):
            summary_output = summarize_research_paper(pdf_text)
            st.subheader("📋 Research Paper Summary")
            st.write(summary_output)

            # ✅ Download Button for Summary
            if summary_output and not summary_output.startswith("⚠️"):
                summary_bytes = io.BytesIO(summary_output.encode("utf-8"))
                st.download_button(
                    label="⬇️ Download Summary as TXT",
                    data=summary_bytes,
                    file_name="Research_Paper_Summary.txt",
                    mime="text/plain"
                )

        st.markdown("---")
        st.subheader("💬 Ask Questions About This Paper")

        question = st.text_input("Enter your question:")
        if st.button("Ask"):
            if question.strip():
                answer = ask_research_question(pdf_text, question)
                st.success("✅ Answer:")
                st.write(answer)
            else:
                st.warning("⚠️ Please enter a question first.")

else:
    st.warning("📤 Please upload a PDF file to begin.")

st.markdown("---")
st.caption("MedText Summarizer")
