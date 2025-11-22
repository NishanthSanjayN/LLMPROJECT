import streamlit as st
from PyPDF2 import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from langchain_community.llms import HuggingFacePipeline
from langchain_core.embeddings import Embeddings


# ---- Custom Embedding Wrapper ----
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0]


# ---- UI ----
st.set_page_config(page_title="DocuGPT", layout="centered")
st.header("📄 DocuGPT")

with st.sidebar:
    st.title("My Notes")
    uploaded_file = st.file_uploader("Upload Notes PDF and start asking questions", type="pdf")


if uploaded_file:
    # Extract text
    reader = PdfReader(uploaded_file)
    raw_text = "".join(page.extract_text() or "" for page in reader.pages)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    # Create vector store
    embeddings = LocalEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever()

    st.success("📂 PDF processed successfully!")

    # Ask for user query
    question = st.text_input("Ask a question about the PDF:")

    if question:
        # Load FLAN-T5
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            max_length=512,
            temperature=0.5,
        )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Retrieve relevant chunks
        docs = retriever.invoke(question)

        # Merge chunks into a prompt (truncate to avoid limit)
        context = "\n\n".join([d.page_content for d in docs])

        # 🔥 Prevent token overflow
        MAX_CHARS = 2000  # tweak depending on model limits
        context = context[:MAX_CHARS]

        # Construct final prompt
        prompt = f"""
You are an AI assistant. Answer concisely based ONLY on the context.

Context:
{context}

Question: {question}

Answer:
"""

        # Generate response
        response = llm.invoke(prompt)

        st.subheader("Answer:")
        st.info(response)
