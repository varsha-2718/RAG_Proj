import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import json

COLLECTION_NAME = "my_collection"

# OLLAMA CALL
def ask_ollama(context, question):
    url = "http://localhost:11434/api/generate"

    prompt = f"""
You are a policy question-answering assistant.

Rules:
- Answer ONLY using the context below
- If the answer is not in the context, say: "The information is not available in the provided documents."
- Be concise and structured.

Context:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    return response.json().get("response", "No response from model")

# STREAMLIT UI 
st.title("📄 H&M Policy Document Q&A (RAG)")
st.markdown("---")

question = st.text_input("Ask a question about the policy documents:")

if question:
    with st.spinner("Searching documents..."):
        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # In-memory Qdrant (same as ingestion)
        client = QdrantClient(path="./qdrant_data")


        # Load vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )

        docs = vector_store.similarity_search(question, k=3)

    if not docs:
        st.warning("No relevant documents found.")
    else:
        context = "\n\n".join([doc.page_content for doc in docs])

        with st.spinner("Generating answer..."):
            answer = ask_ollama(context, question)

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("📚 Retrieved Context")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content)