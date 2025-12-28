import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

COLLECTION_NAME = "my_collection"
DATA_FOLDER = "data"

# Load documents
def load_docs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = filename

            documents.extend(docs)
    return documents

# Split documents
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# Ingest pipeline
def ingest():
    print("📄 Loading documents...")
    documents = load_docs(DATA_FOLDER)

    if not documents:
        raise ValueError("No documents found in data folder.")

    print("✂️ Splitting documents into chunks...")
    chunks = split_docs(documents)
    print(f"🔹 Total chunks created: {len(chunks)}")

    print("🔍 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("🚀 Starting Qdrant (in-memory, no Docker)...")
    client = QdrantClient(path="./qdrant_data")

    # Create collection explicitly
    print("📦 Creating collection...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print("📥 Uploading embeddings to Qdrant...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    vector_store.add_documents(chunks)

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    ingest()
