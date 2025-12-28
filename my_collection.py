from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "my_collection"

def get_qdrant_client():
    """Returns an authenticated Qdrant client."""
    return QdrantClient(
        url=os.getenv("url"),
        api_key=os.getenv("api")
    )

def create_collection(client):
    """
    Creates a fresh collection for policy embeddings.
    Uses cosine similarity for semantic retrieval.
    """
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
