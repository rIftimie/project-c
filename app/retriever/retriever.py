from ..storage.chroma_client import get_chroma_client
from sentence_transformers import SentenceTransformer
from typing import List

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query: str, top_k=5) -> List[dict]:
    client = get_chroma_client()
    collection = client.get_collection("think_bro")

    query_vector = embedder.encode(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        context_blocks.append({
            "text": doc,
            "title": meta.get("title", "Unknown Video"),
            "channel": meta.get("channel", "Unknown Channel"),
            "published": meta.get("published", ""),
            "url": meta.get("url", ""),
            "start": meta.get("start", 0),
            "video_id": meta.get("video_id", "")
        })

    return context_blocks
