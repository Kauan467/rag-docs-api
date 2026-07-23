from pinecone import Pinecone
from src.config import settings


def get_index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index_name)


def upsert_chunks(chunks_with_embeddings: list[dict]) -> int:
    index = get_index()
    vectors = []
    for chunk in chunks_with_embeddings:
        vectors.append({
            "id": chunk["chunk_id"],
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
            },
        })
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])
    return len(vectors)


def search_similar(query_embedding: list[float], top_k: int = None) -> list[dict]:
    index = get_index()
    k = top_k or settings.top_k_results
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True,
    )
    matches = []
    for match in results.matches:
        matches.append({
            "text": match.metadata["text"],
            "source": match.metadata["source"],
            "page": match.metadata["page"],
            "score": round(match.score, 4),
        })
    return matches


def delete_document(source_name: str) -> None:
    index = get_index()
    index.delete(filter={"source": {"$eq": source_name}})
