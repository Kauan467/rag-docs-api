from pinecone import Pinecone
from src.config import settings

pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index_name)


def upsert_chunks(chunks_with_embeddings: list[dict]) -> int:
    """
    Salva chunks com embeddings no Pinecone.
    Retorna quantidade de vetores inseridos.
    """
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
    """
    Busca os chunks mais similares à query.
    Retorna lista de dicts com texto, fonte, página e score.
    """
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
    """Remove todos os chunks de um documento pelo nome do arquivo."""
    index.delete(filter={"source": {"$eq": source_name}})
    