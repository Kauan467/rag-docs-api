from openai import OpenAI
from src.config import settings

client = OpenAI(api_key=settings.openai_api_key)

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    gera embeddings para cada chunk via OpenAI.
    adiciona o campo 'embedding' em cada dict.
    processa em batches de 100 para respeitar rate limits.
    """
    batch_size = 100
    result = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        for chunk, embedding_obj in zip(batch, response.data):
            result.append({**chunk, "embedding": embedding_obj.embedding})

    return result


def embed_query(query: str) -> list[float]:
    """gera embedding para uma pergunta de busca."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding
