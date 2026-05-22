from openai import OpenAI
from src.config import settings
from src.rag.embeddings import embed_query
from src.rag.vector_store import search_similar

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """Você é um especialista em análise de documentos.
Responda à pergunta do usuário APENAS com base no contexto fornecido abaixo.
Se a resposta não estiver no contexto, diga claramente que não encontrou a informação.
Seja preciso e cite a fonte quando relevante."""


def answer_question(question: str) -> dict:
    """
    Pipeline RAG completo:
    1. Gera embedding da pergunta
    2. Busca chunks similares no Pinecone
    3. Monta prompt com contexto
    4. Chama LLM e retorna resposta com fontes
    """
    query_embedding = embed_query(question)
    matches = search_similar(query_embedding)

    if not matches:
        return {
            "answer": "Nenhum documento relevante encontrado. Faça upload de documentos primeiro.",
            "sources": [],
        }

    context_parts = []
    for i, match in enumerate(matches):
        context_parts.append(
            f"[Fonte {i+1}: {match['source']}, página {match['page']}]\n{match['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Contexto dos documentos:\n\n{context}\n\nPergunta: {question}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    sources = [
        {
            "document": m["source"],
            "page": m["page"],
            "excerpt": m["text"][:200] + "..." if len(m["text"]) > 200 else m["text"],
            "relevance_score": m["score"],
        }
        for m in matches
    ]

    return {"answer": answer, "sources": sources}
