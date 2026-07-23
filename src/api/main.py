import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.api.schemas import AskRequest, AskResponse, UploadResponse
from src.processing.chunking import chunk_pages
from src.processing.parse_pdf import parse_pdf
from src.rag.embeddings import embed_chunks
from src.rag.retriever import answer_question
from src.rag.vector_store import upsert_chunks

app = FastAPI(
    title="RAG Docs API",
    description="Faça perguntas sobre seus documentos PDF.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = parse_pdf(tmp_path)
        if not pages:
            raise HTTPException(status_code=422, detail="Não foi possível extrair texto do PDF.")

        chunks = chunk_pages(pages)
        chunks_with_embeddings = embed_chunks(chunks)
        total = upsert_chunks(chunks_with_embeddings)

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(
        message="Documento processado e indexado com sucesso.",
        filename=file.filename,
        chunks_indexed=total,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    result = answer_question(request.question)
    return AskResponse(**result)