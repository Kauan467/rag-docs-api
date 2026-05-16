from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import settings


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Divide páginas em chunks menores preservando metadados.

    Entrada: lista de dicts {page, text, source}
    Saída:   lista de dicts {chunk_id, text, source, page, chunk_index}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])

        for idx, split in enumerate(splits):
            chunk_id = f"{page['source']}_p{page['page']}_c{idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": split,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": idx,
            })

    return chunks