from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    document: str
    page: int
    excerpt: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int