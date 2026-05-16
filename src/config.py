from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "rag-docs"

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "rag-docs-bucket"

    # app
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
