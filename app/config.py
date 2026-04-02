from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    groq_api_key: str = ""
    llm_fast_model: str = "groq/llama-3.1-8b-instant"
    llm_main_model: str = "groq/llama-3.3-70b-versatile"
    llm_vision_model: str = "groq/meta-llama/llama-4-scout-17b-16e-instruct"

    # Embeddings (local SentenceTransformer)
    llm_embedding_model: str = "BAAI/bge-m3"
    llm_embedding_dimensions: int = 1024

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "voltedge-warranty"
    pinecone_namespace: str = "warranty_policy_v1"

    # RAG
    rag_similarity_threshold: float = 0.75
    rag_top_k: int = 5

    # Storage
    image_max_size_mb: int = 10
    results_dir: str = "./results"

    # Langfuse
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"
    langfuse_project_name: str = "warranty-agent"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_tokens_per_session: int = 8000
    log_http_bodies: bool = False
    log_http_body_max_chars: int = 4000


settings = Settings()
