"""Application configuration loaded from environment variables via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object. All values can be overridden via .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Ollama
    ollama_host: str = "http://klarki-ollama:11434"
    ollama_model: str = "phi3:mini"

    # ChromaDB
    chromadb_host: str = "http://klarki-chromadb:8000"

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Uploads
    upload_max_size_mb: int = 10
    upload_dir: str = "/data/uploads"

    # Triton (Phase 5)
    use_triton: bool = False
    triton_host: str = "klarki-triton"
    triton_grpc_port: int = 8001

    @property
    def upload_max_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.upload_max_size_mb * 1024 * 1024

    @property
    def chromadb_url(self) -> str:
        """ChromaDB base URL."""
        return self.chromadb_host

    @property
    def triton_grpc_address(self) -> str:
        """Triton gRPC address string."""
        return f"{self.triton_host}:{self.triton_grpc_port}"


settings = Settings()
