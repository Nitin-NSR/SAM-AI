from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Enterprise Multi-Agent Knowledge Assistant"
    environment: str = "development"
    jwt_secret: str = "replace-this-development-secret"
    access_token_minutes: int = Field(default=60, gt=0, le=1440)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0, le=100 * 1024 * 1024)
    max_request_chars: int = Field(default=4000, gt=1, le=32000)
    hf_token: str | None = None
    hf_chat_model: str = "HuggingFaceH4/zephyr-7b-beta"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "./data/chroma"
    chunk_size: int = Field(default=900, gt=100, le=10000)
    chunk_overlap: int = Field(default=120, ge=0, le=5000)
    max_chunks: int = Field(default=8, gt=0, le=50)
    checkpoint_db_url: str = "sqlite:///./data/checkpoints.db"
    database_url: str = "sqlite:///./data/app.db"
    redis_url: str | None = None
    upload_directory: str = "./data/uploads"
    conversation_max_messages: int = 24
    rate_limit_per_minute: int = 30
    web_search_url: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return self

    def validate_production_secrets(self) -> None:
        if self.environment.lower() == "production" and len(self.jwt_secret) < 32:
            raise ValueError("JWT_SECRET must contain at least 32 characters in production.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
