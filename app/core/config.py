"""
Central application configuration.

All the configurable values like (models, limits, path) pass through here. 
Nothing should be hard-coded elsewhere in the project.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Smart Help API"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    api_prefix: str = "/api"

    hf_token: str = ""

    whisper_model_name: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    text_model_name: str = "REPLACE_WITH_AFROXLMR_CHECKPOINT"
    vision_model_name: str = "openai/clip-vit-base-patch32"
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    max_text_length: int = 2000
    max_audio_size_mb: int = 10
    max_image_size_mb: int = 5
    max_images: int = 3
    max_audio_duration_seconds: int = 300

    allowed_audio_extensions: str = ".mp3,.wav"
    allowed_image_extensions: str = ".jpg,.jpeg,.png"

    knowledge_base_path: Path = Path("data/knowledge_base/politique_support.yaml")
    faiss_index_path: Path = Path("data/index/support.index")
    faiss_metadata_path: Path = Path("data/index/metadata.json")

    rag_top_k: int = 3
    rag_similarity_threshold: float = 0.45
    relevance_threshold: float = 0.70
    uncertain_relevance_threshold: float = 0.45
    vision_threshold: float = 0.60

    cache_dir: Path = Path("data/cache")
    cache_ttl_seconds: int = 86400
    temp_dir: Path = Path("data/tmp")

    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # Helpers pour convertir les chaînes séparées par des virgules en ensembles (sets)
    @property
    def audio_extensions_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_audio_extensions.split(",")}

    @property
    def image_extensions_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_image_extensions.split(",")}

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()