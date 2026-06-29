from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Voice Agent API"
    environment: str = "development"

    device_preference: str = "cuda"
    model_id: str = "openbmb/VoxCPM-0.5B"
    model_local_path: str | None = None

    storage_dir: str = "./storage"
    database_url: str = "sqlite:///./voice_agent.db"

    max_text_length: int = 2000
    default_cfg_value: float = 2.0
    default_inference_timesteps: int = 10
    sample_rate: int = 16000

    cors_origins: str = "http://localhost:3000"
    api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def voices_dir(self) -> Path:
        path = Path(self.storage_dir) / "voices"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def outputs_dir(self) -> Path:
        path = Path(self.storage_dir) / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def loras_dir(self) -> Path:
        path = Path(self.storage_dir) / "loras"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
