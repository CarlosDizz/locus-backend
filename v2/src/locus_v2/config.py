from functools import lru_cache
from typing import Literal

from pydantic import EmailStr, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LOCUS_",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["local", "production"] = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8100
    realtime_port: int = 8101

    database_url: str = "mysql+asyncmy://locus_v2:locus_v2@localhost:3307/locus_v2"
    legacy_database_url: str | None = None
    redis_url: str = "redis://localhost:6380/0"

    jwt_secret: SecretStr = Field(min_length=32)
    jwt_issuer: str = "locus-v2"
    jwt_access_minutes: int = 30
    jwt_refresh_days: int = 30
    admin_email: EmailStr
    allow_insecure_local_admin: bool = True
    google_auth_client_ids: list[str] = Field(default_factory=list)
    admin_session_days: int = 7
    admin_session_cookie: str = "locus_admin_session"
    cors_origins: list[str] = ["http://localhost:4201", "http://localhost:8100"]

    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
