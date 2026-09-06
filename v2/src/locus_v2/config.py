from decimal import Decimal
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
    auth_enable_password_auth: bool = False
    cors_origins: list[str] = ["http://localhost:4201", "http://localhost:8100"]

    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    tool_model: str = "gpt-5-mini"
    tool_timeout_seconds: float = Field(default=60.0, gt=0)
    event_log_retention_days: int = Field(default=30, ge=1, le=365)

    billing_usd_to_eur: Decimal = Field(default=Decimal("0.87"), gt=0)
    billing_margin_multiplier: Decimal = Field(default=Decimal("2.20"), ge=1)
    billing_min_realtime_call_charge_cents: int = Field(default=3, ge=0)
    billing_worker_poll_seconds: float = Field(default=1.0, gt=0, le=60)

    wikidata_base_url: str = "https://www.wikidata.org"
    wikidata_language: str = "es"
    wikidata_sparql_url: str = "https://query.wikidata.org/sparql"
    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    overpass_api_url: str = "https://overpass-api.de/api/interpreter"
    overpass_timeout_seconds: int = 25

    google_play_package_name: str = "com.carlos.locusia"
    google_play_verify_purchases: bool = True
    google_play_service_account_json: str = ""
    google_play_service_account_file: str = ""

    app_android_latest_version_code: int = 10
    app_android_update_url: str = ""
    app_ios_latest_build: int = 1
    app_ios_update_url: str = ""

    def android_update_url(self) -> str:
        return self.app_android_update_url or (
            f"https://play.google.com/store/apps/details?id={self.google_play_package_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
