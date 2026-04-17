import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Locus API")
    app_env: str = os.getenv("APP_ENV", "local")
    app_build: str = os.getenv("APP_BUILD", "v2-scaffold")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    auth_token_ttl_days: int = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "30"))

    db_driver: str = os.getenv("DB_DRIVER", "mysql+pymysql")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "3306"))
    db_name: str = os.getenv("DB_NAME", "locus")
    db_user: str = os.getenv("DB_USER", "locus")
    db_password: str = os.getenv("DB_PASSWORD", "")
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"{os.getenv('DB_DRIVER', 'mysql+pymysql')}://"
        f"{os.getenv('DB_USER', 'locus')}:{os.getenv('DB_PASSWORD', '')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/"
        f"{os.getenv('DB_NAME', 'locus')}",
    )

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")
    openai_realtime_model: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    openai_realtime_voice: str = os.getenv("OPENAI_REALTIME_VOICE", "marin")
    openai_realtime_secret_ttl_seconds: int = int(os.getenv("OPENAI_REALTIME_SECRET_TTL_SECONDS", "600"))

    maps_api_key: str = os.getenv("MAPS_API_KEY", "")
    wikipedia_language: str = os.getenv("WIKIPEDIA_LANGUAGE", "es")
    wikidata_base_url: str = os.getenv("WIKIDATA_BASE_URL", "https://www.wikidata.org")
    wikidata_language: str = os.getenv("WIKIDATA_LANGUAGE", "es")
    wikidata_sparql_url: str = os.getenv("WIKIDATA_SPARQL_URL", "https://query.wikidata.org/sparql")

    billing_margin_multiplier: float = float(os.getenv("BILLING_MARGIN_MULTIPLIER", "1.15"))
    billing_usd_to_eur: float = float(os.getenv("BILLING_USD_TO_EUR", "1.00"))
    billing_signup_bonus_cents: int = int(os.getenv("BILLING_SIGNUP_BONUS_CENTS", "200"))
    billing_min_reserve_cents: int = int(os.getenv("BILLING_MIN_RESERVE_CENTS", "25"))

    legacy_gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    legacy_livekit_url: str = os.getenv("LIVEKIT_URL", "")
    legacy_livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "")
    legacy_livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")


settings = Settings()
