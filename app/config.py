import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


_ORIGINAL_ENV = dict(os.environ)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PRODUCTION_ENV_PATH = _PROJECT_ROOT / "env.pro"

if _PRODUCTION_ENV_PATH.exists():
    load_dotenv(_PRODUCTION_ENV_PATH)
else:
    load_dotenv(_PROJECT_ROOT / ".env")

    for _key, _value in dotenv_values(_PROJECT_ROOT / ".env.local").items():
        if _value is None:
            continue
        if _key not in _ORIGINAL_ENV:
            os.environ[_key] = _value


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def _normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("mysql://"):
        return raw_url.replace("mysql://", "mysql+pymysql://", 1)
    return raw_url


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_database_url() -> str:
    direct_url = _env("MYSQL_PUBLIC_URL", "DATABASE_PUBLIC_URL", "MYSQL_URL", "DATABASE_URL", default="")
    if direct_url:
        return _normalize_database_url(direct_url)

    driver = _env("DB_DRIVER", default="mysql+pymysql")
    host = _env("DB_HOST", "MYSQLHOST", default="localhost")
    port = _env("DB_PORT", "MYSQLPORT", default="3306")
    name = _env("DB_NAME", "MYSQLDATABASE", "MYSQL_DATABASE", default="locus")
    user = _env("DB_USER", "MYSQLUSER", "MYSQL_USER", default="locus")
    password = _env("DB_PASSWORD", "MYSQLPASSWORD", "MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD", default="")
    return f"{driver}://{user}:{password}@{host}:{port}/{name}"


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Locus API")
    app_env: str = os.getenv("APP_ENV", "local")
    app_build: str = os.getenv("APP_BUILD", "v2-scaffold")
    app_secret: str = os.getenv("APP_SECRET", "locus-dev-secret")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    auth_token_ttl_days: int = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "30"))
    auth_enable_password_auth: bool = _bool_env("AUTH_ENABLE_PASSWORD_AUTH", False)
    google_auth_client_ids: list[str] = field(
        default_factory=lambda: _csv_env(
            "GOOGLE_AUTH_CLIENT_IDS",
            _env("GOOGLE_AUTH_WEB_CLIENT_ID", "GOOGLE_WEB_CLIENT_ID", default=""),
        )
    )
    cors_allowed_origins: list[str] = field(
        default_factory=lambda: _csv_env(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:8100,http://127.0.0.1:8100,https://localhost,capacitor://localhost,ionic://localhost,https://locus-backend-production.up.railway.app",
        )
    )

    db_driver: str = _env("DB_DRIVER", default="mysql+pymysql")
    db_host: str = _env("DB_HOST", "MYSQLHOST", default="localhost")
    db_port: int = int(_env("DB_PORT", "MYSQLPORT", default="3306"))
    db_name: str = _env("DB_NAME", "MYSQLDATABASE", "MYSQL_DATABASE", default="locus")
    db_user: str = _env("DB_USER", "MYSQLUSER", "MYSQL_USER", default="locus")
    db_password: str = _env("DB_PASSWORD", "MYSQLPASSWORD", "MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD", default="")
    database_url: str = _build_database_url()

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini")
    openai_chat_enable_web_search: bool = _bool_env("OPENAI_CHAT_ENABLE_WEB_SEARCH", True)
    openai_response_timeout_seconds: int = int(os.getenv("OPENAI_RESPONSE_TIMEOUT_SECONDS", "180"))
    openai_realtime_model: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
    openai_realtime_voice: str = os.getenv("OPENAI_REALTIME_VOICE", "cedar")
    openai_realtime_secret_ttl_seconds: int = int(os.getenv("OPENAI_REALTIME_SECRET_TTL_SECONDS", "600"))
    openai_realtime_max_output_tokens: int = int(os.getenv("OPENAI_REALTIME_MAX_OUTPUT_TOKENS", "1400"))
    openai_realtime_retention_ratio: float = float(os.getenv("OPENAI_REALTIME_RETENTION_RATIO", "0.80"))
    openai_realtime_post_instructions_token_limit: int = int(os.getenv("OPENAI_REALTIME_POST_INSTRUCTIONS_TOKEN_LIMIT", "8000"))
    openai_realtime_input_transcription_model: str = os.getenv("OPENAI_REALTIME_INPUT_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    openai_realtime_input_transcription_language: str = os.getenv("OPENAI_REALTIME_INPUT_TRANSCRIPTION_LANGUAGE", "es")

    maps_api_key: str = os.getenv("MAPS_API_KEY", "")
    web_search_provider: str = os.getenv("WEB_SEARCH_PROVIDER", "brave")
    web_search_api_key: str = _env("WEB_SEARCH_API_KEY", "BRAVE_SEARCH_API_KEY", default="")
    web_search_base_url: str = os.getenv("WEB_SEARCH_BASE_URL", "https://api.search.brave.com/res/v1")
    web_search_timeout_seconds: int = int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "10"))
    web_search_country: str = os.getenv("WEB_SEARCH_COUNTRY", "ES")
    web_search_language: str = os.getenv("WEB_SEARCH_LANGUAGE", "es")
    wikipedia_language: str = os.getenv("WIKIPEDIA_LANGUAGE", "es")
    wikidata_base_url: str = os.getenv("WIKIDATA_BASE_URL", "https://www.wikidata.org")
    wikidata_language: str = os.getenv("WIKIDATA_LANGUAGE", "es")
    wikidata_sparql_url: str = os.getenv("WIKIDATA_SPARQL_URL", "https://query.wikidata.org/sparql")
    overpass_api_url: str = os.getenv("OVERPASS_API_URL", "https://overpass-api.de/api/interpreter")
    overpass_timeout_seconds: int = int(os.getenv("OVERPASS_TIMEOUT_SECONDS", "25"))

    billing_margin_multiplier: float = float(os.getenv("BILLING_MARGIN_MULTIPLIER", "1.80"))
    billing_usd_to_eur: float = float(os.getenv("BILLING_USD_TO_EUR", "1.00"))
    billing_signup_bonus_cents: int = int(os.getenv("BILLING_SIGNUP_BONUS_CENTS", "200"))
    billing_min_reserve_cents: int = int(os.getenv("BILLING_MIN_RESERVE_CENTS", "25"))
    billing_min_realtime_call_charge_cents: int = int(os.getenv("BILLING_MIN_REALTIME_CALL_CHARGE_CENTS", "3"))
    billing_manual_topups_enabled: bool = _bool_env("BILLING_MANUAL_TOPUPS_ENABLED", False)
    billing_client_usage_recording_enabled: bool = _bool_env("BILLING_CLIENT_USAGE_RECORDING_ENABLED", False)
    google_play_package_name: str = os.getenv("GOOGLE_PLAY_PACKAGE_NAME", "com.carlos.locusia")
    google_play_verify_purchases: bool = _bool_env("GOOGLE_PLAY_VERIFY_PURCHASES", True)
    google_play_service_account_json: str = os.getenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", "")
    google_play_service_account_file: str = os.getenv("GOOGLE_PLAY_SERVICE_ACCOUNT_FILE", "")

    getyourguide_referrals_enabled: bool = _bool_env("GETYOURGUIDE_REFERRALS_ENABLED", True)
    getyourguide_partner_id: str = os.getenv("GETYOURGUIDE_PARTNER_ID", "")

    legacy_gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    legacy_livekit_url: str = os.getenv("LIVEKIT_URL", "")
    legacy_livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "")
    legacy_livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")


settings = Settings()
