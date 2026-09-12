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

    # Defaults to production on purpose: `local` unlocks POST /admin/v2/auth/local,
    # which hands out an admin session with no credentials at all. Nothing about
    # that route checks the caller's address — "local" is a claim in the config,
    # not a fact about the network — so a deploy that forgets LOCUS_ENV must fail
    # closed, not open. Both .env.local and .env.example set it explicitly, so
    # local development is unaffected.
    env: Literal["local", "production"] = "production"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8100
    realtime_port: int = 8101
    # Public origin + path prefix this API is reached at, e.g.
    # "https://api.locusguide.es/api". Only used to build absolute URLs that
    # must work from inside the Ionic app (call photos), which is served from a
    # different origin. Deliberately explicit rather than derived from the
    # request: behind a proxy the request host is the proxy's, not ours.
    public_api_base_url: str = "http://localhost:8200/api"

    # Which commit is actually running, injected by bin/pull-deploy at deploy
    # time. The point is to be able to answer "is the last commit live?" without
    # opening an SSH session. Left as "dev" outside a deploy, which is honest:
    # a working copy is not any particular commit. Never derived from the local
    # .git, because the deployed tree is an rsync copy with no .git at all.
    build_sha: str = "dev"
    build_time: str = ""

    database_url: str = "mysql+asyncmy://locus_v2:locus_v2@localhost:3307/locus_v2"
    legacy_database_url: str | None = None
    redis_url: str = "redis://localhost:6380/0"

    jwt_secret: SecretStr = Field(min_length=32)
    jwt_issuer: str = "locus-v2"
    jwt_access_minutes: int = 30
    jwt_refresh_days: int = 30
    admin_email: EmailStr
    # Second lock on the credential-free admin login, and also fail-closed: both
    # this and `env` have to be permissive for that route to answer at all.
    allow_insecure_local_admin: bool = False
    google_auth_client_ids: list[str] = Field(default_factory=list)
    admin_session_days: int = 7
    admin_session_cookie: str = "locus_admin_session"
    auth_enable_password_auth: bool = False
    cors_origins: list[str] = [
        "http://localhost:4201", "http://localhost:8100", "http://localhost:8200"
    ]

    openai_api_key: SecretStr | None = None
    # Both ported from V1 (OPENAI_BASE_URL / OPENAI_RESPONSE_TIMEOUT_SECONDS), which
    # applied them to every OpenAI call. Without the timeout a stuck request rides the
    # SDK's default, far longer than V1 would ever have waited, and a chat turn hangs
    # with it. None on the base URL means the SDK's own default endpoint.
    openai_base_url: str | None = None
    openai_timeout_seconds: float = Field(default=180.0, gt=0)
    gemini_api_key: SecretStr | None = None
    tool_model: str = "gpt-5-mini"
    # Speech-to-text for the shared log. The live provider transcribes the user
    # too, but badly enough to matter: those lines are what the group reads and
    # what rebuilds the context on every provider reconnection. Empty disables
    # it and falls back to the provider's own transcript.
    #
    # A realtime transcription model, not a batch one: it runs in its own
    # session next to the conversation and, unlike the batch endpoint, can be
    # told which proper nouns to expect — which is what was actually coming
    # back wrong.
    transcription_model: str = "gpt-live-transcribe"
    # gpt-5-mini's reasoning tokens count against wall-clock time, not just the token
    # budget: a real document_poi call at the 8000-token ceiling (see voice/tools.py)
    # timed out at exactly 60s (2026-09-06, live), got killed mid-reasoning, and the
    # model narrated the group-call opening without it. Raised so raising the token
    # ceiling doesn't just trade "empty answer" for "no answer at all".
    tool_timeout_seconds: float = Field(default=120.0, gt=0)
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

    # Google Places Text Search, used by the map chat's live place lookup
    # (places/client.py). Optional: with no key the search falls back to the
    # catalog alone, which covers landmarks but not restaurants/pharmacies.
    maps_api_key: SecretStr | None = None
    maps_timeout_seconds: float = Field(default=10.0, gt=0)

    google_play_package_name: str = "com.carlos.locusia"
    google_play_verify_purchases: bool = True
    google_play_service_account_json: str = ""
    google_play_service_account_file: str = ""

    billing_min_reserve_cents: int = 25
    # 200, matching V1's own default and the value in its live environment. V2 had
    # 100, so a new account would have silently received half the welcome credit
    # after the cutover (found 2026-09-08 comparing defaults, not just names).
    billing_signup_bonus_cents: int = Field(default=200, ge=0)
    billing_manual_topups_enabled: bool = False

    # Pagos web, solo para la PWA. Sin clave la función queda apagada y los
    # endpoints responden 503: mejor no poder recargar que abrir un camino de
    # cobro a medio configurar. En Android no se usa nunca — la politica de
    # Google Play exige que los bienes digitales dentro de la app pasen por Play
    # Billing, y saltarsela cuesta la retirada de la ficha.
    #
    # Paddle es el proveedor activo. Se eligio por ser *merchant of record*: el
    # vendedor de cara al cliente es Paddle, que se ocupa del IVA, y eso permite
    # cobrar sin estar dado de alta como empresa. Cuesta 5% + 0,50 $ frente al
    # ~4% de Stripe, y sigue por debajo del 15% de Google Play.
    paddle_api_key: SecretStr | None = None
    paddle_client_token: str = ""
    paddle_webhook_secret: SecretStr | None = None
    paddle_sandbox: bool = True

    # Stripe queda escrito y probado, pero apagado: su formulario de alta no
    # tiene opcion de persona fisica en España, solo empresa o autonomo. El dia
    # que haya alta, es mas barato que Paddle y basta con poner estas claves.
    stripe_secret_key: SecretStr | None = None
    stripe_webhook_secret: SecretStr | None = None
    # Donde vuelve el usuario al terminar o cancelar el pago. Explicito y no
    # derivado de la peticion, por lo mismo que public_api_base_url: detras de un
    # proxy el host de la peticion es el del proxy, no el nuestro.
    web_app_base_url: str = "https://app.locusguide.es"

    getyourguide_referrals_enabled: bool = True
    getyourguide_partner_id: str = ""

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
