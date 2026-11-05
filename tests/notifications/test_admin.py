"""Avisos al administrador de las cuentas nuevas (notifications/admin.py)."""

from typing import Any

from locus_v2.identity.models import User
from locus_v2.notifications import admin
from locus_v2.notifications.apns import SendResult
from locus_v2.notifications.models import PushDevice


def test_the_notice_says_which_app_and_how() -> None:
    thilo = User(display_name="Thilo Richter", preferred_name="", signup_app="borea",
                 auth_provider="apple")
    assert admin.message(thilo) == ("Nueva cuenta en Borea", "Thilo se ha registrado con Apple.")
    ana = User(display_name="Ana", preferred_name="", signup_app="guide", auth_provider="google")
    assert admin.message(ana) == ("Nueva cuenta en Locus", "Ana se ha registrado con Google.")


class _Redis:
    def __init__(self, value: str | None) -> None:
        self.value = value

    async def get(self, key: str) -> str | None:
        return self.value

    async def set(self, key: str, value: Any) -> None:
        self.value = str(value)


class _Rows:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def __iter__(self) -> Any:
        return iter(self.rows)


class _Session:
    """Contesta en el orden en que pregunta notify_new_users."""

    def __init__(self, scalar: list[Any], scalars: list[list[Any]]) -> None:
        self.scalar_answers = scalar
        self.scalars_answers = scalars

    async def scalar(self, _: Any) -> Any:
        return self.scalar_answers.pop(0)

    async def scalars(self, _: Any) -> _Rows:
        return _Rows(self.scalars_answers.pop(0))


class _Sender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send(self, token: str, topic: str, title: str, body: str) -> SendResult:
        self.sent.append((title, body))
        return SendResult("enviada", "")


class _Settings:
    admin_email = "admin@example.com"
    borea_app_store_bundle_id = "com.carlos.borea"
    app_store_bundle_id = "com.carlos.locusia"


async def test_the_first_round_only_marks_the_newest_account() -> None:
    redis, sender = _Redis(None), _Sender()
    session = _Session(scalar=[41], scalars=[])
    sent = await admin.notify_new_users(session, _Settings(), redis, sender)  # type: ignore[arg-type]
    assert (sent, redis.value, sender.sent) == (0, "41", [])


async def test_each_new_account_reaches_the_admin_phone_once() -> None:
    redis, sender = _Redis("41"), _Sender()
    nueva = User(id=42, display_name="Marie Dupont", preferred_name="", signup_app="borea",
                 auth_provider="google")
    movil = PushDevice(id=1, user_id=1, app="borea", platform="ios", token="t")
    session = _Session(scalar=[1], scalars=[[nueva], [movil]])
    sent = await admin.notify_new_users(session, _Settings(), redis, sender)  # type: ignore[arg-type]
    assert sent == 1
    assert redis.value == "42"
    assert sender.sent == [("Nueva cuenta en Borea", "Marie se ha registrado con Google.")]
