"""La sesión de la app se renueva con el uso.

Caducaba a los 30 días del login aunque se usara a diario, así que una vez al mes
había que volver a entrar con Google y Google mandaba su correo de aviso. Ahora
caduca a los 30 días sin abrir la app.
"""

from datetime import timedelta
from types import SimpleNamespace

from locus_v2.identity.application.mobile_auth import MOBILE_SESSION_DAYS, MobileAuthService
from locus_v2.shared.clock import utc_now


class _Session:
    def __init__(self, auth_session, user) -> None:
        self.resultados = [auth_session, user]
        self.commits = 0

    async def scalar(self, _query):
        return self.resultados.pop(0)

    async def commit(self) -> None:
        self.commits += 1


def _servicio(auth_session) -> tuple[MobileAuthService, _Session]:
    session = _Session(auth_session, SimpleNamespace(id=1))
    return MobileAuthService(session, settings=None), session  # type: ignore[arg-type]


async def test_using_the_app_pushes_the_expiry_forward() -> None:
    ahora = utc_now()
    sesion = SimpleNamespace(
        user_id=1,
        last_seen_at=ahora - timedelta(hours=1),
        expires_at=ahora + timedelta(days=2),  # a punto de caducar
    )
    servicio, session = _servicio(sesion)

    assert await servicio.authenticate("token") is not None
    assert sesion.expires_at >= ahora + timedelta(days=MOBILE_SESSION_DAYS - 1)
    assert session.commits == 1


async def test_not_on_every_request() -> None:
    # Se renueva junto con last_seen_at, cada pocos minutos, para no escribir en
    # la base de datos en cada petición.
    ahora = utc_now()
    vence = ahora + timedelta(days=2)
    sesion = SimpleNamespace(user_id=1, last_seen_at=ahora - timedelta(minutes=1), expires_at=vence)
    servicio, session = _servicio(sesion)

    assert await servicio.authenticate("token") is not None
    assert sesion.expires_at == vence
    assert session.commits == 0
