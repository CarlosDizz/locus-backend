"""Avisos para el administrador: alguien se registra (como en BACUL).

Llegan como push a los móviles de la cuenta de LOCUS_ADMIN_EMAIL. El worker mira
cada minuto si hay cuentas nuevas desde la última que avisó (marca en Valkey): así
el login no espera a Apple y un fallo de APNs no rompe nada. La primera vez solo
pone la marca, para no avisar de todas las cuentas que ya había.
"""

import structlog
from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.config import Settings
from locus_v2.identity.models import User, UserStatus
from locus_v2.notifications.apns import ApnsSender
from locus_v2.notifications.models import PushDevice
from locus_v2.notifications.service import _topic, first_name

logger = structlog.get_logger()

LAST_USER_KEY = "admin_notice:last_user_id"
_APPS = {"borea": "Borea", "guide": "Locus"}
_PROVIDERS = {"google": "Google", "apple": "Apple"}


def message(user: User) -> tuple[str, str]:
    app = _APPS.get(user.signup_app, user.signup_app)
    provider = _PROVIDERS.get(user.auth_provider, user.auth_provider)
    return f"Nueva cuenta en {app}", f"{first_name(user)} se ha registrado con {provider}."


async def notify_new_users(
    session: AsyncSession, settings: Settings, redis: Redis, sender: ApnsSender | None = None
) -> int:
    """Avisa al administrador de cada cuenta nueva. Devuelve cuántos avisos salieron."""
    last = await redis.get(LAST_USER_KEY)
    if last is None:
        newest = await session.scalar(select(func.max(User.id))) or 0
        await redis.set(LAST_USER_KEY, newest)
        return 0
    users = list(await session.scalars(
        select(User).where(User.id > int(last)).order_by(User.id).limit(20)
    ))
    if not users:
        return 0
    # La marca avanza antes de enviar: mejor perder un aviso que repetirlo cada minuto.
    await redis.set(LAST_USER_KEY, users[-1].id)
    admin_email = str(settings.admin_email).lower()
    admin_id = await session.scalar(
        select(User.id).where(User.email == admin_email, User.status == UserStatus.ACTIVE)
    )
    if admin_id is None:
        return 0
    devices = list(await session.scalars(
        select(PushDevice).where(
            PushDevice.user_id == admin_id, PushDevice.disabled_at.is_(None),
            PushDevice.platform == "ios",
        )
    ))
    if not devices:
        return 0
    own_sender = sender is None
    sender = sender or ApnsSender(settings)
    sent = 0
    try:
        for user in users:
            if user.id == admin_id:
                continue
            title, body = message(user)
            for device in devices:
                try:
                    result = await sender.send(
                        device.token, _topic(settings, device.app), title, body
                    )
                except Exception as error:  # noqa: BLE001 - un aviso nunca rompe los demás
                    logger.warning("admin_push_failed", error=str(error), device=device.id)
                    continue
                sent += result.status in ("enviada", "simulada")
                logger.info("push_admin", status=result.status, user=user.id, device=device.id)
    finally:
        if own_sender:
            await sender.close()
    return sent
