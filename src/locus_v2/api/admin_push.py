"""Avisos de prueba desde el panel (portado de BACUL).

Sirve para comprobar la cadena entera (clave de APNs, móvil registrado, permiso
concedido) sin esperar a que alguien cambie un viaje compartido. Normalmente lo
lanza el administrador a su propio móvil.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.admin_auth import require_admin
from locus_v2.config import Settings, get_settings
from locus_v2.identity.models import User
from locus_v2.infrastructure.database.session import get_session
from locus_v2.notifications.apns import ApnsSender
from locus_v2.notifications.models import PushDevice
from locus_v2.notifications.service import _topic
from locus_v2.shared.clock import UtcDatetime, utc_now

router = APIRouter(prefix="/admin/v2/avisos", tags=["admin-avisos"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
AdminDep = Annotated[User, Depends(require_admin)]


class Recipient(BaseModel):
    id: int
    display_name: str
    email: str
    devices: int
    apps: list[str]
    last_seen_at: UtcDatetime | None


class TestNotice(BaseModel):
    user_ids: list[int] = Field(min_length=1, max_length=50)
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=300)
    # Opcional: el viaje que se abre al tocar el aviso.
    trip_id: str = ""


class DeviceResult(BaseModel):
    user_id: int
    display_name: str
    device: str
    status: str  # enviada | simulada | fallida | caducado | sin_movil | android
    detail: str


class TestResponse(BaseModel):
    apns: str  # produccion | sandbox | sin_clave
    results: list[DeviceResult]


@router.get("/destinatarios", response_model=list[Recipient])
async def recipients(_: AdminDep, session: SessionDep) -> list[Recipient]:
    """Quien tiene algún móvil activo para avisos, el más reciente primero."""
    rows = (
        await session.execute(
            select(
                User,
                func.count(PushDevice.id),
                func.group_concat(func.distinct(PushDevice.app)),
                func.max(PushDevice.last_seen_at),
            )
            .join(PushDevice, PushDevice.user_id == User.id)
            .where(PushDevice.disabled_at.is_(None))
            .group_by(User.id)
            .order_by(func.max(PushDevice.last_seen_at).desc())
            .limit(300)
        )
    ).all()
    return [
        Recipient(
            id=user.id,
            display_name=user.preferred_name or user.display_name,
            email=user.email,
            devices=count,
            apps=sorted(set((apps or "").split(","))),
            last_seen_at=last_seen,
        )
        for user, count, apps, last_seen in rows
    ]


@router.post("/prueba", response_model=TestResponse)
async def send_test(
    payload: TestNotice, _: AdminDep, session: SessionDep, settings: SettingsDep
) -> TestResponse:
    users = (await session.scalars(select(User).where(User.id.in_(payload.user_ids)))).all()
    if not users:
        raise HTTPException(status_code=404, detail="No encuentro a esas personas")

    sender = ApnsSender(settings)
    now: datetime = utc_now()
    results: list[DeviceResult] = []
    data = {"trip_id": payload.trip_id} if payload.trip_id else None
    try:
        for user in users:
            name = user.preferred_name or user.display_name
            devices = (
                await session.scalars(
                    select(PushDevice).where(
                        PushDevice.user_id == user.id, PushDevice.disabled_at.is_(None)
                    )
                )
            ).all()
            if not devices:
                results.append(DeviceResult(
                    user_id=user.id, display_name=name, device="—", status="sin_movil",
                    detail="No tiene ningún móvil registrado para avisos",
                ))
                continue
            for device in devices:
                etiqueta = f"{device.app} · {device.platform} …{device.token[-6:]}"
                if device.platform != "ios":
                    results.append(DeviceResult(
                        user_id=user.id, display_name=name, device=etiqueta, status="android",
                        detail="Android aún no recibe avisos (falta FCM)",
                    ))
                    continue
                try:
                    result = await sender.send(
                        device.token, _topic(settings, device.app), payload.title,
                        payload.body, data,
                    )
                    status, detail = result.status, result.detail
                except Exception as exc:  # noqa: BLE001 - red caída, APNs sin responder…
                    status, detail = "fallida", f"{type(exc).__name__}: {exc}"[:300]
                if status == "caducado":
                    device.disabled_at = now
                results.append(DeviceResult(
                    user_id=user.id, display_name=name, device=etiqueta, status=status,
                    detail=detail,
                ))
        await session.commit()
    finally:
        await sender.close()

    if not sender.configured:
        apns = "sin_clave"
    else:
        apns = "sandbox" if settings.apns_sandbox else "produccion"
    return TestResponse(apns=apns, results=results)
