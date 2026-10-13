"""Borrar la cuenta desde la app.

Lo exigen Apple (desde 2022, para cualquier app que permita crear cuenta) y
Google Play. Se borra todo lo personal y se anonimiza el usuario en vez de
borrar la fila: los apuntes contables (recargas, consumos, movimientos del
monedero) cuelgan de él y hay que conservarlos por ley; sin email, nombre ni
vínculo con Google ya no identifican a nadie.

Si la persona vuelve a entrar con la misma cuenta de Google, empieza de cero:
el vínculo (provider_subject) se ha borrado y el email anonimizado no choca.
"""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.identity.models import User, UserSession, UserStatus
from locus_v2.reviews.models import CallReview
from locus_v2.sessions.models import MapSession
from locus_v2.trips.models import Trip
from locus_v2.voice.models import VoiceSession, VoiceTurn


def anonymize(user: User) -> None:
    """Quita del usuario todo lo que lo identifica y lo deja sin poder entrar."""
    user.email = f"deleted-{user.public_id}@deleted.locusguide.es"
    user.display_name = "Cuenta eliminada"
    user.preferred_name = ""
    user.avatar_url = None
    user.provider_subject = None
    user.apple_subject = None
    user.profile_context = ""
    user.status = UserStatus.DELETED


async def delete_account(session: AsyncSession, user: User) -> None:
    # Lo que contó en las llamadas: las transcripciones de sus sesiones de voz.
    sesiones_voz = select(VoiceSession.id).where(VoiceSession.user_id == user.id)
    await session.execute(delete(VoiceTurn).where(VoiceTurn.voice_session_id.in_(sesiones_voz)))
    await session.execute(delete(Trip).where(Trip.user_id == user.id))
    await session.execute(delete(CallReview).where(CallReview.user_id == user.id))
    await session.execute(delete(MapSession).where(MapSession.user_id == user.id))
    # Las sesiones de login, todas: el móvil desde el que se borra y cualquier otro.
    await session.execute(delete(UserSession).where(UserSession.user_id == user.id))
    anonymize(user)
    await session.commit()
