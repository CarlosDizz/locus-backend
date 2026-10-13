"""Valoraciones de llamadas: guardarlas, moderarlas y convertirlas en datos del mapa."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.calls.models import Room
from locus_v2.catalog.models import Poi
from locus_v2.config import Settings
from locus_v2.identity.models import User
from locus_v2.reviews.models import CallReview, CommentStatus, ContentReport, UserBlock
from locus_v2.shared.clock import utc_now
from locus_v2.shared.mobile_ids import mobile_id, mobile_id_clause
from locus_v2.shared.openai_client import build_openai_client
from locus_v2.voice.models import VoiceSession

logger = structlog.get_logger()

# Las etiquetas rápidas sobre el guía. Cerradas a propósito: así se pueden
# contar y comparar entre llamadas, que es para lo que sirven.
GUIDE_TAGS = frozenset({"learned", "too_long", "wrong_fact", "cut_off", "misheard"})


class ReviewError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class ReviewInput(BaseModel):
    place_vote: bool | None = None
    place_comment: str | None = Field(default=None, max_length=1000)
    guide_stars: int | None = Field(default=None, ge=1, le=5)
    guide_tags: list[str] = Field(default_factory=list, max_length=len(GUIDE_TAGS))
    guide_comment: str | None = Field(default=None, max_length=2000)

    @field_validator("place_comment", "guide_comment")
    @classmethod
    def _limpiar(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        return value or None

    @field_validator("guide_tags")
    @classmethod
    def _etiquetas(cls, value: list[str]) -> list[str]:
        desconocidas = set(value) - GUIDE_TAGS
        if desconocidas:
            raise ValueError(f"Etiquetas desconocidas: {sorted(desconocidas)}")
        return sorted(set(value))


class Popularity(BaseModel):
    visits: int
    like_ratio: float | None
    tier: int


class PublicReview(BaseModel):
    # Para denunciarlo o bloquear a quien lo escribió desde la app.
    id: int = 0
    author_id: int = 0
    author: str
    created_at: datetime
    liked: bool | None
    text: str


class PoiReviewSummary(BaseModel):
    visits: int
    like_ratio: float | None
    votes: int
    total: int
    latest: list[PublicReview]


@dataclass(frozen=True)
class PoiStats:
    visits: int
    likes: int
    votes: int

    @property
    def ratio(self) -> float | None:
        return self.likes / self.votes if self.votes else None


def popularity_tiers(stats: dict[int, PoiStats]) -> dict[int, int]:
    """El escalón del marcador de cada sitio, relativo a los demás de la lista.

    0 · gusta a menos de la mitad (con al menos 3 votos, para que un mal día de
        alguien no hunda un sitio).
    1 · normal, y todo lo que aún no tiene visitas.
    2 · el 25 % siguiente que más puntúa, con al menos 2 visitas.
    3 · el 10 % que más puntúa, con al menos 3 visitas.

    La puntuación es visitas por ratio de «me gusta»; sin votos se supone 0,8,
    que es lo normal en un sitio que la gente elige visitar.
    """
    tiers = {poi_id: 1 for poi_id in stats}
    if not stats:
        return tiers
    puntuados: list[tuple[float, int]] = []
    for poi_id, stat in stats.items():
        ratio = stat.ratio
        if stat.votes >= 3 and ratio is not None and ratio < 0.5:
            tiers[poi_id] = 0
            continue
        if stat.visits:
            puntuados.append((stat.visits * (ratio if ratio is not None else 0.8), poi_id))
    puntuados.sort(reverse=True)
    top = max(1, math.ceil(len(stats) * 0.10))
    siguiente = max(1, math.ceil(len(stats) * 0.25))
    for posicion, (_, poi_id) in enumerate(puntuados):
        visitas = stats[poi_id].visits
        if posicion < top and visitas >= 3:
            tiers[poi_id] = 3
        elif posicion < top + siguiente and visitas >= 2:
            tiers[poi_id] = 2
    return tiers


class CommentModerator:
    """Decide si un comentario público se puede enseñar a otros usuarios.

    Usa el moderador de OpenAI, que es gratuito. Si no responde, el comentario
    se queda pendiente: no se publica sin haberlo mirado.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def status_for(self, text: str) -> str:
        if self.settings.openai_api_key is None:
            return CommentStatus.PENDING
        client = build_openai_client(self.settings)
        try:
            result = await client.moderations.create(
                model="omni-moderation-latest", input=text
            )
            flagged = bool(result.results and result.results[0].flagged)
        except Exception:  # noqa: BLE001 - sin moderación no se publica, pero no se pierde
            logger.warning("review_moderation_failed")
            return CommentStatus.PENDING
        finally:
            await client.close()
        return CommentStatus.REJECTED if flagged else CommentStatus.PUBLISHED


class ReviewService:
    def __init__(
        self, session: AsyncSession, moderator: CommentModerator | None = None
    ) -> None:
        self.session = session
        self.moderator = moderator

    async def submit(self, user: User, room: Room, payload: ReviewInput) -> CallReview:
        if str(user.id) not in room.members and str(user.id) not in room.participants:
            raise ReviewError("No estuviste en esta llamada", 403)
        if (
            payload.place_vote is None
            and payload.place_comment is None
            and payload.guide_stars is None
            and not payload.guide_tags
            and payload.guide_comment is None
        ):
            raise ReviewError("La valoración está vacía")
        poi = await self.session.scalar(select(Poi).where(Poi.public_id == room.poi_public_id))
        if poi is None:
            raise ReviewError("El sitio de esta llamada ya no existe", 404)

        review = await self.session.scalar(
            select(CallReview).where(
                CallReview.call_id == room.call_id, CallReview.user_id == user.id
            )
        )
        if review is None:
            review = CallReview(call_id=room.call_id, user_id=user.id, poi_id=poi.id)
            self.session.add(review)
        comentario_nuevo = payload.place_comment != review.place_comment
        review.place_vote = payload.place_vote
        review.place_comment = payload.place_comment
        review.guide_stars = payload.guide_stars
        review.guide_tags = payload.guide_tags
        review.guide_comment = payload.guide_comment
        if payload.place_comment is None:
            review.place_comment_status = None
        elif comentario_nuevo or review.place_comment_status is None:
            review.place_comment_status = (
                await self.moderator.status_for(payload.place_comment)
                if self.moderator is not None
                else CommentStatus.PENDING
            )
            review.moderated_at = utc_now()
        await self.session.commit()
        return review

    async def stats_for(self, pois: list[Poi]) -> dict[int, PoiStats]:
        if not pois:
            return {}
        por_public_id = {poi.public_id: poi.id for poi in pois}
        call_id = func.json_unquote(
            func.json_extract(VoiceSession.config_snapshot_json, "$.call_id")
        )
        visitas_rows = await self.session.execute(
            select(VoiceSession.context_public_id, func.count(func.distinct(call_id)))
            .where(
                VoiceSession.context_type == "poi",
                VoiceSession.context_public_id.in_(list(por_public_id)),
            )
            .group_by(VoiceSession.context_public_id)
        )
        visitas = {por_public_id[public_id]: int(n) for public_id, n in visitas_rows}
        votos_rows = await self.session.execute(
            select(
                CallReview.poi_id,
                func.sum(case((CallReview.place_vote.is_(True), 1), else_=0)),
                func.count(CallReview.place_vote),
            )
            .where(CallReview.poi_id.in_(list(por_public_id.values())))
            .group_by(CallReview.poi_id)
        )
        votos = {poi_id: (int(likes or 0), int(total)) for poi_id, likes, total in votos_rows}
        return {
            poi.id: PoiStats(
                visits=visitas.get(poi.id, 0),
                likes=votos.get(poi.id, (0, 0))[0],
                votes=votos.get(poi.id, (0, 0))[1],
            )
            for poi in pois
        }

    async def popularity_for(self, pois: list[Poi]) -> dict[int, Popularity]:
        stats = await self.stats_for(pois)
        tiers = popularity_tiers(stats)
        return {
            poi_id: Popularity(
                visits=stat.visits,
                like_ratio=round(stat.ratio, 3) if stat.ratio is not None else None,
                tier=tiers[poi_id],
            )
            for poi_id, stat in stats.items()
        }

    async def poi_summary(
        self, poi_mobile_id: int, limit: int = 3, viewer: User | None = None
    ) -> PoiReviewSummary:
        poi = await self.session.scalar(select(Poi).where(mobile_id_clause(Poi, poi_mobile_id)))
        if poi is None:
            raise ReviewError("POI no encontrado", 404)
        stat = (await self.stats_for([poi]))[poi.id]
        publicados = and_(
            CallReview.poi_id == poi.id,
            CallReview.place_comment_status == CommentStatus.PUBLISHED,
        )
        if viewer is not None:
            # Lo de quien ha bloqueado no se le enseña a quien bloquea.
            bloqueados = select(UserBlock.blocked_user_id).where(UserBlock.user_id == viewer.id)
            publicados = and_(publicados, CallReview.user_id.not_in(bloqueados))
        total = int(
            await self.session.scalar(select(func.count(CallReview.id)).where(publicados)) or 0
        )
        rows = await self.session.execute(
            select(CallReview, User.display_name, User.legacy_v1_id)
            .join(User, User.id == CallReview.user_id)
            .where(publicados)
            .order_by(CallReview.created_at.desc(), CallReview.id.desc())
            .limit(limit)
        )
        return PoiReviewSummary(
            visits=stat.visits,
            like_ratio=round(stat.ratio, 3) if stat.ratio is not None else None,
            votes=stat.votes,
            total=total,
            latest=[
                PublicReview(
                    id=review.id,
                    author_id=mobile_id(SimpleNamespace(id=review.user_id, legacy_v1_id=legacy)),
                    # Solo el nombre de pila: es un comentario público.
                    author=(nombre or "Viajero").split()[0],
                    # La base guarda UTC sin zona; sin marcarla, el móvil la lee
                    # como hora local y en España salía «hace 2 horas».
                    created_at=review.created_at.replace(tzinfo=UTC),
                    liked=review.place_vote,
                    text=review.place_comment or "",
                )
                for review, nombre, legacy in rows
            ],
        )

    async def admin_list(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self.session.execute(
            select(CallReview, User.email, User.display_name, Poi.name)
            .join(User, User.id == CallReview.user_id)
            .join(Poi, Poi.id == CallReview.poi_id)
            .order_by(CallReview.created_at.desc(), CallReview.id.desc())
            .limit(limit)
        )
        return [
            {
                "id": review.id,
                "call_id": review.call_id,
                "created_at": review.created_at.replace(tzinfo=UTC),
                "user_email": email,
                "user_name": nombre,
                "poi_name": poi_name,
                "place_vote": review.place_vote,
                "place_comment": review.place_comment,
                "place_comment_status": review.place_comment_status,
                "guide_stars": review.guide_stars,
                "guide_tags": review.guide_tags,
                "guide_comment": review.guide_comment,
            }
            for review, email, nombre, poi_name in rows
        ]


REPORT_KINDS = frozenset({"review", "call_image"})


class ReportInput(BaseModel):
    kind: str
    target_id: str = Field(min_length=1, max_length=120)
    reason: str = Field(default="", max_length=500)


class ModerationService:
    """Denuncias y bloqueos: lo que Apple exige para el contenido de los usuarios."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def report(self, reporter: User, payload: ReportInput) -> ContentReport:
        if payload.kind not in REPORT_KINDS:
            raise ReviewError("No se puede denunciar esto")
        if payload.kind == "review":
            review = await self.session.get(CallReview, int(payload.target_id)) if (
                payload.target_id.isdigit()
            ) else None
            if review is None:
                raise ReviewError("Ese comentario ya no existe", 404)
            # Se oculta en cuanto alguien lo denuncia, hasta que lo revisemos.
            if review.place_comment_status == CommentStatus.PUBLISHED:
                review.place_comment_status = CommentStatus.REPORTED
        report = ContentReport(
            reporter_id=reporter.id,
            kind=payload.kind,
            target=payload.target_id,
            reason=payload.reason.strip(),
        )
        self.session.add(report)
        await self.session.commit()
        logger.info("content_reported", kind=payload.kind, target=payload.target_id)
        return report

    async def block(self, user: User, blocked_mobile_id: int) -> None:
        bloqueado = await self.session.scalar(
            select(User).where(mobile_id_clause(User, blocked_mobile_id))
        )
        if bloqueado is None:
            raise ReviewError("Usuario no encontrado", 404)
        if bloqueado.id == user.id:
            raise ReviewError("No puedes bloquearte a ti mismo")
        ya = await self.session.scalar(
            select(UserBlock).where(
                UserBlock.user_id == user.id, UserBlock.blocked_user_id == bloqueado.id
            )
        )
        if ya is None:
            self.session.add(UserBlock(user_id=user.id, blocked_user_id=bloqueado.id))
            await self.session.commit()

    async def resolve_review(self, review_id: int, publish: bool) -> None:
        """Lo que decide el administrador tras una denuncia: vuelve a publicarse o se retira."""
        review = await self.session.get(CallReview, review_id)
        if review is None:
            raise ReviewError("Comentario no encontrado", 404)
        review.place_comment_status = (
            CommentStatus.PUBLISHED if publish else CommentStatus.REJECTED
        )
        abiertas = await self.session.scalars(
            select(ContentReport).where(
                ContentReport.kind == "review",
                ContentReport.target == str(review_id),
                ContentReport.resolved_at.is_(None),
            )
        )
        for report in abiertas:
            report.resolved_at = utc_now()
        await self.session.commit()

    async def open_reports(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self.session.execute(
            select(ContentReport, User.email)
            .join(User, User.id == ContentReport.reporter_id)
            .where(ContentReport.resolved_at.is_(None))
            .order_by(ContentReport.created_at.desc())
            .limit(limit)
        )
        return [
            {
                "id": report.id,
                "kind": report.kind,
                "target": report.target,
                "reason": report.reason,
                "reporter_email": email,
                "created_at": report.created_at.replace(tzinfo=UTC),
            }
            for report, email in rows
        ]
