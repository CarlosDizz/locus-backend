from collections.abc import Mapping, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.infrastructure.database.base import Base


class SQLAlchemyRepository[ModelT: Base]:
    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self, *, limit: int, offset: int) -> Sequence[ModelT]:
        result = await self.session.scalars(
            select(self.model)
            .order_by(self.model.__table__.c.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.all()

    async def count(self) -> int:
        return await self.session.scalar(select(func.count()).select_from(self.model)) or 0

    async def get(self, entity_id: int) -> ModelT | None:
        return await self.session.get(self.model, entity_id)

    async def add(self, values: Mapping[str, object]) -> ModelT:
        entity = self.model(**dict(values))
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def update(self, entity: ModelT, values: Mapping[str, object]) -> ModelT:
        for field, value in values.items():
            setattr(entity, field, value)
        await self.session.flush()
        return entity

    async def delete(self, entity: ModelT) -> None:
        await self.session.delete(entity)
        await self.session.flush()
