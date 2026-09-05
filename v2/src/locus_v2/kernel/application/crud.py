from collections.abc import Mapping, Sequence
from typing import Generic, Protocol, TypeVar

EntityT = TypeVar("EntityT")


class ReadRepository(Protocol[EntityT]):
    async def list(self, *, limit: int, offset: int) -> Sequence[EntityT]: ...

    async def count(self) -> int: ...

    async def get(self, entity_id: int) -> EntityT | None: ...


class WriteRepository(ReadRepository[EntityT], Protocol[EntityT]):
    async def add(self, values: Mapping[str, object]) -> EntityT: ...

    async def update(self, entity: EntityT, values: Mapping[str, object]) -> EntityT: ...

    async def delete(self, entity: EntityT) -> None: ...


class CrudService(Generic[EntityT]):
    """Shared orchestration only; domain rules belong in subclasses."""

    def __init__(self, repository: WriteRepository[EntityT]) -> None:
        self.repository = repository

    async def list(self, *, limit: int = 100, offset: int = 0) -> Sequence[EntityT]:
        return await self.repository.list(limit=min(limit, 500), offset=max(offset, 0))

    async def count(self) -> int:
        return await self.repository.count()

    async def require(self, entity_id: int) -> EntityT:
        entity = await self.repository.get(entity_id)
        if entity is None:
            raise LookupError(f"Entity {entity_id} was not found")
        return entity

    async def create(self, values: Mapping[str, object]) -> EntityT:
        return await self.repository.add(values)

    async def change(self, entity_id: int, values: Mapping[str, object]) -> EntityT:
        return await self.repository.update(await self.require(entity_id), values)

    async def remove(self, entity_id: int) -> None:
        await self.repository.delete(await self.require(entity_id))
