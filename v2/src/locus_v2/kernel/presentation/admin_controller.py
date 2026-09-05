from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from fastapi import APIRouter

ServiceT = TypeVar("ServiceT")


class AdminController(Generic[ServiceT], ABC):
    """Base controller wiring; concrete controllers own routes and schemas."""

    prefix: str
    tags: list[str]

    def __init__(self) -> None:
        self.router = APIRouter(prefix=self.prefix, tags=self.tags)
        self.register_routes()

    @abstractmethod
    def register_routes(self) -> None:
        raise NotImplementedError
