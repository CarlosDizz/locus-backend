from abc import ABC, abstractmethod
from enum import Enum

from fastapi import APIRouter


class AdminController[ServiceT](ABC):
    """Base controller wiring; concrete controllers own routes and schemas."""

    prefix: str
    tags: list[str | Enum]

    def __init__(self) -> None:
        self.router = APIRouter(prefix=self.prefix, tags=self.tags)
        self.register_routes()

    @abstractmethod
    def register_routes(self) -> None:
        raise NotImplementedError
