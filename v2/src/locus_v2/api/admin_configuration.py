from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.admin.application.configuration import (
    AdminConfigurationService,
    ConfigurationError,
    RoutingChange,
)
from locus_v2.api.admin_auth import require_admin
from locus_v2.identity.models import User
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(prefix="/admin/v2/configuration", tags=["admin-configuration"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[User, Depends(require_admin)]


class ModelStateRequest(BaseModel):
    enabled: bool
    selectable: bool


class PromptVersionRequest(BaseModel):
    content: str = Field(min_length=20)


class RoutingRequest(BaseModel):
    primary_model_id: int
    fallback_model_id: int | None = None
    prompt_version_id: int


def _bad_request(error: ConfigurationError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))


@router.get("")
async def configuration(session: SessionDep, admin: AdminDep) -> dict[str, Any]:
    return await AdminConfigurationService(session, admin).snapshot()


@router.patch("/models/{model_id}")
async def set_model_state(
    model_id: int, payload: ModelStateRequest, session: SessionDep, admin: AdminDep
) -> dict:
    try:
        return await AdminConfigurationService(session, admin).set_model_state(
            model_id, payload.enabled, payload.selectable
        )
    except ConfigurationError as error:
        raise _bad_request(error) from error


@router.post("/prompts/{definition_id}/versions", status_code=status.HTTP_201_CREATED)
async def create_prompt_version(
    definition_id: int, payload: PromptVersionRequest, session: SessionDep, admin: AdminDep
) -> dict:
    try:
        return await AdminConfigurationService(session, admin).create_prompt_version(
            definition_id, payload.content
        )
    except ConfigurationError as error:
        raise _bad_request(error) from error


@router.post("/prompt-versions/{version_id}/publish")
async def publish_prompt_version(
    version_id: int, session: SessionDep, admin: AdminDep
) -> dict:
    try:
        return await AdminConfigurationService(session, admin).publish_prompt_version(version_id)
    except ConfigurationError as error:
        raise _bad_request(error) from error


@router.put("/routing-profiles/{profile_id}")
async def change_routing(
    profile_id: int, payload: RoutingRequest, session: SessionDep, admin: AdminDep
) -> dict:
    try:
        return await AdminConfigurationService(session, admin).change_routing(
            profile_id,
            RoutingChange(
                primary_model_id=payload.primary_model_id,
                fallback_model_id=payload.fallback_model_id,
                prompt_version_id=payload.prompt_version_id,
            ),
        )
    except ConfigurationError as error:
        raise _bad_request(error) from error
