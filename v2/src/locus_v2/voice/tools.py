import asyncio
from time import perf_counter

import structlog
from openai import AsyncOpenAI

from locus_v2.affiliates.service import ReferralService
from locus_v2.config import Settings
from locus_v2.shared.openai_usage import ToolUsage, usage_from_openai_response

logger = structlog.get_logger()


class VoiceToolError(RuntimeError):
    pass


class VoiceToolDispatcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Set by execute() on every call - real cost incurred by the plain OpenAI calls
        # below, outside the LiveProvider abstraction whose usage voice/gateway.py's
        # _persist_usage() already captures. Read this right after execute() returns
        # and bill it the same way, or the tool call is free money out the door.
        self.last_usage: ToolUsage | None = None

    async def execute(
        self,
        handler_code: str,
        arguments: dict,
        context: dict,
        locale: str,
    ) -> dict:
        self.last_usage = None
        started_at = perf_counter()
        logger.info(
            "voice_tool_started",
            handler=handler_code,
            locale=locale,
            model=self.settings.tool_model,
        )
        try:
            async with asyncio.timeout(self.settings.tool_timeout_seconds):
                if handler_code == "catalog.document_poi":
                    result = await self._research(arguments, context, locale)
                elif handler_code == "catalog.plan_poi_visit":
                    result = await self._plan_visit(arguments, context, locale)
                elif handler_code == "affiliates.find_activities":
                    result = await self._find_activities(arguments, context)
                else:
                    raise VoiceToolError(
                        f"Unsupported voice tool handler: {handler_code}"
                    )
        except TimeoutError as error:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
            logger.warning(
                "voice_tool_timed_out",
                handler=handler_code,
                model=self.settings.tool_model,
                elapsed_ms=elapsed_ms,
            )
            raise VoiceToolError(
                f"Voice tool {handler_code} timed out after "
                f"{self.settings.tool_timeout_seconds:g} seconds"
            ) from error
        except Exception:
            logger.exception(
                "voice_tool_failed",
                handler=handler_code,
                model=self.settings.tool_model,
                elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
            )
            raise
        logger.info(
            "voice_tool_completed",
            handler=handler_code,
            model=self.settings.tool_model,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
            answer_chars=len(result.get("answer", "")),
            billable=bool(self.last_usage and self.last_usage.billable),
        )
        return result

    def _accumulate_usage(self, usage: ToolUsage) -> None:
        self.last_usage = usage if self.last_usage is None else self.last_usage + usage

    async def _research(self, arguments: dict, context: dict, locale: str) -> dict:
        question = arguments.get("question") or "Historia y detalles relevantes del lugar"
        focus = arguments.get("focus") or ""
        prompt = f"""Documenta este lugar para un guía turístico en directo.
Lugar: {context.get('name', '')}
Ciudad: {context.get('city_name', '')}
Descripción disponible: {context.get('description', '')}
Wikidata: {context.get('wikidata_id', '')}
Wikipedia: {context.get('wikipedia_title', '')}
Pregunta: {question}
Enfoque: {focus}
Idioma de respuesta: {locale}

Devuelve hechos concretos, fechas, protagonistas, contexto y detalles observables. No escribas
una introducción ni menciones limitaciones. Si un dato no es fiable, omítelo."""
        answer = await self._ask_model(prompt)
        return {
            "kind": "poi_research",
            "answer": answer,
            "model": self.settings.tool_model,
        }

    async def _plan_visit(self, arguments: dict, context: dict, locale: str) -> dict:
        mode = arguments.get("mode", "scene")
        intent = arguments.get("user_intent", "Conocer bien el lugar")
        prompt = f"""Diseña una experiencia guiada para {context.get('name', '')},
en {context.get('city_name', '')}. Modo: {mode}. Intención: {intent}. Idioma: {locale}.
Si es scene, crea una secuencia natural de observación y relato. Si es stops, devuelve paradas
ordenadas con qué mirar y qué contar. Sé concreto y útil para que otro modelo lo narre en vivo."""
        answer = await self._ask_model(prompt)
        return {
            "kind": "poi_visit_plan",
            "mode": mode,
            "answer": answer,
            "model": self.settings.tool_model,
        }

    async def _find_activities(self, arguments: dict, context: dict) -> dict:
        referrals = ReferralService(self.settings)
        result = await referrals.activity_referrals(
            session_id="",
            query=arguments.get("query", ""),
            poi_name=arguments.get("poi_name") or context.get("name", ""),
            city_name=arguments.get("city_name") or context.get("city_name", ""),
            intent=arguments.get("intent", ""),
        )
        usage = result.pop("_usage", None)
        if usage is not None:
            self._accumulate_usage(usage)
        result.setdefault("answer", "")
        return result

    async def _ask_model(self, prompt: str) -> str:
        if self.settings.openai_api_key is None:
            raise VoiceToolError("The OpenAI key required by voice tools is not configured")
        client = AsyncOpenAI(api_key=self.settings.openai_api_key.get_secret_value())
        try:
            response = await client.responses.create(
                model=self.settings.tool_model,
                input=prompt,
                # tool_model (gpt-5-mini) is a reasoning model that spends hidden reasoning
                # tokens before writing any visible answer - confirmed live (2026-09-06):
                # document_poi burned all 1792 of its 1800-token budget on reasoning and
                # returned an empty answer, a billed call with nothing to show for it. The
                # same failure mode as affiliates/service.py's web_search call (220 -> 1500).
                max_output_tokens=4000,
            )
            self._accumulate_usage(usage_from_openai_response(response))
            return response.output_text.strip()
        finally:
            await client.close()
