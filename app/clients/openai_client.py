from __future__ import annotations

from typing import Any

import requests

from app.config import settings


class OpenAIClientError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(self) -> None:
        self.base_url = settings.openai_base_url.rstrip("/")

    def chat_model(self) -> str:
        return settings.openai_chat_model

    def realtime_model(self) -> str:
        return settings.openai_realtime_model

    def realtime_voice(self) -> str:
        return settings.openai_realtime_voice

    def is_configured(self) -> bool:
        return bool(settings.openai_api_key)

    def _headers(self) -> dict[str, str]:
        if not self.is_configured():
            raise OpenAIClientError("OPENAI_API_KEY no está configurada")
        return {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }

    def create_response(
        self,
        *,
        model: str,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        max_output_tokens: int = 700,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
            "tool_choice": tool_choice,
            "max_output_tokens": max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        response = requests.post(
            f"{self.base_url}/responses",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise OpenAIClientError(f"Responses API error {response.status_code}: {response.text}")
        return response.json()

    def create_realtime_client_secret(
        self,
        *,
        model: str,
        instructions: str,
        tools: list[dict[str, Any]],
        modalities: list[str],
        max_output_tokens: int = 700,
        voice: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "expires_after": {
                "anchor": "created_at",
                "seconds": settings.openai_realtime_secret_ttl_seconds,
            },
            "session": {
                "type": "realtime",
                "model": model,
                "instructions": instructions,
                "output_modalities": modalities,
                "max_output_tokens": max_output_tokens,
                "tool_choice": "auto",
                "tools": tools,
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "interrupt_response": True,
                        }
                    },
                    "output": {
                        "voice": voice or self.realtime_voice(),
                    },
                },
            },
        }

        response = requests.post(
            f"{self.base_url}/realtime/client_secrets",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise OpenAIClientError(f"Realtime client secret error {response.status_code}: {response.text}")
        return response.json()
