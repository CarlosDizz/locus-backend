"""Enlaces de Travelpayouts pedidos a su API al hacer clic (affiliates/travelpayouts.py)."""

from types import SimpleNamespace

import httpx
from pydantic import SecretStr

from locus_v2.affiliates import travelpayouts
from locus_v2.affiliates.programs import is_configured, program_for

PARTNER = "https://tp.media/r?campaign_id=137&marker=794924&p=4110&trs=590543&u=x"


def _settings(token: str | None = "tok") -> SimpleNamespace:
    return SimpleNamespace(
        travelpayouts_token=SecretStr(token) if token else None,
        travelpayouts_marker=794924, travelpayouts_trs=590543, affiliate_links={},
        getyourguide_partner_id="",
    )


def _api(monkeypatch, respuesta=None, error: Exception | None = None) -> list[dict]:
    pedidas: list[dict] = []

    class _Client:
        def __init__(self, **_) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_) -> None:
            return None

        async def post(self, url, headers, json):
            pedidas.append({"headers": headers, "json": json})
            if error is not None:
                raise error
            return httpx.Response(200, json=respuesta, request=httpx.Request("POST", url))

    monkeypatch.setattr(travelpayouts.httpx, "AsyncClient", _Client)
    travelpayouts._cache.clear()
    return pedidas


async def test_a_klook_link_becomes_ours_and_is_cached(monkeypatch) -> None:
    pedidas = _api(monkeypatch, {"result": {"links": [
        {"url": "https://www.klook.com/es/activity/1/", "code": "success",
         "partner_url": PARTNER},
    ]}})
    settings = _settings()

    assert await travelpayouts.partner_url(settings, "https://www.klook.com/es/activity/1/") \
        == PARTNER
    assert await travelpayouts.partner_url(settings, "https://www.klook.com/es/activity/1/") \
        == PARTNER
    assert len(pedidas) == 1
    assert pedidas[0]["headers"]["X-Access-Token"] == "tok"
    assert pedidas[0]["json"]["marker"] == 794924 and pedidas[0]["json"]["trs"] == 590543


async def test_links_of_other_programs_are_left_alone(monkeypatch) -> None:
    pedidas = _api(monkeypatch, {})

    for url in ("https://www.booking.com/hotel/x.html", "https://www.amazon.es/dp/1",
                "https://example.com/"):
        assert await travelpayouts.partner_url(_settings(), url) is None
    assert pedidas == []


async def test_without_token_or_when_the_api_fails_the_plain_link_is_used(monkeypatch) -> None:
    _api(monkeypatch, error=httpx.ConnectTimeout("lento"))
    assert await travelpayouts.partner_url(_settings(), "https://www.airalo.com/vietnam-esim") \
        is None
    assert await travelpayouts.partner_url(_settings(None), "https://www.airalo.com/x") is None

    _api(monkeypatch, {"result": {"links": [{"code": "trs is not subscribed for brand"}]}})
    assert await travelpayouts.partner_url(_settings(), "https://saily.com/") is None


def test_travelpayouts_programs_count_as_configured_only_with_a_token() -> None:
    assert program_for("https://www.welcomepickups.com/hanoi").code == "welcomepickups"
    assert is_configured(_settings(), "airalo")
    assert not is_configured(_settings(None), "airalo")
