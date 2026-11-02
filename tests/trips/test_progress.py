"""Los pasos que se enseñan mientras el planificador trabaja (trips/progress.py)."""

from locus_v2.trips import progress


class _Valkey:
    def __init__(self) -> None:
        self.datos: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.datos.get(key)

    async def set(self, key: str, value: str, ex: int) -> None:
        self.datos[key] = value


class _Caida:
    async def get(self, key: str) -> str | None:
        raise ConnectionError("sin valkey")

    async def set(self, key: str, value: str, ex: int) -> None:
        raise ConnectionError("sin valkey")


def test_each_tool_becomes_a_step_with_its_detail() -> None:
    assert progress.step_for_tool(
        "buscar_transporte", {"tipo": "vuelo", "consulta": "vuelo Tokio Okinawa"}
    ) == ("flights", "vuelo Tokio Okinawa")
    assert progress.step_for_tool("buscar_alojamiento", {"ciudad": "Hué"}) == ("hotels", "Hué")
    assert progress.step_for_tool("buscar_alojamiento", {}, "Roma") == ("hotels", "Roma")
    assert progress.step_for_tool("buscar_esim", {"pais": "Japan"}) == ("esim", "Japan")
    assert progress.step_for_tool("set_trip_details", {}) is None


async def test_steps_are_kept_in_order_without_repeating(monkeypatch) -> None:
    valkey = _Valkey()
    monkeypatch.setattr(progress, "get_redis", lambda: valkey)

    await progress.start("t1")
    await progress.step("t1", "thinking")
    await progress.step("t1", "thinking")
    await progress.step("t1", "flights", "Tokio → Okinawa")
    await progress.finish("t1")

    estado = await progress.read("t1")
    assert estado["running"] is False
    assert [s["kind"] for s in estado["steps"]] == ["thinking", "flights"]


async def test_without_valkey_nothing_breaks(monkeypatch) -> None:
    monkeypatch.setattr(progress, "get_redis", lambda: _Caida())

    await progress.start("t1")
    await progress.step("t1", "hotels", "Hué")
    assert await progress.read("t1") == {"running": False, "steps": []}
