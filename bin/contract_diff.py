"""Put V1 and V2 side by side and compare the contract the Ionic app depends on.

Run it right before the cutover, with both stacks up:

    # Materialise frozen V1 next to this checkout.
    git worktree add ../locus-backend-v1 v1-production-before-v2-cutover-20260908

    # V1's .env points at a dead Railway host; the override pins its local DB.
    cd ../locus-backend-v1
    docker compose -f docker-compose.yml -f ../locus-backend-1/bin/v1-local-db.yml up -d
    docker compose -f docker-compose.yml -f ../locus-backend-1/bin/v1-local-db.yml exec api alembic upgrade head

    # V2
    cd ../locus-backend-1
    LOCUS_API_HOST_PORT=8200 ./bin/locus up

    python3 bin/contract_diff.py [token_v1] [token_v2]

Tokens are optional; without them only the public half runs. Both versions hash
their session token with plain SHA-256, so one can be minted straight into
`auth_tokens` (V1) or `user_sessions` (V2) for the comparison.

What matters for the cutover is not that both return the same *content* — the
two databases hold different rows and an AI reply is never identical twice —
but that they return the same *shape*: same status code, same JSON keys, same
types, same nesting. That is exactly what lets the app keep working when only
apiBaseUrl changes.

So every check reports three things: the status code from each side, the keys
each returned, and the difference between them. Anything listed under FALTA in
V2 is a field the app may read and would find missing.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx

V1 = "http://localhost:8000/api"
V2 = "http://localhost:8200/api"
V1_TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""
V2_TOKEN = sys.argv[2] if len(sys.argv) > 2 else ""


def shape(value: Any, prefix: str = "") -> set[str]:
    """Every key path in a payload, with the type of its leaf.

    A null is recorded as the key alone, with no type: the field is present and
    the app can read it, and whether this particular row happened to fill it is
    data, not contract. Without that, a ledger row that is a top-up on one side
    and a charge on the other reports six false differences.
    """
    out: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            out |= shape(item, path)
    elif isinstance(value, list):
        # Only the first element: a list of identical shapes says all it needs to.
        if value:
            out |= shape(value[0], f"{prefix}[]")
        else:
            out.add(f"{prefix}[] (vacio)")
    elif value is None:
        out.add(prefix)
    else:
        out.add(f"{prefix}: {type(value).__name__}")
    return out


def keys_only(paths: set[str]) -> set[str]:
    return {p.split(":")[0].strip() for p in paths}


async def probe(
    client: httpx.AsyncClient, name: str, method: str, path: str,
    body: dict | None = None, auth: bool = False,
) -> None:
    async def call(base: str, token: str) -> tuple[int, Any]:
        headers = {"Authorization": f"Bearer {token}"} if auth and token else {}
        try:
            response = await client.request(
                method, f"{base}{path}", json=body, headers=headers, timeout=45
            )
            try:
                return response.status_code, response.json()
            except json.JSONDecodeError:
                return response.status_code, response.text[:80]
        except Exception as error:  # noqa: BLE001 - a dead endpoint is a result
            return 0, f"{type(error).__name__}: {error}"

    (s1, b1), (s2, b2) = await asyncio.gather(call(V1, V1_TOKEN), call(V2, V2_TOKEN))
    same_status = s1 == s2
    k1, k2 = shape(b1), shape(b2)
    # Compare the key paths; a type that differs only because one side is null
    # is not a contract break.
    n1, n2 = keys_only(k1), keys_only(k2)
    missing = sorted(n1 - n2)
    extra = sorted(n2 - n1)
    retyped = sorted(
        p for p in (k1 - k2)
        if ":" in p and p.split(":")[0].strip() in n2
        and f"{p.split(':')[0].strip()}" not in {q.split(":")[0].strip() for q in (k2 - k1) if ":" not in q}
        and any(q.split(":")[0].strip() == p.split(":")[0].strip() and ":" in q for q in k2)
    )

    mark = "OK " if same_status and not missing else "!! "
    print(f"{mark}{name:<34} {method} {path}")
    print(f"     estado  V1={s1}  V2={s2}{'' if same_status else '   <-- DISTINTO'}")
    if missing:
        print(f"     FALTA en V2 ({len(missing)}): {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if extra:
        print(f"     extra en V2 ({len(extra)}): {', '.join(extra[:4])}"
              + (" ..." if len(extra) > 4 else ""))
    if retyped:
        print(f"     tipo distinto ({len(retyped)}): {', '.join(retyped[:4])}"
              + (" ..." if len(retyped) > 4 else ""))
    if not same_status and isinstance(b1, str):
        print(f"     V1 dijo: {str(b1)[:70]}")
    if not same_status and isinstance(b2, str):
        print(f"     V2 dijo: {str(b2)[:70]}")


async def main() -> None:
    async with httpx.AsyncClient() as client:
        print("=" * 78)
        print("CONTRATO PUBLICO — lo que la app llama sin estar autenticada")
        print("=" * 78)
        await probe(client, "version de la app", "GET", "/app/version")
        await probe(client, "tipos de POI", "GET", "/catalog/poi-types")
        await probe(client, "ciudades", "GET", "/catalog/cities")
        await probe(client, "POIs", "GET", "/catalog/pois?limit=3")
        await probe(client, "registro (desactivado)", "POST", "/auth/register",
                    {"email": "x@y.com", "password": "secreto123", "display_name": "X"})
        await probe(client, "login (desactivado)", "POST", "/auth/login",
                    {"email": "x@y.com", "password": "secreto123"})
        await probe(client, "google con token invalido", "POST", "/auth/google",
                    {"id_token": "no-es-un-token"})

        if not (V1_TOKEN and V2_TOKEN):
            print("\n(sin tokens: se omite la parte autenticada)")
            return
        print()
        print("=" * 78)
        print("CONTRATO AUTENTICADO — lo que la app llama con sesion")
        print("=" * 78)
        await probe(client, "quien soy", "GET", "/auth/me", auth=True)
        await probe(client, "monedero", "GET", "/billing/wallet", auth=True)
        await probe(client, "movimientos", "GET", "/billing/ledger?limit=3", auth=True)
        await probe(client, "consumo", "GET", "/billing/usage-events?limit=3", auth=True)


asyncio.run(main())
