#!/usr/bin/env python3
"""Translate the proven V1/V2 settings into a private production dotenv file."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from urllib.parse import quote


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.startswith(('"', "'")) and not raw.endswith(raw[0]):
            parts = [raw]
            while index < len(lines):
                parts.append(lines[index])
                index += 1
                if parts[-1].rstrip().endswith(raw[0]):
                    break
            raw = "\n".join(parts)
        if len(raw) >= 2 and raw[0] == raw[-1] == "'":
            value = raw[1:-1]
        elif len(raw) >= 2 and raw[0] == raw[-1] == '"':
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw[1:-1]
        else:
            value = raw
        values[key] = value
    return values


def first(*candidates: str | None, default: str = "") -> str:
    return next((value for value in candidates if value), default)


def json_list(value: str) -> str:
    value = value.strip()
    if not value:
        return "[]"
    if value.startswith("["):
        json.loads(value)
        return value
    return json.dumps([item.strip() for item in value.split(",") if item.strip()])


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def build(legacy: dict[str, str], local: dict[str, str], existing: dict[str, str]) -> str:
    db_user = first(legacy.get("DB_USER"), default="locus")
    db_password = legacy["DB_PASSWORD"]
    encoded_user = quote(db_user, safe="")
    encoded_password = quote(db_password, safe="")
    jwt_secret = first(existing.get("LOCUS_JWT_SECRET"), default=secrets.token_hex(48))
    google_ids = json_list(
        first(local.get("LOCUS_GOOGLE_AUTH_CLIENT_IDS"), legacy.get("GOOGLE_AUTH_CLIENT_IDS"))
    )

    values = {
        "MYSQL_ROOT_PASSWORD": legacy["MYSQL_ROOT_PASSWORD"],
        "DB_NAME": first(legacy.get("DB_NAME"), default="locus"),
        "DB_USER": db_user,
        "DB_PASSWORD": db_password,
        "LOCUS_ENV": "production",
        "LOCUS_LOG_LEVEL": first(legacy.get("LOG_LEVEL"), default="INFO").upper(),
        "LOCUS_DATABASE_URL": (
            f"mysql+asyncmy://{encoded_user}:{encoded_password}@mysql:3306/locus_v2"
        ),
        "LOCUS_LEGACY_DATABASE_URL": (
            f"mysql+asyncmy://{encoded_user}:{encoded_password}@mysql:3306/locus"
        ),
        "LOCUS_REDIS_URL": "redis://valkey:6379/0",
        "LOCUS_JWT_SECRET": jwt_secret,
        "LOCUS_ADMIN_EMAIL": first(local.get("LOCUS_ADMIN_EMAIL"), default="dizz01@gmail.com"),
        "LOCUS_ALLOW_INSECURE_LOCAL_ADMIN": "false",
        "LOCUS_ADMIN_SESSION_COOKIE": "locus_admin_session",
        "LOCUS_GOOGLE_AUTH_CLIENT_IDS": google_ids,
        "LOCUS_CORS_ORIGINS": json.dumps(
            [
                "https://api.locusguide.es",
                "https://admin.locusguide.es",
                "capacitor://localhost",
                "ionic://localhost",
                "http://localhost",
                "https://localhost",
            ]
        ),
        "LOCUS_PUBLIC_API_BASE_URL": "https://api.locusguide.es/api",
        "LOCUS_OPENAI_API_KEY": first(
            local.get("LOCUS_OPENAI_API_KEY"), legacy.get("OPENAI_API_KEY")
        ),
        "LOCUS_GEMINI_API_KEY": local.get("LOCUS_GEMINI_API_KEY", ""),
        "LOCUS_MAPS_API_KEY": first(local.get("LOCUS_MAPS_API_KEY"), legacy.get("MAPS_API_KEY")),
        "LOCUS_GETYOURGUIDE_REFERRALS_ENABLED": first(
            legacy.get("GETYOURGUIDE_REFERRALS_ENABLED"), default="true"
        ),
        "LOCUS_GETYOURGUIDE_PARTNER_ID": first(
            local.get("LOCUS_GETYOURGUIDE_PARTNER_ID"), legacy.get("GETYOURGUIDE_PARTNER_ID")
        ),
        "LOCUS_GOOGLE_PLAY_SERVICE_ACCOUNT_JSON": legacy.get(
            "GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", ""
        ),
        "LOCUS_GOOGLE_PLAY_PACKAGE_NAME": first(
            legacy.get("GOOGLE_PLAY_PACKAGE_NAME"), default="com.carlos.locusia"
        ),
        "LOCUS_GOOGLE_PLAY_VERIFY_PURCHASES": first(
            legacy.get("GOOGLE_PLAY_VERIFY_PURCHASES"), default="true"
        ),
        "LOCUS_BILLING_USD_TO_EUR": first(
            local.get("LOCUS_BILLING_USD_TO_EUR"), default="0.87"
        ),
        "LOCUS_BILLING_MARGIN_MULTIPLIER": first(
            local.get("LOCUS_BILLING_MARGIN_MULTIPLIER"), default="2.20"
        ),
        "LOCUS_BILLING_MIN_REALTIME_CALL_CHARGE_CENTS": first(
            legacy.get("BILLING_MIN_REALTIME_CALL_CHARGE_CENTS"), default="3"
        ),
        "LOCUS_BILLING_MIN_RESERVE_CENTS": first(
            legacy.get("BILLING_MIN_RESERVE_CENTS"), default="25"
        ),
        "LOCUS_BILLING_SIGNUP_BONUS_CENTS": first(
            legacy.get("BILLING_SIGNUP_BONUS_CENTS"), default="200"
        ),
        "LOCUS_BILLING_MANUAL_TOPUPS_ENABLED": first(
            legacy.get("BILLING_MANUAL_TOPUPS_ENABLED"), default="true"
        ),
        "LOCUS_APP_ANDROID_LATEST_VERSION_CODE": first(
            legacy.get("APP_ANDROID_LATEST_VERSION_CODE"), default="10"
        ),
        "LOCUS_APP_ANDROID_UPDATE_URL": first(
            legacy.get("APP_ANDROID_UPDATE_URL"),
            default="https://play.google.com/store/apps/details?id=com.carlos.locusia",
        ),
        "LOCUS_APP_IOS_LATEST_BUILD": first(
            legacy.get("APP_IOS_LATEST_BUILD"), default="1"
        ),
        "LOCUS_APP_IOS_UPDATE_URL": legacy.get("APP_IOS_UPDATE_URL", ""),
    }

    raw_json_keys = {"LOCUS_GOOGLE_AUTH_CLIENT_IDS", "LOCUS_CORS_ORIGINS"}
    return "\n".join(
        f"{key}={value if key in raw_json_keys else quoted(value)}"
        for key, value in values.items()
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".env.production"))
    args = parser.parse_args()
    existing = read_dotenv(args.output) if args.output.exists() else {}
    content = build(read_dotenv(args.legacy), read_dotenv(args.local), existing)
    args.output.write_text(content)
    os.chmod(args.output, 0o600)
    print(f"Wrote {args.output} with {len(content.splitlines())} configured keys")


if __name__ == "__main__":
    main()
