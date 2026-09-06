"""Shared Redis connection, used by the calls domain (see locus_v2.calls.store)."""

from functools import lru_cache

from redis.asyncio import Redis

from locus_v2.config import get_settings


@lru_cache
def get_redis() -> Redis:
    client: Redis = Redis.from_url(get_settings().redis_url, decode_responses=True)
    return client
