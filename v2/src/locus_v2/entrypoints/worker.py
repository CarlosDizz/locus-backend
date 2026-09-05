import asyncio
import signal

import structlog
from redis.asyncio import Redis

from locus_v2.config import get_settings
from locus_v2.logging import configure_logging


async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = structlog.get_logger()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()
    logger.info("worker_ready", environment=settings.env)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())
