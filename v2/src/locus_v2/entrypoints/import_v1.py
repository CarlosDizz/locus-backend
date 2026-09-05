import argparse
import asyncio
import json

from locus_v2.config import get_settings
from locus_v2.infrastructure.database.session import get_database
from locus_v2.migrations.legacy_v1 import LegacyV1Importer


async def execute(inspect_only: bool) -> None:
    settings = get_settings()
    if not settings.legacy_database_url:
        raise SystemExit("LOCUS_LEGACY_DATABASE_URL is required")
    async with get_database().sessions() as session:
        importer = LegacyV1Importer(
            settings.legacy_database_url,
            session,
            str(settings.admin_email),
        )
        if inspect_only:
            print(json.dumps(await importer.inspect(), indent=2, sort_keys=True))
            await importer.source.dispose()
            return
        run = await importer.run()
        print(json.dumps({"run_id": run.id, "status": run.status, **run.table_counts_json}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import useful V1 data into Locus V2")
    parser.add_argument("--inspect", action="store_true", help="Only list source row counts")
    args = parser.parse_args()
    asyncio.run(execute(args.inspect))
