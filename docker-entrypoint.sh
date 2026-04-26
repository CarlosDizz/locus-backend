#!/bin/sh
set -eu

RUN_MIGRATIONS="${RUN_MIGRATIONS:-true}"
MIGRATION_MAX_ATTEMPTS="${MIGRATION_MAX_ATTEMPTS:-20}"
MIGRATION_RETRY_DELAY="${MIGRATION_RETRY_DELAY:-3}"

if [ "$RUN_MIGRATIONS" = "true" ]; then
  attempt=1
  while [ "$attempt" -le "$MIGRATION_MAX_ATTEMPTS" ]; do
    echo "Running database migrations (attempt $attempt/$MIGRATION_MAX_ATTEMPTS)..."
    if alembic upgrade head; then
      echo "Database migrations completed."
      break
    fi

    if [ "$attempt" -eq "$MIGRATION_MAX_ATTEMPTS" ]; then
      echo "Database migrations failed after $MIGRATION_MAX_ATTEMPTS attempts."
      exit 1
    fi

    echo "Database not ready yet. Retrying in ${MIGRATION_RETRY_DELAY}s..."
    attempt=$((attempt + 1))
    sleep "$MIGRATION_RETRY_DELAY"
  done
fi

exec "$@"
