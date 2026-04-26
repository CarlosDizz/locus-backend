#!/bin/sh
set -eu

RUN_MIGRATIONS="${RUN_MIGRATIONS:-true}"
MIGRATION_MODE="${MIGRATION_MODE:-auto}"
MIGRATION_MAX_ATTEMPTS="${MIGRATION_MAX_ATTEMPTS:-20}"
MIGRATION_RETRY_DELAY="${MIGRATION_RETRY_DELAY:-3}"
MIGRATION_BLOCKING="${MIGRATION_BLOCKING:-false}"

get_alembic_revision() {
  command_output="$1"
  printf '%s\n' "$command_output" | awk '/^[0-9A-Za-z_]+/ {print $1; exit}'
}

migrations_pending() {
  current_output="$(alembic current 2>/dev/null || true)"
  head_output="$(alembic heads 2>/dev/null || true)"
  current_revision="$(get_alembic_revision "$current_output")"
  head_revision="$(get_alembic_revision "$head_output")"

  if [ -z "$head_revision" ]; then
    echo "Could not determine Alembic head revision. Running migrations to be safe."
    return 0
  fi

  if [ -z "$current_revision" ]; then
    echo "No current Alembic revision detected. Migrations are pending."
    return 0
  fi

  if [ "$current_revision" != "$head_revision" ]; then
    echo "Alembic revision $current_revision is behind head $head_revision."
    return 0
  fi

  echo "Database already at Alembic head $head_revision. Skipping migration step."
  return 1
}

if [ "$RUN_MIGRATIONS" = "true" ]; then
  if [ "$MIGRATION_BLOCKING" != "true" ]; then
    echo "Starting application without blocking on migrations (MIGRATION_BLOCKING=false)."
    (
      attempt=1
      while [ "$attempt" -le "$MIGRATION_MAX_ATTEMPTS" ]; do
        should_run_migrations="true"
        if [ "$MIGRATION_MODE" = "auto" ] && ! migrations_pending; then
          should_run_migrations="false"
        fi

        if [ "$should_run_migrations" = "false" ]; then
          exit 0
        fi

        echo "Running database migrations in background (attempt $attempt/$MIGRATION_MAX_ATTEMPTS, mode=$MIGRATION_MODE)..."
        if alembic upgrade head; then
          echo "Database migrations completed."
          exit 0
        fi

        if [ "$attempt" -eq "$MIGRATION_MAX_ATTEMPTS" ]; then
          echo "Database migrations failed after $MIGRATION_MAX_ATTEMPTS attempts."
          exit 1
        fi

        echo "Database not ready yet. Retrying in ${MIGRATION_RETRY_DELAY}s..."
        attempt=$((attempt + 1))
        sleep "$MIGRATION_RETRY_DELAY"
      done
    ) &
    exec "$@"
  fi

  attempt=1
  while [ "$attempt" -le "$MIGRATION_MAX_ATTEMPTS" ]; do
    should_run_migrations="true"
    if [ "$MIGRATION_MODE" = "auto" ] && ! migrations_pending; then
      should_run_migrations="false"
    fi

    if [ "$should_run_migrations" = "false" ]; then
      break
    fi

    echo "Running database migrations (attempt $attempt/$MIGRATION_MAX_ATTEMPTS, mode=$MIGRATION_MODE)..."
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

echo "Starting application process: $*"
exec "$@"
