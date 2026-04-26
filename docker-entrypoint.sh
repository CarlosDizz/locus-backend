#!/bin/sh
set -eu

HOST_VALUE="${HOST:-0.0.0.0}"
PORT_VALUE="${PORT:-8000}"

echo "Starting application process: $* --host ${HOST_VALUE} --port ${PORT_VALUE}"
exec "$@" --host "${HOST_VALUE}" --port "${PORT_VALUE}"
