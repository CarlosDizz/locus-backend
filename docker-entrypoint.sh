#!/bin/sh
set -eu

echo "Starting application process: $*"
exec "$@"
