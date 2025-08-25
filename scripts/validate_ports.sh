#!/usr/bin/env bash
set -euo pipefail
if git grep -nE '(^|[^0-9])8000([^0-9]|$)' -- . \
    | grep -v 'CHANGELOG\|MIGRATION\|scripts/validate_ports.sh' ; then
  echo "Found lingering :8000 references"; exit 1
fi
echo "Ports OK"
