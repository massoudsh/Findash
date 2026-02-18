#!/usr/bin/env bash
# Smoke test for the core Docker stack (API, Redis, Postgres).
# Run after: docker compose -f docker-compose-core.yml up -d
# Usage: ./scripts/healthcheck-core.sh

set -e

API_URL="${API_URL:-http://localhost:8011}"
REDIS_PORT="${REDIS_HOST_PORT:-6380}"
PG_PORT="${PG_PORT:-5433}"
FAILED=0

echo "Checking core stack..."
echo "  API_URL=${API_URL}"
echo "  Redis port=${REDIS_PORT}, Postgres port=${PG_PORT}"
echo ""

# API health
if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
  echo "  [OK] API (${API_URL}/health)"
else
  echo "  [FAIL] API (${API_URL}/health) not reachable"
  FAILED=1
fi

# Redis (host port 6380 by default)
if command -v redis-cli > /dev/null 2>&1; then
  if redis-cli -p "${REDIS_PORT}" ping 2>/dev/null | grep -q PONG; then
    echo "  [OK] Redis (port ${REDIS_PORT})"
  else
    echo "  [FAIL] Redis (port ${REDIS_PORT}) not responding"
    FAILED=1
  fi
else
  echo "  [SKIP] redis-cli not installed; cannot check Redis"
fi

# Postgres (host port 5433 in docker-compose-core)
if command -v pg_isready > /dev/null 2>&1; then
  if pg_isready -h localhost -p "${PG_PORT}" -U postgres > /dev/null 2>&1; then
    echo "  [OK] Postgres (port ${PG_PORT})"
  else
    echo "  [FAIL] Postgres (port ${PG_PORT}) not ready"
    FAILED=1
  fi
else
  echo "  [SKIP] pg_isready not installed; cannot check Postgres"
fi

echo ""
if [ $FAILED -eq 0 ]; then
  echo "Core stack health check passed."
  exit 0
else
  echo "One or more checks failed."
  exit 1
fi
