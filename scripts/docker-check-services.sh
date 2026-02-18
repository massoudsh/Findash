#!/usr/bin/env bash
# Check that all Docker Compose (core) services are up.
# Usage: ./scripts/docker-check-services.sh
# Requires: Docker Desktop running.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE_FILE="docker-compose-core.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Please start Docker Desktop and try again."
  exit 1
fi

echo "=============================================="
echo "  Docker services (docker-compose-core.yml)"
echo "=============================================="
echo ""

docker compose -f "$COMPOSE_FILE" ps -a

echo ""
echo "Expected services (core stack):"
echo "  api, frontend, db, redis, celery-worker, celery-beat, prometheus, grafana"
echo "Optional (--profile llm): tgi-falcon, fingpt-inference"
echo ""

# Count running vs total (excluding volumes/networks)
RUNNING=$(docker compose -f "$COMPOSE_FILE" ps -a --status running -q 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$(docker compose -f "$COMPOSE_FILE" ps -a --format json 2>/dev/null | grep -c '"Name"' || echo "0")

if [ "$RUNNING" -gt 0 ]; then
  echo "Running containers: $RUNNING"
  echo ""
  echo "Quick health check (API, Redis, Postgres):"
  if [ -x "$SCRIPT_DIR/healthcheck-core.sh" ]; then
    "$SCRIPT_DIR/healthcheck-core.sh" || true
  else
    curl -sf "http://localhost:8011/health" >/dev/null 2>&1 && echo "  [OK] API http://localhost:8011/health" || echo "  [FAIL] API not reachable"
  fi
  echo ""
  echo "Celery (worker + beat):"
  if docker ps -q -f name=octopus-celery-worker | grep -q .; then
    docker exec octopus-celery-worker celery -A src.core.celery_app inspect ping 2>/dev/null && echo "  [OK] celery-worker responding" || echo "  [--] celery-worker running (inspect may need a moment)"
  else
    echo "  [--] celery-worker not running"
  fi
  if docker ps -q -f name=octopus-celery-beat | grep -q .; then
    echo "  [OK] celery-beat container up"
  else
    echo "  [--] celery-beat not running"
  fi
else
  echo "No containers running. Start with:"
  echo "  docker compose -f $COMPOSE_FILE up -d"
fi
