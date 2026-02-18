#!/usr/bin/env bash
# =============================================================================
# One-command demo for stakeholders (fundraising).
# Runs the full stack with Docker. Requires Docker Desktop.
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================="
echo "  Octopus (Findash) — Stakeholder Demo"
echo "=============================================="
echo ""

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH."
  echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop/"
  exit 1
fi

echo "Starting core stack (API, frontend, DB, Redis)..."
docker compose -f docker-compose-core.yml up -d

echo ""
echo "Waiting for services to be ready (up to 60s)..."
for i in $(seq 1 30); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8011/health 2>/dev/null | grep -q 200; then
    echo "API is up."
    break
  fi
  sleep 2
done

echo ""
echo "----------------------------------------------"
echo "  Demo is ready."
echo "  App:    http://localhost:3000"
echo "  API:    http://localhost:8011"
echo "  Docs:   http://localhost:8011/docs"
echo "----------------------------------------------"
echo ""
echo "To stop: docker compose -f docker-compose-core.yml down"
echo ""
