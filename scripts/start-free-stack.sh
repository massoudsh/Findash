#!/bin/bash

# =============================================================================
# Octopus Trading Platform - Free Open Source Stack Startup
# =============================================================================
# This script starts the trading platform using 100% free, open-source alternatives
# 
# Free Components Used:
# - Traefik (instead of Kong) - API Gateway & Load Balancer
# - Apache APISIX - Alternative high-performance API Gateway  
# - PostgreSQL & TimescaleDB - Time-series database
# - Redis - Caching and message broker
# - Redis Streams - Event streaming
# - Prometheus & Grafana - Monitoring
# - Elasticsearch & Kibana - Search and analytics
# - Keycloak - Identity and access management
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose-complete.yml"
PROJECT_NAME="octopus-trading"

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    🐙 OCTOPUS TRADING PLATFORM (FREE STACK)                  ║"
echo "║                          Starting Free Open Source Stack                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    print_error "docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Navigate to the correct directory
cd "$(dirname "$0")/.."

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from config/env.example..."
    if [ -f "config/env.example" ]; then
        cp config/env.example .env
        print_status "Created .env file from config/env.example"
    else
        print_error "config/env.example not found. Please create a .env file with required environment variables."
        exit 1
    fi
fi

# Check if APISIX config exists
if [ ! -f "apisix_conf/config.yaml" ]; then
    print_error "APISIX configuration not found at apisix_conf/config.yaml"
    exit 1
fi

print_status "Starting Octopus Trading Platform with free alternatives..."

# Start the core services first
print_status "Starting core infrastructure services..."
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d \
    db \
    redis \
    etcd

# Wait for core services to be ready
print_status "Waiting for core services to initialize..."
sleep 15

# Start data and monitoring services
print_status "Starting data and monitoring services..."
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d \
    elasticsearch \
    prometheus \
    grafana

# Wait for data services
sleep 10

# Start API gateways
print_status "Starting API Gateway services..."
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d \
    traefik \
    apisix

# Start security services
print_status "Starting security services..."
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d \
    keycloak-db \
    keycloak

# Wait for auth services
sleep 10

# Start application services
print_status "Starting application services..."
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d \
    api \
    celery-worker \
    celery-beat \
    flower

# Start development tools
print_status "Starting development tools..."
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d \
    kibana \
    pgadmin \
    redisinsight

print_status "All services started successfully!"

echo -e "\n${GREEN}🎉 Octopus Trading Platform is now running with FREE alternatives!${NC}\n"

echo "=== SERVICE ACCESS URLS ==="
echo -e "${BLUE}Core Application:${NC}"
echo "  • FastAPI Backend:        http://localhost:8000"
echo "  • FastAPI Docs:           http://localhost:8000/docs"
echo ""
echo -e "${BLUE}API Gateways (FREE):${NC}"
echo "  • Traefik Dashboard:      http://localhost:8080"
echo "  • Apache APISIX:          http://localhost:9080"
echo "  • APISIX Admin:           http://localhost:9091"
echo ""
echo -e "${BLUE}Monitoring (FREE):${NC}"
echo "  • Prometheus:             http://localhost:9090"
echo "  • Grafana:                http://localhost:3001 (admin/admin)"
echo "  • Flower (Celery):        http://localhost:5555"
echo ""
echo -e "${BLUE}Data & Search (FREE):${NC}"
echo "  • Elasticsearch:          http://localhost:9200"
echo "  • Kibana:                 http://localhost:5601"
echo ""
echo -e "${BLUE}Database & Management (FREE):${NC}"
echo "  • PostgreSQL:             localhost:5432"
echo "  • PgAdmin:                http://localhost:5050"
echo "  • Redis:                  localhost:6379"
echo "  • RedisInsight:           http://localhost:8083"
echo ""
echo -e "${BLUE}Security & Auth (FREE):${NC}"
echo "  • Keycloak:               http://localhost:8081 (admin/admin)"
echo ""
echo -e "${BLUE}Message Queue (FREE):${NC}"
echo "  • Redis Streams:          redis://localhost:6379/0 (stream key: market-data-stream)"

echo -e "\n${GREEN}💰 COST SAVINGS: You're using 100% FREE alternatives!${NC}"
echo -e "${GREEN}🚀 Total saved: Potentially $1000s/month vs commercial solutions${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Configure Traefik routes for your services"
echo "2. Set up APISIX routes via the admin API"
echo "3. Configure Keycloak realms and clients"
echo "4. Set up Grafana dashboards"
echo "5. Configure Elasticsearch indices"

echo -e "\n${BLUE}To stop all services:${NC}"
echo "  docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down"

echo -e "\n${BLUE}To view logs:${NC}"
echo "  docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f [service-name]"

echo -e "\n${GREEN}Happy Trading! 🐙📈${NC}" 