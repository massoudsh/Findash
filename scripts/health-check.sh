#!/bin/bash

# 🏥 Health Check Script - Octopus Trading Platform™
# This script performs comprehensive health checks on all platform services

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-octopus_app}"
DB_NAME="${DB_NAME:-trading_db}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
KONG_ADMIN_URL="${KONG_ADMIN_URL:-http://localhost:8001}"
KEYCLOAK_URL="${KEYCLOAK_URL:-http://localhost:8080}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3001}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

# Health check results
HEALTH_RESULTS=()
FAILED_CHECKS=0
TOTAL_CHECKS=0

echo -e "${BLUE}🏥 Octopus Trading Platform Health Check${NC}"
echo -e "${CYAN}=====================================${NC}"
echo ""

# Function to add health result
add_result() {
    local service=$1
    local status=$2
    local message=$3
    local details=${4:-""}
    
    HEALTH_RESULTS+=("$service:$status:$message:$details")
    ((TOTAL_CHECKS++))
    
    if [[ "$status" == "FAIL" ]]; then
        ((FAILED_CHECKS++))
        echo -e "${RED}❌ $service: $message${NC}"
        if [[ -n "$details" ]]; then
            echo -e "   ${YELLOW}Details: $details${NC}"
        fi
    elif [[ "$status" == "WARN" ]]; then
        echo -e "${YELLOW}⚠️  $service: $message${NC}"
        if [[ -n "$details" ]]; then
            echo -e "   ${CYAN}Details: $details${NC}"
        fi
    else
        echo -e "${GREEN}✅ $service: $message${NC}"
        if [[ -n "$details" ]]; then
            echo -e "   ${CYAN}Details: $details${NC}"
        fi
    fi
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    local timeout=${4:-10}
    
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" -m "$timeout" "$url" 2>/dev/null || echo "000")
    local response_time=$(curl -s -o /dev/null -w "%{time_total}" -m "$timeout" "$url" 2>/dev/null || echo "0")
    
    if [[ "$response_code" == "$expected_status" ]]; then
        add_result "$name" "PASS" "HTTP $response_code OK" "Response time: ${response_time}s"
    elif [[ "$response_code" == "000" ]]; then
        add_result "$name" "FAIL" "Connection failed" "Timeout after ${timeout}s"
    else
        add_result "$name" "FAIL" "HTTP $response_code" "Expected $expected_status"
    fi
}

# Function to check TCP port
check_tcp_port() {
    local name=$1
    local host=$2
    local port=$3
    local timeout=${4:-5}
    
    if timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        add_result "$name" "PASS" "Port $port is open" "Host: $host"
    else
        add_result "$name" "FAIL" "Port $port is closed or unreachable" "Host: $host"
    fi
}

# Function to check PostgreSQL
check_postgresql() {
    echo -e "${BLUE}🗄️  Checking PostgreSQL Database...${NC}"
    
    # Check if psql is available
    if ! command -v psql > /dev/null 2>&1; then
        add_result "PostgreSQL" "WARN" "psql not found, using TCP check only"
        check_tcp_port "PostgreSQL-TCP" "$DB_HOST" "$DB_PORT"
        return
    fi
    
    # Check connection
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
        # Get database size and connection count
        local db_info=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
            SELECT 
                pg_size_pretty(pg_database_size('$DB_NAME')) as db_size,
                (SELECT count(*) FROM pg_stat_activity WHERE datname = '$DB_NAME') as connections;
        " 2>/dev/null | tr -d ' ')
        
        add_result "PostgreSQL" "PASS" "Database connection successful" "Size: $db_info"
        
        # Check TimescaleDB extension
        local timescale_version=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
            SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';
        " 2>/dev/null | tr -d ' ')
        
        if [[ -n "$timescale_version" ]]; then
            add_result "TimescaleDB" "PASS" "Extension loaded" "Version: $timescale_version"
        else
            add_result "TimescaleDB" "WARN" "Extension not found"
        fi
        
        # Check key tables
        local table_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
            SELECT count(*) FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name IN ('users', 'portfolios', 'orders', 'positions');
        " 2>/dev/null | tr -d ' ')
        
        if [[ "$table_count" == "4" ]]; then
            add_result "Database-Schema" "PASS" "Core tables present" "Tables: $table_count/4"
        else
            add_result "Database-Schema" "WARN" "Some core tables missing" "Tables: $table_count/4"
        fi
    else
        add_result "PostgreSQL" "FAIL" "Database connection failed"
    fi
}

# Function to check Redis
check_redis() {
    echo -e "${BLUE}📦 Checking Redis...${NC}"
    
    if command -v redis-cli > /dev/null 2>&1; then
        local redis_info=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null)
        if [[ "$redis_info" == "PONG" ]]; then
            local memory_info=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
            local connected_clients=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" info clients | grep connected_clients | cut -d: -f2 | tr -d '\r')
            add_result "Redis" "PASS" "Connection successful" "Memory: $memory_info, Clients: $connected_clients"
        else
            add_result "Redis" "FAIL" "Connection failed"
        fi
    else
        add_result "Redis" "WARN" "redis-cli not found, using TCP check"
        check_tcp_port "Redis-TCP" "$REDIS_HOST" "$REDIS_PORT"
    fi
}

# Function to check Redis Streams
check_redis_streams() {
    echo -e "${BLUE}🧵 Checking Redis Streams...${NC}"
    local stream_key="${REDIS_STREAM_KEY:-market-data-stream}"

    if command -v redis-cli > /dev/null 2>&1; then
        # Check stream existence (MKSTREAM will create it at runtime; warn if empty/missing)
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XINFO STREAM "$stream_key" > /dev/null 2>&1; then
            local length=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XLEN "$stream_key" 2>/dev/null | tr -d '\r')
            add_result "Redis-Streams" "PASS" "Stream available" "Stream: $stream_key, length: ${length:-0}"
        else
            add_result "Redis-Streams" "WARN" "Stream not found (yet)" "Stream: $stream_key"
        fi
    else
        add_result "Redis-Streams" "WARN" "redis-cli not found; cannot inspect streams"
    fi
}

# Function to check API endpoints
check_api_services() {
    echo -e "${BLUE}🔗 Checking API Services...${NC}"
    
    # Main API health endpoint
    check_http_endpoint "API-Health" "$API_URL/health"
    
    # API documentation
    check_http_endpoint "API-Docs" "$API_URL/docs"
    
    # Authentication endpoint (should return 401 without credentials)
    check_http_endpoint "API-Auth" "$API_URL/api/auth/me" "401"
    
    # Market data endpoint (should return 401 without credentials)
    check_http_endpoint "API-Market" "$API_URL/api/market/trending" "401"
    
    # WebSocket health (basic HTTP check)
    check_http_endpoint "WebSocket-Health" "$API_URL/ws" "400"
}

# Function to check infrastructure services
check_infrastructure() {
    echo -e "${BLUE}🏗️  Checking Infrastructure Services...${NC}"
    
    # Kong API Gateway
    if [[ -n "$KONG_ADMIN_URL" ]]; then
        check_http_endpoint "Kong-Admin" "$KONG_ADMIN_URL"
        check_http_endpoint "Kong-Status" "$KONG_ADMIN_URL/status"
    fi
    
    # Keycloak
    if [[ -n "$KEYCLOAK_URL" ]]; then
        check_http_endpoint "Keycloak" "$KEYCLOAK_URL/auth/realms/master"
    fi
    
    # Frontend
    check_http_endpoint "Frontend" "$FRONTEND_URL"
}

# Function to check monitoring services
check_monitoring() {
    echo -e "${BLUE}📊 Checking Monitoring Services...${NC}"
    
    # Prometheus
    if [[ -n "$PROMETHEUS_URL" ]]; then
        check_http_endpoint "Prometheus" "$PROMETHEUS_URL/api/v1/query?query=up"
        check_http_endpoint "Prometheus-Targets" "$PROMETHEUS_URL/api/v1/targets"
    fi
    
    # Grafana
    if [[ -n "$GRAFANA_URL" ]]; then
        check_http_endpoint "Grafana" "$GRAFANA_URL/api/health"
    fi
}

# Function to check system resources
check_system_resources() {
    echo -e "${BLUE}💻 Checking System Resources...${NC}"
    
    # Check available memory
    if command -v free > /dev/null 2>&1; then
        local memory_usage=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
        local available_memory=$(free -h | grep Mem | awk '{print $7}')
        
        if (( $(echo "$memory_usage" | cut -d% -f1 | cut -d. -f1) < 80 )); then
            add_result "Memory" "PASS" "Usage: $memory_usage" "Available: $available_memory"
        else
            add_result "Memory" "WARN" "High usage: $memory_usage" "Available: $available_memory"
        fi
    fi
    
    # Check disk space
    if command -v df > /dev/null 2>&1; then
        local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
        local available_disk=$(df -h / | awk 'NR==2 {print $4}')
        
        if (( disk_usage < 80 )); then
            add_result "Disk" "PASS" "Usage: ${disk_usage}%" "Available: $available_disk"
        elif (( disk_usage < 90 )); then
            add_result "Disk" "WARN" "High usage: ${disk_usage}%" "Available: $available_disk"
        else
            add_result "Disk" "FAIL" "Critical usage: ${disk_usage}%" "Available: $available_disk"
        fi
    fi
    
    # Check CPU load
    if command -v uptime > /dev/null 2>&1; then
        local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
        local cpu_cores=$(nproc 2>/dev/null || echo "1")
        local load_percent=$(echo "$load_avg $cpu_cores" | awk '{printf "%.1f%%", $1/$2 * 100}')
        
        if (( $(echo "$load_avg $cpu_cores" | awk '{print ($1/$2 < 0.7)}') )); then
            add_result "CPU" "PASS" "Load: $load_percent" "1min avg: $load_avg"
        elif (( $(echo "$load_avg $cpu_cores" | awk '{print ($1/$2 < 1.0)}') )); then
            add_result "CPU" "WARN" "High load: $load_percent" "1min avg: $load_avg"
        else
            add_result "CPU" "FAIL" "Critical load: $load_percent" "1min avg: $load_avg"
        fi
    fi
}

# Function to check Docker containers (if running in Docker)
check_docker_containers() {
    if command -v docker > /dev/null 2>&1; then
        echo -e "${BLUE}🐳 Checking Docker Containers...${NC}"
        
        local containers=("octopus-api" "octopus-frontend" "octopus-db" "octopus-redis")
        
        for container in "${containers[@]}"; do
            if docker ps --format "table {{.Names}}" | grep -q "$container"; then
                local status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null)
                local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no-healthcheck")
                
                if [[ "$status" == "running" ]]; then
                    if [[ "$health" == "healthy" || "$health" == "no-healthcheck" ]]; then
                        add_result "Docker-$container" "PASS" "Running" "Health: $health"
                    else
                        add_result "Docker-$container" "WARN" "Running but unhealthy" "Health: $health"
                    fi
                else
                    add_result "Docker-$container" "FAIL" "Not running" "Status: $status"
                fi
            else
                add_result "Docker-$container" "WARN" "Container not found"
            fi
        done
    fi
}

# Function to perform performance tests
check_performance() {
    echo -e "${BLUE}🚀 Checking Performance...${NC}"
    
    # API response time test
    if curl -s "$API_URL/health" > /dev/null 2>&1; then
        local api_response_time=$(curl -s -o /dev/null -w "%{time_total}" "$API_URL/health")
        local api_response_ms=$(echo "$api_response_time * 1000" | bc -l | cut -d. -f1)
        
        if (( api_response_ms < 500 )); then
            add_result "API-Performance" "PASS" "Response time: ${api_response_ms}ms"
        elif (( api_response_ms < 1000 )); then
            add_result "API-Performance" "WARN" "Slow response: ${api_response_ms}ms"
        else
            add_result "API-Performance" "FAIL" "Very slow response: ${api_response_ms}ms"
        fi
    fi
    
    # Database query performance
    if command -v psql > /dev/null 2>&1 && [[ -n "${DB_PASSWORD:-}" ]]; then
        local db_start_time=$(date +%s.%N)
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT count(*) FROM users;" > /dev/null 2>&1
        local db_end_time=$(date +%s.%N)
        local db_response_time=$(echo "$db_end_time - $db_start_time" | bc -l)
        local db_response_ms=$(echo "$db_response_time * 1000" | bc -l | cut -d. -f1)
        
        if (( db_response_ms < 100 )); then
            add_result "DB-Performance" "PASS" "Query time: ${db_response_ms}ms"
        elif (( db_response_ms < 500 )); then
            add_result "DB-Performance" "WARN" "Slow query: ${db_response_ms}ms"
        else
            add_result "DB-Performance" "FAIL" "Very slow query: ${db_response_ms}ms"
        fi
    fi
}

# Main execution
echo -e "${CYAN}Starting comprehensive health check...${NC}"
echo ""

# Core Infrastructure
check_postgresql
check_redis
check_redis_streams

# Application Services
check_api_services
check_infrastructure

# Monitoring
check_monitoring

# System Resources
check_system_resources

# Docker (if available)
check_docker_containers

# Performance
check_performance

# Summary
echo ""
echo -e "${CYAN}=====================================${NC}"
echo -e "${BLUE}🏥 Health Check Summary${NC}"
echo -e "${CYAN}=====================================${NC}"

local passed_checks=$((TOTAL_CHECKS - FAILED_CHECKS))
local success_rate=$(( passed_checks * 100 / TOTAL_CHECKS ))

echo -e "${BLUE}📊 Overall Status:${NC}"
echo -e "  • Total Checks: $TOTAL_CHECKS"
echo -e "  • Passed: ${GREEN}$passed_checks${NC}"
echo -e "  • Failed: ${RED}$FAILED_CHECKS${NC}"
echo -e "  • Success Rate: ${success_rate}%"

echo ""
echo -e "${BLUE}📋 Detailed Results:${NC}"

for result in "${HEALTH_RESULTS[@]}"; do
    IFS=':' read -r service status message details <<< "$result"
    case "$status" in
        "PASS") echo -e "  ${GREEN}✅ $service${NC}: $message" ;;
        "WARN") echo -e "  ${YELLOW}⚠️  $service${NC}: $message" ;;
        "FAIL") echo -e "  ${RED}❌ $service${NC}: $message" ;;
    esac
done

echo ""
echo -e "${BLUE}💡 Recommendations:${NC}"

if (( FAILED_CHECKS > 0 )); then
    echo -e "  ${RED}• Address failed health checks before proceeding${NC}"
    echo -e "  • Check service logs for detailed error information"
    echo -e "  • Verify network connectivity between services"
fi

if (( success_rate < 90 )); then
    echo -e "  ${YELLOW}• Success rate below 90% - investigate issues${NC}"
    echo -e "  • Consider restarting failing services"
fi

if (( success_rate >= 95 )); then
    echo -e "  ${GREEN}• System health is excellent!${NC}"
    echo -e "  • All critical services are operational"
fi

echo ""
echo -e "${BLUE}🔍 Monitoring Commands:${NC}"
echo -e "  • Docker logs: docker compose logs <service>"
echo -e "  • System logs: journalctl -u <service>"
echo -e "  • Database logs: docker compose logs db"
echo -e "  • API logs: curl $API_URL/health"

echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo -e "  1. Review any failed or warning checks"
echo -e "  2. Check application logs for errors"
echo -e "  3. Verify environment configuration"
echo -e "  4. Test end-to-end functionality"
echo -e "  5. Set up automated health monitoring"

# Exit with proper code
if (( FAILED_CHECKS > 0 )); then
    echo -e "${RED}Health check completed with failures${NC}"
    exit 1
elif (( success_rate < 90 )); then
    echo -e "${YELLOW}Health check completed with warnings${NC}"
    exit 2
else
    echo -e "${GREEN}🎉 All health checks passed!${NC}"
    exit 0
fi 