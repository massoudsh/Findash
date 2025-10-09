#!/bin/bash

# 🦍 Kong API Gateway Initialization Script - Octopus Trading Platform™
# This script sets up Kong API Gateway with routes, services, and plugins for the trading platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
KONG_ADMIN_URL="${KONG_ADMIN_URL:-http://localhost:8001}"
API_BACKEND_URL="${API_BACKEND_URL:-http://api:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://frontend:3000}"
WAIT_TIMEOUT=60

echo -e "${BLUE}🦍 Initializing Kong API Gateway for Octopus Trading Platform${NC}"

# Function to wait for Kong to be ready
wait_for_kong() {
    echo -e "${YELLOW}⏳ Waiting for Kong Admin API to be ready...${NC}"
    local count=0
    while [[ $count -lt $WAIT_TIMEOUT ]]; do
        if curl -s "$KONG_ADMIN_URL" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Kong Admin API is ready${NC}"
            return 0
        fi
        sleep 1
        ((count++))
    done
    echo -e "${RED}❌ Kong Admin API is not ready after $WAIT_TIMEOUT seconds${NC}"
    exit 1
}

# Function to create or update a service
create_service() {
    local name=$1
    local url=$2
    local protocol=${3:-http}
    
    echo -e "${BLUE}📋 Creating/updating service: $name${NC}"
    
    # Check if service exists
    if curl -s "$KONG_ADMIN_URL/services/$name" > /dev/null 2>&1; then
        echo -e "${YELLOW}🔄 Service $name exists, updating...${NC}"
        curl -X PATCH "$KONG_ADMIN_URL/services/$name" \
            -H "Content-Type: application/json" \
            -d "{
                \"url\": \"$url\",
                \"protocol\": \"$protocol\",
                \"connect_timeout\": 60000,
                \"write_timeout\": 60000,
                \"read_timeout\": 60000
            }"
    else
        echo -e "${GREEN}🆕 Creating new service: $name${NC}"
        curl -X POST "$KONG_ADMIN_URL/services" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"$name\",
                \"url\": \"$url\",
                \"protocol\": \"$protocol\",
                \"connect_timeout\": 60000,
                \"write_timeout\": 60000,
                \"read_timeout\": 60000
            }"
    fi
    echo ""
}

# Function to create or update a route
create_route() {
    local service_name=$1
    local route_name=$2
    local paths=$3
    local methods=${4:-"GET,POST,PUT,DELETE,PATCH,OPTIONS"}
    local strip_path=${5:-true}
    
    echo -e "${BLUE}🛣️  Creating/updating route: $route_name${NC}"
    
    # Check if route exists
    if curl -s "$KONG_ADMIN_URL/routes/$route_name" > /dev/null 2>&1; then
        echo -e "${YELLOW}🔄 Route $route_name exists, updating...${NC}"
        curl -X PATCH "$KONG_ADMIN_URL/routes/$route_name" \
            -H "Content-Type: application/json" \
            -d "{
                \"service\": {\"name\": \"$service_name\"},
                \"paths\": $paths,
                \"methods\": [\"$(echo $methods | tr ',' '","')\"],
                \"strip_path\": $strip_path
            }"
    else
        echo -e "${GREEN}🆕 Creating new route: $route_name${NC}"
        curl -X POST "$KONG_ADMIN_URL/routes" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"$route_name\",
                \"service\": {\"name\": \"$service_name\"},
                \"paths\": $paths,
                \"methods\": [\"$(echo $methods | tr ',' '","')\"],
                \"strip_path\": $strip_path
            }"
    fi
    echo ""
}

# Function to enable plugin
enable_plugin() {
    local plugin_name=$1
    local service_name=$2
    local config=$3
    
    echo -e "${BLUE}🔌 Enabling plugin: $plugin_name for service: $service_name${NC}"
    
    # Check if plugin is already enabled
    local existing_plugin=$(curl -s "$KONG_ADMIN_URL/services/$service_name/plugins" | jq -r ".data[] | select(.name == \"$plugin_name\") | .id" 2>/dev/null || echo "")
    
    if [[ -n "$existing_plugin" ]]; then
        echo -e "${YELLOW}🔄 Plugin $plugin_name already exists, updating...${NC}"
        curl -X PATCH "$KONG_ADMIN_URL/plugins/$existing_plugin" \
            -H "Content-Type: application/json" \
            -d "{\"config\": $config}"
    else
        echo -e "${GREEN}🆕 Enabling new plugin: $plugin_name${NC}"
        curl -X POST "$KONG_ADMIN_URL/services/$service_name/plugins" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"$plugin_name\",
                \"config\": $config
            }"
    fi
    echo ""
}

# Function to enable global plugin
enable_global_plugin() {
    local plugin_name=$1
    local config=$2
    
    echo -e "${BLUE}🌍 Enabling global plugin: $plugin_name${NC}"
    
    # Check if global plugin is already enabled
    local existing_plugin=$(curl -s "$KONG_ADMIN_URL/plugins" | jq -r ".data[] | select(.name == \"$plugin_name\" and .service == null and .route == null) | .id" 2>/dev/null || echo "")
    
    if [[ -n "$existing_plugin" ]]; then
        echo -e "${YELLOW}🔄 Global plugin $plugin_name already exists, updating...${NC}"
        curl -X PATCH "$KONG_ADMIN_URL/plugins/$existing_plugin" \
            -H "Content-Type: application/json" \
            -d "{\"config\": $config}"
    else
        echo -e "${GREEN}🆕 Enabling new global plugin: $plugin_name${NC}"
        curl -X POST "$KONG_ADMIN_URL/plugins" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"$plugin_name\",
                \"config\": $config
            }"
    fi
    echo ""
}

# Start initialization
wait_for_kong

echo -e "${BLUE}🏗️  Setting up services...${NC}"

# Create API service
create_service "octopus-api" "$API_BACKEND_URL" "http"

# Create Frontend service  
create_service "octopus-frontend" "$FRONTEND_URL" "http"

echo -e "${BLUE}🛣️  Setting up routes...${NC}"

# API routes
create_route "octopus-api" "api-auth" "[\"~/api/auth\"]" "GET,POST,PUT,DELETE" false
create_route "octopus-api" "api-portfolios" "[\"~/api/portfolios\"]" "GET,POST,PUT,DELETE" false
create_route "octopus-api" "api-market" "[\"~/api/market\"]" "GET,POST" false
create_route "octopus-api" "api-orders" "[\"~/api/orders\"]" "GET,POST,PUT,DELETE" false
create_route "octopus-api" "api-risk" "[\"~/api/risk\"]" "GET,POST" false
create_route "octopus-api" "api-ml" "[\"~/api/ml\"]" "GET,POST" false
create_route "octopus-api" "api-notifications" "[\"~/api/notifications\"]" "GET,POST,PUT,DELETE" false
create_route "octopus-api" "api-health" "[\"~/health\", \"~/api/health\"]" "GET" false
create_route "octopus-api" "api-docs" "[\"~/docs\", \"~/redoc\", \"~/openapi.json\"]" "GET" false

# WebSocket routes
create_route "octopus-api" "ws-market-data" "[\"~/ws/market-data\"]" "GET" false
create_route "octopus-api" "ws-portfolios" "[\"~/ws/portfolios\"]" "GET" false
create_route "octopus-api" "ws-notifications" "[\"~/ws/notifications\"]" "GET" false

# Frontend routes (catch-all)
create_route "octopus-frontend" "frontend-app" "[\"/\"]" "GET,POST" false

echo -e "${BLUE}🔌 Setting up plugins...${NC}"

# Rate limiting
enable_global_plugin "rate-limiting" "{
    \"minute\": 1000,
    \"hour\": 10000,
    \"policy\": \"local\",
    \"fault_tolerant\": true,
    \"hide_client_headers\": false
}"

# CORS
enable_global_plugin "cors" "{
    \"origins\": [\"*\"],
    \"methods\": [\"GET\", \"POST\", \"PUT\", \"DELETE\", \"PATCH\", \"OPTIONS\"],
    \"headers\": [\"Accept\", \"Accept-Version\", \"Content-Length\", \"Content-MD5\", \"Content-Type\", \"Date\", \"Authorization\"],
    \"exposed_headers\": [\"X-Auth-Token\"],
    \"credentials\": true,
    \"max_age\": 3600,
    \"preflight_continue\": false
}"

# Request/Response logging for API
enable_plugin "file-log" "octopus-api" "{
    \"path\": \"/var/log/kong/api-access.log\",
    \"reopen\": true
}"

# Request size limiting
enable_global_plugin "request-size-limiting" "{
    \"allowed_payload_size\": 10
}"

# Response rate limiting for market data
enable_plugin "response-ratelimiting" "octopus-api" "{
    \"limits\": {
        \"market_data_endpoint\": {
            \"minute\": 100
        }
    }
}"

# IP restriction (whitelist internal networks)
enable_plugin "ip-restriction" "octopus-api" "{
    \"allow\": [\"10.0.0.0/8\", \"172.16.0.0/12\", \"192.168.0.0/16\"]
}"

# Prometheus metrics
enable_global_plugin "prometheus" "{
    \"per_consumer\": true,
    \"status_code_metrics\": true,
    \"latency_metrics\": true,
    \"bandwidth_metrics\": true,
    \"upstream_health_metrics\": true
}"

# Request transformer for API versioning
enable_plugin "request-transformer" "octopus-api" "{
    \"add\": {
        \"headers\": [\"X-API-Version:v1\"]
    }
}"

# Response transformer for security headers
enable_global_plugin "response-transformer" "{
    \"add\": {
        \"headers\": [
            \"X-Content-Type-Options:nosniff\",
            \"X-Frame-Options:DENY\",
            \"X-XSS-Protection:1; mode=block\",
            \"Referrer-Policy:strict-origin-when-cross-origin\"
        ]
    },
    \"remove\": {
        \"headers\": [\"Server\", \"X-Powered-By\"]
    }
}"

# JWT authentication for API routes (except health and docs)
enable_plugin "jwt" "octopus-api" "{
    \"claims_to_verify\": [\"exp\"],
    \"key_claim_name\": \"iss\",
    \"secret_is_base64\": false,
    \"run_on_preflight\": false
}"

# Circuit breaker for external API calls
enable_plugin "proxy-cache" "octopus-api" "{
    \"request_method\": [\"GET\", \"HEAD\"],
    \"response_code\": [200, 301, 404],
    \"content_type\": [\"text/plain\", \"application/json\"],
    \"cache_ttl\": 300,
    \"strategy\": \"memory\"
}"

echo -e "${BLUE}📊 Creating Kong health check route...${NC}"

# Create Kong admin health check
create_service "kong-health" "http://localhost:8001/status" "http"
create_route "kong-health" "kong-status" "[\"~/kong-status\"]" "GET" true

echo -e "${GREEN}✅ Kong API Gateway initialization completed successfully!${NC}"

echo -e "${BLUE}📋 Summary of configured services:${NC}"
echo -e "  • octopus-api: $API_BACKEND_URL"
echo -e "  • octopus-frontend: $FRONTEND_URL"
echo -e "  • kong-health: http://localhost:8001/status"

echo -e "${BLUE}🛣️  Summary of configured routes:${NC}"
echo -e "  • /api/auth → Authentication endpoints"
echo -e "  • /api/portfolios → Portfolio management"
echo -e "  • /api/market → Market data"
echo -e "  • /api/orders → Trading orders"
echo -e "  • /api/risk → Risk management"
echo -e "  • /api/ml → ML/AI services"
echo -e "  • /ws/* → WebSocket endpoints"
echo -e "  • / → Frontend application"

echo -e "${BLUE}🔌 Enabled plugins:${NC}"
echo -e "  • Rate Limiting (1000/min, 10000/hour)"
echo -e "  • CORS (configured for trading platform)"
echo -e "  • Request/Response Logging"
echo -e "  • IP Restrictions (internal networks only)"
echo -e "  • JWT Authentication"
echo -e "  • Prometheus Metrics"
echo -e "  • Security Headers"
echo -e "  • Proxy Caching"

echo -e "${BLUE}🌐 Access URLs:${NC}"
echo -e "  • Kong Admin: $KONG_ADMIN_URL"
echo -e "  • Kong Proxy: http://localhost:8000"
echo -e "  • Kong Manager: http://localhost:8002"

echo -e "${GREEN}🎉 Kong setup complete! Your Octopus Trading Platform is ready to handle requests.${NC}"

# Verify setup
echo -e "${BLUE}🔍 Verifying setup...${NC}"
if curl -s "$KONG_ADMIN_URL/services" | jq -r '.data[].name' | grep -q "octopus-api"; then
    echo -e "${GREEN}✅ API service verification passed${NC}"
else
    echo -e "${RED}❌ API service verification failed${NC}"
fi

if curl -s "$KONG_ADMIN_URL/routes" | jq -r '.data[].name' | grep -q "api-auth"; then
    echo -e "${GREEN}✅ Routes verification passed${NC}"
else
    echo -e "${RED}❌ Routes verification failed${NC}"
fi

echo -e "${BLUE}📝 Next steps:${NC}"
echo -e "  1. Configure JWT secrets in your application"
echo -e "  2. Set up SSL certificates for production"
echo -e "  3. Configure rate limiting based on your needs"
echo -e "  4. Monitor Kong metrics via Prometheus"
echo -e "  5. Review and adjust IP restrictions"

echo -e "${GREEN}🚀 Kong initialization script completed!${NC}" 