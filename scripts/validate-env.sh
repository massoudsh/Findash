#!/bin/bash

# 🔍 Environment Validation Script - Octopus Trading Platform™
# This script validates all required environment variables and configurations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
ENV_FILE="${ENV_FILE:-/Users/massoudshemirani/MyProjects/Octopus/Modules/.env}"
VALIDATION_RESULTS=()
FAILED_VALIDATIONS=0
TOTAL_VALIDATIONS=0

echo -e "${BLUE}🔍 Octopus Trading Platform Environment Validation${NC}"
echo -e "${CYAN}=================================================${NC}"
echo ""

# Function to add validation result
add_validation() {
    local category=$1
    local status=$2
    local message=$3
    local details=${4:-""}
    
    VALIDATION_RESULTS+=("$category:$status:$message:$details")
    ((TOTAL_VALIDATIONS++))
    
    if [[ "$status" == "FAIL" ]]; then
        ((FAILED_VALIDATIONS++))
        echo -e "${RED}❌ $category: $message${NC}"
        if [[ -n "$details" ]]; then
            echo -e "   ${YELLOW}Details: $details${NC}"
        fi
    elif [[ "$status" == "WARN" ]]; then
        echo -e "${YELLOW}⚠️  $category: $message${NC}"
        if [[ -n "$details" ]]; then
            echo -e "   ${CYAN}Details: $details${NC}"
        fi
    else
        echo -e "${GREEN}✅ $category: $message${NC}"
        if [[ -n "$details" ]]; then
            echo -e "   ${CYAN}Details: $details${NC}"
        fi
    fi
}

# Function to check if variable is set and not empty
check_required_var() {
    local var_name=$1
    local var_value="${!var_name:-}"
    local description=${2:-"Required variable"}
    
    if [[ -z "$var_value" ]]; then
        add_validation "$var_name" "FAIL" "Variable not set or empty" "$description"
        return 1
    else
        add_validation "$var_name" "PASS" "Variable is set" "Value: ${var_value:0:20}..."
        return 0
    fi
}

# Function to check optional variable
check_optional_var() {
    local var_name=$1
    local var_value="${!var_name:-}"
    local description=${2:-"Optional variable"}
    
    if [[ -z "$var_value" ]]; then
        add_validation "$var_name" "WARN" "Optional variable not set" "$description"
    else
        add_validation "$var_name" "PASS" "Variable is set" "Value: ${var_value:0:20}..."
    fi
}

# Function to validate URL format
validate_url() {
    local var_name=$1
    local var_value="${!var_name:-}"
    local description=${2:-"URL variable"}
    
    if [[ -z "$var_value" ]]; then
        add_validation "$var_name" "FAIL" "URL not set" "$description"
        return 1
    fi
    
    if [[ "$var_value" =~ ^https?://[^[:space:]]+$ ]]; then
        add_validation "$var_name" "PASS" "Valid URL format" "$var_value"
    else
        add_validation "$var_name" "FAIL" "Invalid URL format" "$var_value"
        return 1
    fi
}

# Function to validate database URL
validate_database_url() {
    local var_name=$1
    local var_value="${!var_name:-}"
    
    if [[ -z "$var_value" ]]; then
        add_validation "$var_name" "FAIL" "Database URL not set"
        return 1
    fi
    
    if [[ "$var_value" =~ ^postgresql://[^[:space:]]+$ ]]; then
        add_validation "$var_name" "PASS" "Valid PostgreSQL URL format"
        
        # Extract components for additional validation
        local db_user=$(echo "$var_value" | sed -n 's/^postgresql:\/\/\([^:]*\):.*/\1/p')
        local db_host=$(echo "$var_value" | sed -n 's/^postgresql:\/\/[^@]*@\([^:]*\):.*/\1/p')
        local db_port=$(echo "$var_value" | sed -n 's/^postgresql:\/\/[^@]*@[^:]*:\([0-9]*\)\/.*/\1/p')
        local db_name=$(echo "$var_value" | sed -n 's/^postgresql:\/\/[^\/]*\/\([^?]*\).*/\1/p')
        
        add_validation "DB_COMPONENTS" "PASS" "Database components extracted" "User: $db_user, Host: $db_host, Port: $db_port, DB: $db_name"
    else
        add_validation "$var_name" "FAIL" "Invalid PostgreSQL URL format" "Expected: postgresql://user:pass@host:port/dbname"
        return 1
    fi
}

# Function to validate API key format
validate_api_key() {
    local var_name=$1
    local var_value="${!var_name:-}"
    local min_length=${2:-10}
    
    if [[ -z "$var_value" ]]; then
        add_validation "$var_name" "WARN" "API key not set (external service may not work)"
        return 1
    fi
    
    if [[ ${#var_value} -ge $min_length ]]; then
        add_validation "$var_name" "PASS" "API key format valid" "Length: ${#var_value} chars"
    else
        add_validation "$var_name" "WARN" "API key too short" "Length: ${#var_value}, minimum: $min_length"
        return 1
    fi
}

# Function to validate secret key strength
validate_secret_key() {
    local var_name=$1
    local var_value="${!var_name:-}"
    local min_length=${2:-32}
    
    if [[ -z "$var_value" ]]; then
        add_validation "$var_name" "FAIL" "Secret key not set - SECURITY RISK"
        return 1
    fi
    
    if [[ ${#var_value} -ge $min_length ]]; then
        # Check for some complexity
        if [[ "$var_value" =~ [A-Za-z] && "$var_value" =~ [0-9] ]]; then
            add_validation "$var_name" "PASS" "Secret key is strong" "Length: ${#var_value} chars"
        else
            add_validation "$var_name" "WARN" "Secret key lacks complexity" "Consider adding letters and numbers"
        fi
    else
        add_validation "$var_name" "FAIL" "Secret key too short - SECURITY RISK" "Length: ${#var_value}, minimum: $min_length"
        return 1
    fi
}

# Function to validate boolean values
validate_boolean() {
    local var_name=$1
    local var_value="${!var_name:-}"
    local default_value=${2:-"false"}
    
    if [[ -z "$var_value" ]]; then
        add_validation "$var_name" "WARN" "Boolean not set, using default" "Default: $default_value"
        return 0
    fi
    
    if [[ "$var_value" =~ ^(true|false|1|0|yes|no)$ ]]; then
        add_validation "$var_name" "PASS" "Valid boolean value" "Value: $var_value"
    else
        add_validation "$var_name" "WARN" "Invalid boolean format" "Value: $var_value, expected: true/false"
    fi
}

# Function to validate port numbers
validate_port() {
    local var_name=$1
    local var_value="${!var_name:-}"
    local description=${2:-"Port number"}
    
    if [[ -z "$var_value" ]]; then
        add_validation "$var_name" "WARN" "Port not set" "$description"
        return 1
    fi
    
    if [[ "$var_value" =~ ^[0-9]+$ ]] && (( var_value >= 1 && var_value <= 65535 )); then
        add_validation "$var_name" "PASS" "Valid port number" "$var_value"
    else
        add_validation "$var_name" "FAIL" "Invalid port number" "Value: $var_value, range: 1-65535"
        return 1
    fi
}

# Load environment file if it exists
if [[ -f "$ENV_FILE" ]]; then
    echo -e "${BLUE}📁 Loading environment from: $ENV_FILE${NC}"
    set -a  # Automatically export all variables
    source "$ENV_FILE"
    set +a
    echo ""
else
    echo -e "${RED}❌ Environment file not found: $ENV_FILE${NC}"
    echo -e "${YELLOW}💡 Create it by copying from env.example${NC}"
    exit 1
fi

echo -e "${BLUE}🔐 Validating Security Configuration...${NC}"

# Security variables
validate_secret_key "SECRET_KEY" 32
validate_secret_key "JWT_SECRET_KEY" 32
validate_boolean "DEBUG" "false"
validate_required_var "ENVIRONMENT" "Environment type (development/staging/production)"

# SSL and security settings
validate_boolean "FORCE_HTTPS" "true"
validate_boolean "SECURE_COOKIES" "true"
check_optional_var "HSTS_MAX_AGE" "HTTP Strict Transport Security max age"

echo ""
echo -e "${BLUE}🗄️  Validating Database Configuration...${NC}"

# Database configuration
validate_database_url "DATABASE_URL"
check_required_var "DB_PASSWORD" "Database password"
validate_port "DB_PORT" "Database port"
check_optional_var "DB_POOL_SIZE" "Database connection pool size"
check_optional_var "DB_MAX_OVERFLOW" "Database max overflow connections"

echo ""
echo -e "${BLUE}📦 Validating Cache Configuration...${NC}"

# Redis configuration
validate_url "REDIS_URL"
check_optional_var "REDIS_PASSWORD" "Redis password"
validate_port "REDIS_PORT" "Redis port"

echo ""
echo -e "${BLUE}🌐 Validating API Configuration...${NC}"

# API configuration
validate_port "API_PORT" "API server port"
check_required_var "API_HOST" "API server host"
check_optional_var "CORS_ORIGINS" "CORS allowed origins"
check_optional_var "API_RATE_LIMIT" "API rate limiting"

echo ""
echo -e "${BLUE}🔑 Validating External API Keys...${NC}"

# External API keys
validate_api_key "ALPHA_VANTAGE_API_KEY" 16
validate_api_key "NEWS_API_KEY" 16
validate_api_key "OPENAI_API_KEY" 20
check_optional_var "FINNHUB_API_KEY" "Finnhub API key"
check_optional_var "POLYGON_API_KEY" "Polygon.io API key"
check_optional_var "TWITTER_BEARER_TOKEN" "Twitter API bearer token"

echo ""
echo -e "${BLUE}📧 Validating Email Configuration...${NC}"

# Email configuration
check_optional_var "SMTP_HOST" "SMTP server host"
validate_port "SMTP_PORT" "SMTP server port" || true
check_optional_var "SMTP_USER" "SMTP username"
check_optional_var "SMTP_PASSWORD" "SMTP password"
validate_boolean "SMTP_USE_TLS" "true"

echo ""
echo -e "${BLUE}📊 Validating Monitoring Configuration...${NC}"

# Monitoring configuration
check_optional_var "PROMETHEUS_PORT" "Prometheus metrics port"
check_optional_var "GRAFANA_ADMIN_PASSWORD" "Grafana admin password"
validate_boolean "ENABLE_METRICS" "true"
check_optional_var "SENTRY_DSN" "Sentry error tracking DSN"

echo ""
echo -e "${BLUE}☁️  Validating Cloud Configuration...${NC}"

# Cloud storage configuration
check_optional_var "AWS_ACCESS_KEY_ID" "AWS access key"
check_optional_var "AWS_SECRET_ACCESS_KEY" "AWS secret key"
check_optional_var "AWS_DEFAULT_REGION" "AWS region"
check_optional_var "S3_BUCKET_NAME" "S3 bucket for file storage"

echo ""
echo -e "${BLUE}🚀 Validating Message Queue Configuration...${NC}"

# Kafka configuration
check_required_var "KAFKA_BOOTSTRAP_SERVERS" "Kafka bootstrap servers"
check_optional_var "KAFKA_SECURITY_PROTOCOL" "Kafka security protocol"
check_optional_var "KAFKA_SASL_MECHANISM" "Kafka SASL mechanism"

echo ""
echo -e "${BLUE}🔐 Validating Authentication Configuration...${NC}"

# Authentication configuration
check_optional_var "KEYCLOAK_URL" "Keycloak authentication server URL"
check_optional_var "KEYCLOAK_REALM" "Keycloak realm name"
check_optional_var "KEYCLOAK_CLIENT_ID" "Keycloak client ID"
check_optional_var "KEYCLOAK_CLIENT_SECRET" "Keycloak client secret"

echo ""
echo -e "${BLUE}🏗️  Validating Infrastructure Configuration...${NC}"

# Infrastructure configuration
check_optional_var "KONG_ADMIN_URL" "Kong API Gateway admin URL"
check_optional_var "ELASTICSEARCH_URL" "Elasticsearch search engine URL"
check_optional_var "VAULT_URL" "HashiCorp Vault URL"
check_optional_var "VAULT_TOKEN" "HashiCorp Vault token"

# Function to generate missing secrets
generate_missing_secrets() {
    echo -e "${BLUE}🔧 Generating missing secrets...${NC}"
    
    local needs_update=false
    
    if [[ -z "${SECRET_KEY:-}" ]]; then
        local new_secret=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        echo "SECRET_KEY=$new_secret" >> "$ENV_FILE"
        echo -e "${GREEN}✅ Generated SECRET_KEY${NC}"
        needs_update=true
    fi
    
    if [[ -z "${JWT_SECRET_KEY:-}" ]]; then
        local new_jwt_secret=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        echo "JWT_SECRET_KEY=$new_jwt_secret" >> "$ENV_FILE"
        echo -e "${GREEN}✅ Generated JWT_SECRET_KEY${NC}"
        needs_update=true
    fi
    
    if [[ "$needs_update" == "true" ]]; then
        echo -e "${YELLOW}⚠️  Please reload your environment variables${NC}"
        echo -e "   Run: source $ENV_FILE"
    fi
}

# Function to check file permissions
check_file_permissions() {
    echo -e "${BLUE}🔒 Checking file permissions...${NC}"
    
    local env_perms=$(stat -c "%a" "$ENV_FILE" 2>/dev/null || stat -f "%A" "$ENV_FILE" 2>/dev/null)
    
    if [[ "$env_perms" == "600" || "$env_perms" == "0600" ]]; then
        add_validation "ENV_PERMISSIONS" "PASS" "Environment file has secure permissions" "Permissions: $env_perms"
    else
        add_validation "ENV_PERMISSIONS" "WARN" "Environment file permissions not secure" "Current: $env_perms, recommended: 600"
        echo -e "${YELLOW}💡 Fix with: chmod 600 $ENV_FILE${NC}"
    fi
}

# Function to validate environment-specific settings
validate_environment_specific() {
    local env_type="${ENVIRONMENT:-development}"
    
    echo -e "${BLUE}🎯 Validating $env_type environment settings...${NC}"
    
    case "$env_type" in
        "production")
            # Production-specific validations
            if [[ "${DEBUG:-}" == "true" ]]; then
                add_validation "PROD_DEBUG" "FAIL" "DEBUG should be false in production" "Current: $DEBUG"
            else
                add_validation "PROD_DEBUG" "PASS" "DEBUG properly disabled in production"
            fi
            
            if [[ "${FORCE_HTTPS:-}" != "true" ]]; then
                add_validation "PROD_HTTPS" "FAIL" "HTTPS should be enforced in production" "Current: ${FORCE_HTTPS:-not_set}"
            else
                add_validation "PROD_HTTPS" "PASS" "HTTPS properly enforced in production"
            fi
            
            # Check for localhost URLs in production
            if [[ "${DATABASE_URL:-}" =~ localhost ]]; then
                add_validation "PROD_DB" "WARN" "Database URL contains localhost in production"
            fi
            ;;
        "development")
            add_validation "DEV_CONFIG" "PASS" "Development environment detected"
            if [[ "${DEBUG:-}" != "true" ]]; then
                add_validation "DEV_DEBUG" "WARN" "Consider enabling DEBUG in development"
            fi
            ;;
        "staging")
            add_validation "STAGING_CONFIG" "PASS" "Staging environment detected"
            ;;
        *)
            add_validation "ENV_TYPE" "WARN" "Unknown environment type" "Value: $env_type"
            ;;
    esac
}

# Run additional validations
check_file_permissions
validate_environment_specific

# Offer to generate missing secrets
if [[ $FAILED_VALIDATIONS -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}🔧 Some critical variables are missing. Would you like to generate them? [y/N]${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy] ]]; then
        generate_missing_secrets
    fi
fi

# Summary
echo ""
echo -e "${CYAN}=================================================${NC}"
echo -e "${BLUE}🔍 Environment Validation Summary${NC}"
echo -e "${CYAN}=================================================${NC}"

local passed_validations=$((TOTAL_VALIDATIONS - FAILED_VALIDATIONS))
local success_rate=$(( passed_validations * 100 / TOTAL_VALIDATIONS ))

echo -e "${BLUE}📊 Overall Status:${NC}"
echo -e "  • Total Validations: $TOTAL_VALIDATIONS"
echo -e "  • Passed: ${GREEN}$passed_validations${NC}"
echo -e "  • Failed: ${RED}$FAILED_VALIDATIONS${NC}"
echo -e "  • Success Rate: ${success_rate}%"

echo ""
echo -e "${BLUE}📋 Detailed Results:${NC}"

# Group results by category
declare -A categories
for result in "${VALIDATION_RESULTS[@]}"; do
    IFS=':' read -r category status message details <<< "$result"
    case "$status" in
        "PASS") categories["PASS"]+="  ${GREEN}✅ $category${NC}: $message\n" ;;
        "WARN") categories["WARN"]+="  ${YELLOW}⚠️  $category${NC}: $message\n" ;;
        "FAIL") categories["FAIL"]+="  ${RED}❌ $category${NC}: $message\n" ;;
    esac
done

# Display grouped results
if [[ -n "${categories["FAIL"]:-}" ]]; then
    echo -e "${RED}Failed Validations:${NC}"
    echo -e "${categories["FAIL"]}"
fi

if [[ -n "${categories["WARN"]:-}" ]]; then
    echo -e "${YELLOW}Warnings:${NC}"
    echo -e "${categories["WARN"]}"
fi

if [[ -n "${categories["PASS"]:-}" && "${SHOW_PASSED:-false}" == "true" ]]; then
    echo -e "${GREEN}Passed Validations:${NC}"
    echo -e "${categories["PASS"]}"
fi

echo ""
echo -e "${BLUE}💡 Recommendations:${NC}"

if (( FAILED_VALIDATIONS > 0 )); then
    echo -e "  ${RED}• Address failed validations before deploying${NC}"
    echo -e "  • Generate missing secret keys for security"
    echo -e "  • Review database and Redis connection strings"
fi

if (( success_rate < 80 )); then
    echo -e "  ${YELLOW}• Success rate below 80% - review configuration${NC}"
    echo -e "  • Consider using environment-specific .env files"
fi

if (( success_rate >= 95 )); then
    echo -e "  ${GREEN}• Configuration is excellent!${NC}"
    echo -e "  • Environment is properly configured"
fi

echo ""
echo -e "${BLUE}🔧 Quick Fixes:${NC}"
echo -e "  • Generate secrets: openssl rand -base64 32"
echo -e "  • Secure .env file: chmod 600 $ENV_FILE"
echo -e "  • Validate URLs: curl -s <URL> > /dev/null"
echo -e "  • Test database: psql \$DATABASE_URL -c 'SELECT 1;'"

echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo -e "  1. Fix any failed validations"
echo -e "  2. Set up external API keys for full functionality"
echo -e "  3. Configure monitoring and alerting"
echo -e "  4. Test environment with health checks"
echo -e "  5. Set up proper secrets management for production"

# Exit with proper code
if (( FAILED_VALIDATIONS > 0 )); then
    echo -e "${RED}Environment validation completed with failures${NC}"
    exit 1
elif (( success_rate < 90 )); then
    echo -e "${YELLOW}Environment validation completed with warnings${NC}"
    exit 2
else
    echo -e "${GREEN}🎉 All environment validations passed!${NC}"
    exit 0
fi 