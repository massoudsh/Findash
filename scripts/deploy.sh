#!/bin/bash
# =========================================================
# Quantum Trading Matrix™ - Production Deployment Script
# =========================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/opt/qtm/backups/deployment_$TIMESTAMP"
LOG_FILE="/var/log/qtm/deployment_$TIMESTAMP.log"

# Default values
ENVIRONMENT="production"
FORCE_DEPLOY=false
SKIP_TESTS=false
SKIP_BACKUP=false
DRY_RUN=false

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] $message${NC}" | tee -a "$LOG_FILE"
}

print_success() { print_message "$GREEN" "✅ $1"; }
print_warning() { print_message "$YELLOW" "⚠️  $1"; }
print_error() { print_message "$RED" "❌ $1"; }
print_info() { print_message "$BLUE" "ℹ️  $1"; }

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Quantum Trading Matrix™ to production environment.

OPTIONS:
    -e, --environment ENV     Target environment (default: production)
    -f, --force              Force deployment without confirmation
    -t, --skip-tests         Skip running tests
    -b, --skip-backup        Skip creating backup
    -d, --dry-run            Show what would be deployed without actually deploying
    -h, --help               Show this help message

EXAMPLES:
    $0                       # Interactive deployment with all checks
    $0 --force --skip-tests  # Quick deployment without tests
    $0 --dry-run             # Preview deployment actions
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -b|--skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown argument: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Pre-deployment checks
pre_deployment_checks() {
    print_info "Running pre-deployment checks..."
    
    # Check if running as root or with sudo
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. This is not recommended for production deployments."
        if [[ "$FORCE_DEPLOY" == false ]]; then
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_error "Deployment cancelled."
                exit 1
            fi
        fi
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    
    # Check if required environment file exists
    if [[ ! -f "$PROJECT_ROOT/config/production.env" ]]; then
        print_error "Production environment file not found at config/production.env"
        exit 1
    fi
    
    # Check if production Docker Compose file exists
    if [[ ! -f "$PROJECT_ROOT/deploy/production.yml" ]]; then
        print_error "Production Docker Compose file not found at deploy/production.yml"
        exit 1
    fi
    
    # Check disk space (require at least 5GB free)
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=5242880  # 5GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
        print_error "Insufficient disk space. Required: 5GB, Available: $((AVAILABLE_SPACE/1024/1024))GB"
        exit 1
    fi
    
    # Check if ports are available
    REQUIRED_PORTS=(80 443 5432 6379 9090 3000)
    for port in "${REQUIRED_PORTS[@]}"; do
        if netstat -tuln | grep -q ":$port "; then
            print_warning "Port $port is already in use"
        fi
    done
    
    print_success "Pre-deployment checks completed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        print_warning "Skipping tests as requested"
        return 0
    fi
    
    print_info "Running test suite..."
    
    cd "$PROJECT_ROOT"
    
    # Run Python tests
    if [[ -f "requirements-dev.txt" ]]; then
        python -m pytest tests/ -v --tb=short || {
            print_error "Tests failed. Deployment aborted."
            exit 1
        }
    else
        print_warning "requirements-dev.txt not found, skipping Python tests"
    fi
    
    # Run frontend tests if they exist
    if [[ -f "frontend/package.json" ]]; then
        cd frontend
        if npm test --passWithNoTests; then
            print_success "Frontend tests passed"
        else
            print_error "Frontend tests failed. Deployment aborted."
            exit 1
        fi
        cd "$PROJECT_ROOT"
    fi
    
    print_success "All tests passed"
}

# Create backup
create_backup() {
    if [[ "$SKIP_BACKUP" == true ]]; then
        print_warning "Skipping backup as requested"
        return 0
    fi
    
    print_info "Creating deployment backup..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Backup database if it exists
    if docker ps | grep -q qtm-postgres; then
        print_info "Backing up database..."
        docker exec qtm-postgres pg_dump -U qtm_user qtm_prod > "$BACKUP_DIR/database_backup.sql" || {
            print_warning "Database backup failed, continuing..."
        }
    fi
    
    # Backup application data
    if [[ -d "/opt/qtm/data" ]]; then
        print_info "Backing up application data..."
        cp -r /opt/qtm/data "$BACKUP_DIR/" || {
            print_warning "Data backup failed, continuing..."
        }
    fi
    
    # Backup current Docker images
    print_info "Backing up current Docker images..."
    docker images --format "table {{.Repository}}:{{.Tag}}" | grep quantumtradingmatrix > "$BACKUP_DIR/docker_images.txt" || true
    
    print_success "Backup created at $BACKUP_DIR"
}

# Build Docker images
build_images() {
    print_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    if [[ "$DRY_RUN" == false ]]; then
        docker build -t quantumtradingmatrix/api:latest -f Dockerfile . || {
            print_error "Failed to build API Docker image"
            exit 1
        }
        
        # Build frontend image if Dockerfile exists
        if [[ -f "frontend/Dockerfile" ]]; then
            docker build -t quantumtradingmatrix/frontend:latest -f frontend/Dockerfile frontend/ || {
                print_error "Failed to build frontend Docker image"
                exit 1
            }
        fi
    else
        print_info "[DRY RUN] Would build Docker images"
    fi
    
    print_success "Docker images built successfully"
}

# Deploy application
deploy_application() {
    print_info "Deploying application..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == false ]]; then
        # Copy environment file
        cp config/production.env .env
        
        # Deploy using Docker Compose
        docker-compose -f deploy/production.yml down || true
        docker-compose -f deploy/production.yml pull || true
        docker-compose -f deploy/production.yml up -d || {
            print_error "Deployment failed"
            rollback_deployment
            exit 1
        }
        
        # Wait for services to be ready
        print_info "Waiting for services to start..."
        sleep 30
        
        # Health check
        for i in {1..10}; do
            if curl -f http://localhost/health >/dev/null 2>&1; then
                print_success "Application is responding to health checks"
                break
            else
                print_info "Waiting for application to start... (attempt $i/10)"
                sleep 10
            fi
            
            if [[ $i -eq 10 ]]; then
                print_error "Application failed to start properly"
                rollback_deployment
                exit 1
            fi
        done
    else
        print_info "[DRY RUN] Would deploy application using deploy/production.yml"
    fi
    
    print_success "Application deployed successfully"
}

# Initialize database
initialize_database() {
    print_info "Initializing PostgreSQL database..."
    
    if [[ "$DRY_RUN" == false ]]; then
        # Wait for database to be ready
        for i in {1..30}; do
            if docker exec qtm-postgres pg_isready -U qtm_user >/dev/null 2>&1; then
                break
            fi
            print_info "Waiting for PostgreSQL database... (attempt $i/30)"
            sleep 2
        done
        
        # Run database initialization using new PostgreSQL script
        print_info "Running PostgreSQL database initialization..."
        docker exec qtm-api python database/postgres_init.py || {
            print_error "PostgreSQL database initialization failed"
            exit 1
        }
        
        # Verify database setup
        print_info "Verifying database setup..."
        docker exec qtm-api python database/postgres_init.py --verify || {
            print_warning "Database verification failed, but continuing..."
        }
    else
        print_info "[DRY RUN] Would initialize PostgreSQL database"
    fi
    
    print_success "PostgreSQL database initialized"
}

# Run post-deployment checks
post_deployment_checks() {
    print_info "Running post-deployment checks..."
    
    if [[ "$DRY_RUN" == false ]]; then
        # Check if all containers are running
        EXPECTED_CONTAINERS=("qtm-nginx" "qtm-postgres" "qtm-redis" "qtm-grafana")
        for container in "${EXPECTED_CONTAINERS[@]}"; do
            if docker ps | grep -q "$container"; then
                print_success "Container $container is running"
            else
                print_warning "Container $container is not running"
            fi
        done
        
        # Check API endpoints
        API_ENDPOINTS=("/health" "/docs" "/portfolio/greeks")
        for endpoint in "${API_ENDPOINTS[@]}"; do
            if curl -f "http://localhost$endpoint" >/dev/null 2>&1; then
                print_success "Endpoint $endpoint is responding"
            else
                print_warning "Endpoint $endpoint is not responding"
            fi
        done
        
        # Check PostgreSQL connection
        if docker exec qtm-postgres pg_isready -U qtm_user >/dev/null 2>&1; then
            print_success "PostgreSQL database is ready"
        else
            print_warning "PostgreSQL database is not responding"
        fi
        
        # Check database tables
        TABLE_COUNT=$(docker exec qtm-postgres psql -U qtm_user -d qtm_prod -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ')
        if [[ "$TABLE_COUNT" -gt 0 ]]; then
            print_success "Database tables verified ($TABLE_COUNT tables found)"
        else
            print_warning "No database tables found"
        fi
        
        # Check logs for errors
        if docker logs qtm-api 2>&1 | grep -i error | head -5; then
            print_warning "Found errors in application logs (shown above)"
        fi
    else
        print_info "[DRY RUN] Would run post-deployment checks"
    fi
    
    print_success "Post-deployment checks completed"
}

# Rollback deployment
rollback_deployment() {
    print_warning "Rolling back deployment..."
    
    if [[ -d "$BACKUP_DIR" ]]; then
        # Stop current deployment
        docker-compose -f deploy/production.yml down || true
        
        # Restore database if backup exists
        if [[ -f "$BACKUP_DIR/database_backup.sql" ]]; then
            print_info "Restoring database backup..."
            docker exec -i qtm-postgres psql -U qtm_user -d qtm_prod < "$BACKUP_DIR/database_backup.sql" || true
        fi
        
        print_warning "Rollback completed. Please check the system manually."
    else
        print_error "No backup found for rollback"
    fi
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    
    # Remove temporary files
    rm -f .env
    
    # Cleanup old Docker images (keep last 3 versions)
    docker images | grep quantumtradingmatrix | awk '{print $3}' | tail -n +4 | xargs -r docker rmi || true
    
    print_success "Cleanup completed"
}

# Main deployment function
main() {
    print_info "Starting Quantum Trading Matrix™ deployment..."
    print_info "Environment: $ENVIRONMENT"
    print_info "Database: PostgreSQL (Direct Connection)"
    print_info "Timestamp: $TIMESTAMP"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Confirmation prompt
    if [[ "$FORCE_DEPLOY" == false && "$DRY_RUN" == false ]]; then
        echo
        print_warning "You are about to deploy to $ENVIRONMENT environment."
        print_warning "This will update the running application and may cause brief downtime."
        echo
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Deployment cancelled by user."
            exit 1
        fi
    fi
    
    # Set trap for cleanup on exit
    trap cleanup EXIT
    
    # Run deployment steps
    pre_deployment_checks
    run_tests
    create_backup
    build_images
    deploy_application
    initialize_database
    post_deployment_checks
    
    print_success "🎉 Deployment completed successfully!"
    print_info "Application is available at: http://localhost"
    print_info "API documentation: http://localhost/docs"
    print_info "Monitoring dashboard: http://localhost:3000"
    print_info "Database: PostgreSQL with direct connection"
    print_info "Deployment log: $LOG_FILE"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    main
fi 