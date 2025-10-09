#!/bin/bash

# Octopus Trading Platform - Service Startup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐙 Octopus Trading Platform - Service Manager${NC}"
echo "=================================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
}

# Function to create required directories
create_directories() {
    echo -e "${YELLOW}📁 Creating required directories...${NC}"
    mkdir -p ./data ./logs ./models
    echo -e "${GREEN}✅ Directories created${NC}"
}

# Function to check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
        if [ -f env.example ]; then
            cp env.example .env
            echo -e "${GREEN}✅ Created .env from template${NC}"
            echo -e "${YELLOW}🔧 Please edit .env file with your configuration${NC}"
        else
            echo -e "${RED}❌ No env.example found. Please create .env file manually.${NC}"
        fi
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Choose deployment option:"
    echo "1) 🚀 Core Services (Recommended for development)"
    echo "   - API, Frontend, Database, Redis, Celery, Basic Monitoring"
    echo ""
    echo "2) 🏢 Complete Enterprise Stack"
    echo "   - All services including Kong, Kafka, Elasticsearch, Keycloak, etc."
    echo ""
    echo "3) 🛑 Stop All Services"
    echo ""
    echo "4) 🧹 Clean Up (Remove containers and volumes)"
    echo ""
    echo "5) 📊 View Service Status"
    echo ""
    echo "6) 🔍 View Logs"
    echo ""
    echo "0) Exit"
    echo ""
}

# Start core services
start_core() {
    echo -e "${GREEN}🚀 Starting Core Services...${NC}"
    docker-compose -f docker-compose-core.yml up -d
    echo ""
    echo -e "${GREEN}✅ Core services started!${NC}"
    echo -e "${BLUE}📍 Access points:${NC}"
    echo "  • Frontend: http://localhost:3000"
    echo "  • API: http://localhost:8010"
    echo "  • Grafana: http://localhost:3001 (admin/admin)"
    echo "  • Prometheus: http://localhost:9090"
}

# Start complete stack
start_complete() {
    echo -e "${GREEN}🏢 Starting Complete Enterprise Stack...${NC}"
    echo -e "${YELLOW}⚠️  This may take several minutes on first run...${NC}"
    docker-compose -f docker-compose-complete.yml up -d
    echo ""
    echo -e "${GREEN}✅ Complete stack started!${NC}"
    echo -e "${BLUE}📍 Access points:${NC}"
    echo "  • Frontend: http://localhost:3000"
    echo "  • Kong API Gateway: http://localhost:8000"
    echo "  • Kong Admin: http://localhost:8001"
    echo "  • Konga (Kong UI): http://localhost:1337"
    echo "  • Direct API: http://localhost:8010"
    echo "  • Grafana: http://localhost:3001"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Keycloak: http://localhost:8080"
    echo "  • Kafka UI: http://localhost:8082"
    echo "  • Jaeger: http://localhost:16686"
    echo "  • Elasticsearch: http://localhost:9200"
    echo "  • Kibana: http://localhost:5601"
}

# Stop services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    docker-compose -f docker-compose-core.yml down 2>/dev/null || true
    docker-compose -f docker-compose-complete.yml down 2>/dev/null || true
    echo -e "${GREEN}✅ All services stopped${NC}"
}

# Clean up
cleanup() {
    echo -e "${RED}🧹 Cleaning up containers and volumes...${NC}"
    echo -e "${YELLOW}⚠️  This will remove all data! Are you sure? (y/N)${NC}"
    read -r confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        docker-compose -f docker-compose-core.yml down -v --remove-orphans 2>/dev/null || true
        docker-compose -f docker-compose-complete.yml down -v --remove-orphans 2>/dev/null || true
        docker system prune -f
        echo -e "${GREEN}✅ Cleanup completed${NC}"
    else
        echo -e "${BLUE}ℹ️  Cleanup cancelled${NC}"
    fi
}

# View status
view_status() {
    echo -e "${BLUE}📊 Service Status:${NC}"
    echo "===================="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(octopus|NAME)" || echo "No Octopus services running"
}

# View logs
view_logs() {
    echo -e "${BLUE}🔍 Available services for logs:${NC}"
    services=$(docker ps --format "{{.Names}}" | grep octopus | head -10)
    if [ -z "$services" ]; then
        echo "No Octopus services running"
        return
    fi
    
    echo "$services" | nl
    echo ""
    echo "Enter service number (or 'all' for all services):"
    read -r choice
    
    if [ "$choice" = "all" ]; then
        echo -e "${BLUE}🔍 Showing logs for all services (last 50 lines each):${NC}"
        for service in $services; do
            echo -e "\n${YELLOW}=== $service ===${NC}"
            docker logs --tail 50 "$service"
        done
    else
        service=$(echo "$services" | sed -n "${choice}p")
        if [ -n "$service" ]; then
            echo -e "${BLUE}🔍 Showing logs for $service:${NC}"
            docker logs -f "$service"
        else
            echo -e "${RED}❌ Invalid selection${NC}"
        fi
    fi
}

# Main script execution
main() {
    check_docker
    create_directories
    check_env
    
    while true; do
        show_menu
        read -r choice
        
        case $choice in
            1)
                start_core
                ;;
            2)
                start_complete
                ;;
            3)
                stop_services
                ;;
            4)
                cleanup
                ;;
            5)
                view_status
                ;;
            6)
                view_logs
                ;;
            0)
                echo -e "${BLUE}👋 Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Invalid option${NC}"
                ;;
        esac
        
        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    done
}

# Run main function
main 