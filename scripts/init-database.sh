#!/bin/bash

# 🗄️ Database Initialization Script - Octopus Trading Platform™
# This script initializes the database with schema and seed data

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-trading_db}"
DB_USER="${DB_USER:-octopus_app}"
DB_PASSWORD="${DB_PASSWORD:-secure_password}"
DB_ADMIN_USER="${DB_ADMIN_USER:-postgres}"
DB_ADMIN_PASSWORD="${DB_ADMIN_PASSWORD:-postgres}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATABASE_DIR="$PROJECT_ROOT/database"

echo -e "${BLUE}🗄️ Initializing Octopus Trading Platform Database${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to check if PostgreSQL is available
check_postgresql() {
    echo -e "${BLUE}🔍 Checking PostgreSQL availability...${NC}"
    
    if ! command -v psql > /dev/null 2>&1; then
        echo -e "${RED}❌ PostgreSQL client (psql) not found${NC}"
        echo -e "${YELLOW}💡 Please install PostgreSQL client tools${NC}"
        exit 1
    fi
    
    # Test connection
    if PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d postgres -c "SELECT 1;" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PostgreSQL is accessible${NC}"
    else
        echo -e "${RED}❌ Cannot connect to PostgreSQL${NC}"
        echo -e "${YELLOW}Details: Host=$DB_HOST, Port=$DB_PORT, User=$DB_ADMIN_USER${NC}"
        exit 1
    fi
    echo ""
}

# Function to create database and user if they don't exist
setup_database() {
    echo -e "${BLUE}🏗️ Setting up database and user...${NC}"
    
    # Check if database exists
    DB_EXISTS=$(PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME';" || echo "")
    
    if [[ -z "$DB_EXISTS" ]]; then
        echo -e "${GREEN}🆕 Creating database: $DB_NAME${NC}"
        PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d postgres -c "CREATE DATABASE $DB_NAME;"
    else
        echo -e "${YELLOW}🔄 Database $DB_NAME already exists${NC}"
    fi
    
    # Check if user exists
    USER_EXISTS=$(PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER';" || echo "")
    
    if [[ -z "$USER_EXISTS" ]]; then
        echo -e "${GREEN}🆕 Creating user: $DB_USER${NC}"
        PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
        PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
        PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;"
        PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_USER;"
    else
        echo -e "${YELLOW}🔄 User $DB_USER already exists${NC}"
    fi
    echo ""
}

# Function to install extensions
install_extensions() {
    echo -e "${BLUE}🔌 Installing required extensions...${NC}"
    
    # Install TimescaleDB extension (if available)
    if PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ TimescaleDB extension installed${NC}"
    else
        echo -e "${YELLOW}⚠️ TimescaleDB extension not available (will use regular PostgreSQL)${NC}"
    fi
    
    # Install other required extensions
    PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
    echo -e "${GREEN}✅ uuid-ossp extension installed${NC}"
    
    PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";"
    echo -e "${GREEN}✅ pgcrypto extension installed${NC}"
    
    echo ""
}

# Function to run schema files
run_schema() {
    echo -e "${BLUE}📋 Running database schema...${NC}"
    
    if [[ -f "$DATABASE_DIR/schemas/01_initial_schema.sql" ]]; then
        echo -e "${GREEN}🔧 Applying initial schema...${NC}"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$DATABASE_DIR/schemas/01_initial_schema.sql"
        echo -e "${GREEN}✅ Initial schema applied successfully${NC}"
    else
        echo -e "${RED}❌ Schema file not found: $DATABASE_DIR/schemas/01_initial_schema.sql${NC}"
        exit 1
    fi
    echo ""
}

# Function to run migration files
run_migrations() {
    echo -e "${BLUE}🔄 Running database migrations...${NC}"
    
    # Create migrations tracking table if it doesn't exist
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) UNIQUE NOT NULL,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    "
    
    # Run migration files in order
    migration_count=0
    if [[ -d "$DATABASE_DIR/migrations" ]]; then
        for migration_file in "$DATABASE_DIR/migrations"/*.sql; do
            if [[ -f "$migration_file" ]]; then
                migration_name=$(basename "$migration_file")
                
                # Check if migration already applied
                already_applied=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT 1 FROM schema_migrations WHERE migration_name='$migration_name';" || echo "")
                
                if [[ -z "$already_applied" ]]; then
                    echo -e "${GREEN}🔧 Applying migration: $migration_name${NC}"
                    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$migration_file"
                    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "INSERT INTO schema_migrations (migration_name) VALUES ('$migration_name');"
                    ((migration_count++))
                else
                    echo -e "${YELLOW}⏭️ Migration already applied: $migration_name${NC}"
                fi
            fi
        done
    fi
    
    if [[ $migration_count -eq 0 ]]; then
        echo -e "${YELLOW}ℹ️ No new migrations to apply${NC}"
    else
        echo -e "${GREEN}✅ Applied $migration_count migrations${NC}"
    fi
    echo ""
}

# Function to run seed data
run_seeds() {
    echo -e "${BLUE}🌱 Running seed data...${NC}"
    
    seed_count=0
    if [[ -d "$DATABASE_DIR/seeds" ]]; then
        for seed_file in "$DATABASE_DIR/seeds"/*.sql; do
            if [[ -f "$seed_file" ]]; then
                seed_name=$(basename "$seed_file")
                echo -e "${GREEN}🌱 Running seed: $seed_name${NC}"
                PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$seed_file"
                ((seed_count++))
            fi
        done
    fi
    
    if [[ $seed_count -eq 0 ]]; then
        echo -e "${YELLOW}ℹ️ No seed files found${NC}"
    else
        echo -e "${GREEN}✅ Applied $seed_count seed files${NC}"
    fi
    echo ""
}

# Function to verify database setup
verify_setup() {
    echo -e "${BLUE}🔍 Verifying database setup...${NC}"
    
    # Check if key tables exist
    tables=("users" "portfolios" "orders" "positions" "market_quotes")
    missing_tables=0
    
    for table in "${tables[@]}"; do
        table_exists=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT 1 FROM information_schema.tables WHERE table_name='$table';" || echo "")
        
        if [[ -n "$table_exists" ]]; then
            echo -e "${GREEN}✅ Table exists: $table${NC}"
        else
            echo -e "${RED}❌ Table missing: $table${NC}"
            ((missing_tables++))
        fi
    done
    
    if [[ $missing_tables -eq 0 ]]; then
        echo -e "${GREEN}✅ All core tables verified${NC}"
    else
        echo -e "${RED}❌ $missing_tables tables are missing${NC}"
        exit 1
    fi
    
    # Check for TimescaleDB hypertables
    hypertable_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM timescaledb_information.hypertables;" 2>/dev/null || echo "0")
    
    if [[ "$hypertable_count" -gt 0 ]]; then
        echo -e "${GREEN}✅ TimescaleDB hypertables: $hypertable_count${NC}"
    else
        echo -e "${YELLOW}⚠️ No TimescaleDB hypertables found (using regular tables)${NC}"
    fi
    
    # Check user count
    user_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM users;" || echo "0")
    echo -e "${GREEN}✅ Users in database: $user_count${NC}"
    
    # Check portfolio count
    portfolio_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM portfolios;" || echo "0")
    echo -e "${GREEN}✅ Portfolios in database: $portfolio_count${NC}"
    
    echo ""
}

# Function to create database backup
create_backup() {
    if [[ "${CREATE_BACKUP:-false}" == "true" ]]; then
        echo -e "${BLUE}💾 Creating database backup...${NC}"
        
        backup_dir="$PROJECT_ROOT/backups"
        mkdir -p "$backup_dir"
        
        backup_file="$backup_dir/octopus_db_backup_$(date +%Y%m%d_%H%M%S).sql"
        
        if PGPASSWORD="$DB_PASSWORD" pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > "$backup_file"; then
            echo -e "${GREEN}✅ Backup created: $backup_file${NC}"
        else
            echo -e "${YELLOW}⚠️ Backup creation failed${NC}"
        fi
        echo ""
    fi
}

# Function to show connection info
show_connection_info() {
    echo -e "${BLUE}📋 Database Connection Information${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo -e "Host: $DB_HOST"
    echo -e "Port: $DB_PORT"
    echo -e "Database: $DB_NAME"
    echo -e "User: $DB_USER"
    echo -e "Connection URL: postgresql://$DB_USER:****@$DB_HOST:$DB_PORT/$DB_NAME"
    echo ""
    
    echo -e "${BLUE}🔧 Quick Test Commands:${NC}"
    echo -e "psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
    echo -e "psql \"postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME\""
    echo ""
}

# Main execution
echo -e "${BLUE}Starting database initialization...${NC}"
echo ""

# Check prerequisites
check_postgresql

# Setup database and user
setup_database

# Install extensions
install_extensions

# Run schema
run_schema

# Run migrations
run_migrations

# Run seed data
run_seeds

# Verify setup
verify_setup

# Create backup if requested
create_backup

# Show connection info
show_connection_info

echo -e "${GREEN}🎉 Database initialization completed successfully!${NC}"
echo ""

echo -e "${BLUE}📝 Next Steps:${NC}"
echo -e "  1. Update your application's DATABASE_URL environment variable"
echo -e "  2. Test the connection using the provided commands"
echo -e "  3. Run the application and verify everything works"
echo -e "  4. Set up regular database backups"
echo -e "  5. Configure monitoring for the database"

echo ""
echo -e "${BLUE}💡 Troubleshooting:${NC}"
echo -e "  • Check PostgreSQL logs: docker logs <postgres_container>"
echo -e "  • Verify network connectivity: telnet $DB_HOST $DB_PORT"
echo -e "  • Check permissions: GRANT ALL ON DATABASE $DB_NAME TO $DB_USER;"
echo -e "  • Verify TimescaleDB: SELECT * FROM timescaledb_information.hypertables;"

echo ""
echo -e "${GREEN}✅ Database initialization script completed!${NC}" 