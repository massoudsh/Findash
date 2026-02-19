# =====================================================================
# Octopus Trading Platform - Production Makefile
# Secure, maintainable commands for development and deployment
# =====================================================================

.PHONY: help install setup dev build test deploy clean security-check lint

# Default target
.DEFAULT_GOAL := help

# Configuration
PROJECT_NAME := octopus-trading-platform
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose
NODE := npm

# Environment setup
VENV_NAME := .venv
REQUIREMENTS := requirements/requirements.txt
DEV_REQUIREMENTS := requirements/requirements-dev.txt

# =====================================================================
# HELP & DOCUMENTATION
# =====================================================================

help: ## Show this help message
	@echo "🐙 Octopus Trading Platform - Development Commands"
	@echo "=================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =====================================================================
# ENVIRONMENT SETUP
# =====================================================================

install: ## Install Python dependencies
	@echo "📦 Installing Python dependencies..."
	$(PIP) install -r $(REQUIREMENTS)
	$(PIP) install -r $(DEV_REQUIREMENTS)

setup-env: ## Setup environment from template
	@echo "🔧 Setting up environment configuration..."
	@if [ ! -f .env ]; then \
		cp config/env.example .env; \
		echo "✅ Created .env from template. Please update with your values."; \
		echo "🔐 Generate secure secrets using: python -c \"import secrets; print(secrets.token_urlsafe(32))\""; \
	else \
		echo "⚠️  .env already exists. Use 'make reset-env' to recreate."; \
	fi

setup: setup-env install ## Complete project setup
	@echo "🚀 Setting up Octopus Trading Platform..."
	@$(PYTHON) -m venv $(VENV_NAME) || echo "Virtual environment already exists"
	@echo "✅ Setup complete! Activate venv with: source $(VENV_NAME)/bin/activate"

# =====================================================================
# DEVELOPMENT COMMANDS
# =====================================================================

dev: ## Start development server
	@echo "🚀 Starting FastAPI development server..."
	$(PYTHON) -m uvicorn src.main_refactored:app --host 0.0.0.0 --port 8000 --reload

dev-frontend: ## Start frontend development server
	@echo "🎨 Starting Next.js frontend..."
	cd frontend-nextjs && $(NODE) run dev

dev-all: ## Start all development services
	@echo "🚀 Starting all development services..."
	@echo "Starting backend in background..."
	@$(PYTHON) -m uvicorn src.main_refactored:app --host 0.0.0.0 --port 8000 --reload &
	@echo "Starting frontend..."
	@cd frontend-nextjs && $(NODE) run dev

celery-worker: ## Start Celery worker
	@echo "⚙️ Starting Celery worker..."
	celery -A src.core.celery_app worker -l info

celery-beat: ## Start Celery beat scheduler
	@echo "⏰ Starting Celery beat scheduler..."
	celery -A src.core.celery_app beat -l info

# =====================================================================
# DOCKER COMMANDS
# =====================================================================

build: ## Build Docker images
	@echo "🐳 Building Docker images..."
	$(DOCKER_COMPOSE) build

up: ## Start all services with Docker
	@echo "🚀 Starting all services with Docker..."
	$(DOCKER_COMPOSE) up -d

down: ## Stop all Docker services
	@echo "🛑 Stopping all Docker services..."
	$(DOCKER_COMPOSE) down

logs: ## View Docker logs
	@echo "📋 Viewing Docker logs..."
	$(DOCKER_COMPOSE) logs -f

restart: down up ## Restart all Docker services

# =====================================================================
# DATABASE COMMANDS
# =====================================================================

db-up: ## Start only database services
	@echo "🗄️ Starting database services..."
	$(DOCKER_COMPOSE) up -d db redis

db-migrate: ## Run database migrations
	@echo "🔄 Running database migrations..."
	$(PYTHON) -m alembic upgrade head

db-reset: ## Reset database (CAUTION: Deletes all data)
	@echo "⚠️ WARNING: This will delete all data!"
	@read -p "Type 'YES' to confirm: " confirm; \
	if [ "$$confirm" = "YES" ]; then \
		$(DOCKER_COMPOSE) down -v; \
		$(DOCKER_COMPOSE) up -d db redis; \
		sleep 5; \
		$(PYTHON) -m alembic upgrade head; \
		echo "✅ Database reset complete"; \
	else \
		echo "❌ Database reset cancelled"; \
	fi

# =====================================================================
# TESTING & QUALITY ASSURANCE
# =====================================================================

test: ## Run all tests
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	@echo "⚡ Running fast tests..."
	$(PYTHON) -m pytest tests/ -v -x

lint: ## Run code linting
	@echo "🔍 Running code linting..."
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m black --check src/ tests/
	$(PYTHON) -m isort --check-only src/ tests/

format: ## Format code
	@echo "✨ Formatting code..."
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

security-check: ## Run security checks
	@echo "🔒 Running security checks..."
	$(PYTHON) -m bandit -r src/ -ll
	$(PYTHON) -m safety check

# =====================================================================
# MAINTENANCE & CLEANUP
# =====================================================================

clean: ## Clean up temporary files
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.log" -delete

clean-docker: ## Clean Docker resources
	@echo "🐳 Cleaning Docker resources..."
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker image prune -f
	docker volume prune -f

reset-env: ## Reset environment file
	@echo "🔄 Resetting environment file..."
	@rm -f .env
	@cp config/env.example .env
	@echo "✅ Environment file reset. Please update with your values."

# =====================================================================
# DEPLOYMENT COMMANDS
# =====================================================================

deploy-staging: ## Deploy to staging environment
	@echo "🚀 Deploying to staging..."
	@echo "ENVIRONMENT=staging" > .env.staging
	@cat config/env.example >> .env.staging
	$(DOCKER_COMPOSE) -f docker-compose.yml --env-file .env.staging up -d

deploy-prod: ## Deploy to production environment
	@echo "🚀 Deploying to production..."
	@echo "⚠️ WARNING: Production deployment!"
	@read -p "Type 'DEPLOY' to confirm: " confirm; \
	if [ "$$confirm" = "DEPLOY" ]; then \
		echo "ENVIRONMENT=production" > .env.production; \
		cat config/env.example >> .env.production; \
		$(DOCKER_COMPOSE) -f docker-compose.yml --env-file .env.production up -d; \
		echo "✅ Production deployment complete"; \
	else \
		echo "❌ Production deployment cancelled"; \
	fi

# =====================================================================
# MONITORING & HEALTH
# =====================================================================

health: ## Check service health
	@echo "🏥 Checking service health..."
	@curl -f http://localhost:8000/health || echo "❌ API not responding"
	@curl -f http://localhost:3000 || echo "❌ Frontend not responding"
	@curl -f http://localhost:9090 || echo "❌ Prometheus not responding"
	@curl -f http://localhost:3001 || echo "❌ Grafana not responding"

status: ## Show service status
	@echo "📊 Service Status:"
	@$(DOCKER_COMPOSE) ps

# =====================================================================
# UTILITY COMMANDS
# =====================================================================

generate-secrets: ## Generate secure secrets
	@echo "🔐 Generating secure secrets..."
	@echo "SECRET_KEY=$$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
	@echo "JWT_SECRET_KEY=$$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

backup-data: ## Backup important data
	@echo "💾 Creating data backup..."
	@mkdir -p backups
	@$(DOCKER_COMPOSE) exec db pg_dump -U postgres trading_db > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backup created in backups/"

# =====================================================================
# DEVELOPMENT UTILITIES
# =====================================================================

shell: ## Open Python shell with app context
	@echo "🐍 Opening Python shell..."
	$(PYTHON) -c "from src.main_refactored import app; import IPython; IPython.start_ipython(argv=[])"

db-shell: ## Open database shell
	@echo "🗄️ Opening database shell..."
	$(DOCKER_COMPOSE) exec db psql -U postgres trading_db

redis-shell: ## Open Redis shell
	@echo "📦 Opening Redis shell..."
	$(DOCKER_COMPOSE) exec redis redis-cli

docs: ## Generate API documentation
	@echo "📚 Generating API documentation..."
	@echo "API docs available at: http://localhost:8000/docs"
	@echo "ReDoc available at: http://localhost:8000/redoc" 