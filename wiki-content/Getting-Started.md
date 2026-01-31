# Getting Started

This guide will help you set up and run the Octopus Trading Platform (Findash) on your local machine.

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| Storage | 50GB SSD | 100GB+ SSD |
| OS | macOS 12+, Ubuntu 20.04+, Windows 10+ | macOS 13+, Ubuntu 22.04+ |

### Required Software

- **Node.js**: 18.x or higher
- **Python**: 3.10 or higher
- **PostgreSQL**: 14+ (optional for development)
- **Redis**: 7+ (optional for development)
- **Docker**: 24.0+ (for containerized deployment)
- **Git**: Latest version

---

## Installation Methods

### Method 1: Local Development (Recommended for Development)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/massoudsh/Findash.git
cd Findash
```

#### Step 2: Backend Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Frontend Setup

```bash
cd frontend-nextjs
npm install
cd ..
```

#### Step 4: Environment Configuration

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your configuration
# At minimum, you need:
# - SECRET_KEY (generate a random string)
# - JWT_SECRET_KEY (generate a random string)
# - DATABASE_URL (optional for dev mode)
# - REDIS_URL (optional for dev mode)
```

**Minimal .env for Development:**

```bash
# Environment
ENVIRONMENT=development
DEBUG=true

# Security (generate random strings)
SECRET_KEY=your-secret-key-here-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-here-change-in-production

# Database (optional - app runs in dev mode without it)
DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_db

# Redis (optional - app runs in dev mode without it)
REDIS_URL=redis://localhost:6379/0
```

#### Step 5: Start the Application

**Terminal 1 - Backend:**
```bash
python3 start.py --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend-nextjs
npm run dev
```

#### Step 6: Access the Platform

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |

---

### Method 2: Docker Compose (Recommended for Full Stack)

#### Core Services Only (Development)

```bash
# Start core services (API, Frontend, DB, Redis, Celery)
docker compose -f docker-compose-core.yml up -d

# View logs
docker compose -f docker-compose-core.yml logs -f
```

#### Complete Stack (Production-like)

```bash
# Start all 24 services
docker compose -f docker-compose-complete.yml up -d

# View logs
docker compose -f docker-compose-complete.yml logs -f
```

---

### Method 3: Using Makefile

```bash
# View available commands
make help

# Start development environment
make dev

# Run tests
make test

# Build Docker images
make build
```

---

## Database Setup (Optional)

If you want to use PostgreSQL with full features:

### Using Docker

```bash
# Start PostgreSQL container
docker run -d \
  --name findash-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=trading_db \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15
```

### Run Migrations

```bash
# Initialize database with Alembic
alembic upgrade head
```

---

## Redis Setup (Optional)

For full caching and pub/sub functionality:

### Using Docker

```bash
docker run -d \
  --name findash-redis \
  -p 6379:6379 \
  redis:7-alpine
```

---

## Verify Installation

### Health Check

```bash
# Check backend health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "..."}
```

### API Documentation

Visit http://localhost:8000/docs to see the interactive API documentation.

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

#### 2. Python Dependencies Fail

```bash
# Upgrade pip first
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v
```

#### 3. Node Modules Issues

```bash
# Clear node_modules and reinstall
cd frontend-nextjs
rm -rf node_modules package-lock.json
npm install
```

#### 4. Database Connection Errors

The application runs in development mode without database. If you need database:

```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Check connection string in .env
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

#### 5. Environment Variables Not Loading

```bash
# Ensure .env file exists
ls -la .env

# Check file contents
cat .env
```

---

## Next Steps

- [[Architecture]] - Understand the system architecture
- [[API Reference]] - Explore available API endpoints
- [[AI Agents]] - Learn about the 11 AI agents
- [[Configuration]] - Advanced configuration options

---

## Demo Credentials

For testing purposes, use these demo credentials:

| Field | Value |
|-------|-------|
| Email | demo@octopus.trading |
| Password | demo123 |

**Note**: Demo credentials are for local development only.
