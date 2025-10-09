#!/usr/bin/env python3
"""
🚀 Octopus Trading Platform - Quick Start Script
Automated setup for rapid deployment using free services only

This script will:
1. Check and install dependencies
2. Set up environment with secure defaults
3. Initialize database with PostgreSQL
4. Start Redis for caching
5. Launch the trading platform
6. Provide access URLs and demo credentials

Usage: python quick_start.py
"""

import os
import sys
import subprocess
import time
import logging
import secrets
import docker
import psutil
from pathlib import Path
from typing import Dict, List, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickStartManager:
    """Manages the quick start process for Octopus Trading Platform"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / "env.local"
        self.services = {}
        self.required_ports = [8000, 5432, 6379, 3000]
        
    def run(self):
        """Execute the complete quick start process"""
        try:
            print("🐙 OCTOPUS TRADING PLATFORM - QUICK START")
            print("=" * 50)
            
            # Step 1: Check system requirements
            self.check_system_requirements()
            
            # Step 2: Generate secure environment
            self.setup_environment()
            
            # Step 3: Check and start required services
            self.setup_services()
            
            # Step 4: Initialize database
            self.setup_database()
            
            # Step 5: Install Python dependencies
            self.install_dependencies()
            
            # Step 6: Start the application
            self.start_application()
            
            # Step 7: Provide access information
            self.show_success_info()
            
        except Exception as e:
            logger.error(f"❌ Quick start failed: {e}")
            self.cleanup_on_error()
            sys.exit(1)
    
    def check_system_requirements(self):
        """Check if system has required dependencies"""
        print("\n📋 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception("Python 3.8+ is required")
        print("✅ Python version OK")
        
        # Check Docker
        try:
            docker_client = docker.from_env()
            docker_client.ping()
            print("✅ Docker is available")
        except Exception:
            logger.warning("⚠️ Docker not available - will use local services")
        
        # Check available ports
        busy_ports = []
        for port in self.required_ports:
            if self.is_port_busy(port):
                busy_ports.append(port)
        
        if busy_ports:
            print(f"⚠️ Ports {busy_ports} are busy. Will attempt to use alternative ports.")
    
    def is_port_busy(self, port: int) -> bool:
        """Check if a port is in use"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False
    
    def setup_environment(self):
        """Set up secure environment configuration"""
        print("\n🔐 Setting up secure environment...")
        
        if self.env_file.exists():
            print("✅ Environment file already exists")
            return
        
        # Generate secure secrets
        secret_key = secrets.token_urlsafe(32)
        jwt_secret = secrets.token_urlsafe(32)
        db_password = secrets.token_urlsafe(16)
        
        env_content = f"""# OCTOPUS TRADING PLATFORM - AUTO-GENERATED SECURE CONFIGURATION
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# SECURITY: Auto-generated secure secrets
SECRET_KEY={secret_key}
JWT_SECRET_KEY={jwt_secret}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=true

# Database Configuration
DATABASE_URL=postgresql://postgres:{db_password}@localhost:5432/trading_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=postgres
DB_PASSWORD={db_password}
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# External API Keys - FREE TIER READY
ALPHA_VANTAGE_API_KEY=demo
YAHOO_FINANCE_API_KEY=free
NEWS_API_KEY=demo
FINNHUB_API_KEY=demo

# Authentication & Security
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
BCRYPT_ROUNDS=12

# Rate Limiting (Free tier friendly)
RATE_LIMIT_PER_MINUTE=30
RATE_LIMIT_BURST=10
MAX_LOGIN_ATTEMPTS=5

# Demo Account Passwords (Change these!)
DEMO_ADMIN_PASSWORD=SecureAdmin2025!
DEMO_TRADER_PASSWORD=TraderPro2025!
DEMO_USER_PASSWORD=DemoUser2025!

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002
CORS_ALLOW_CREDENTIALS=true

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_SECRET={secret_key}
NEXTAUTH_URL=http://localhost:3000

# File Storage
DATA_DIR=./data
LOGS_DIR=./logs
MODELS_DIR=./models
UPLOAD_DIR=./uploads

# Trading Configuration
DEFAULT_PORTFOLIO_VALUE=100000
MAX_POSITION_SIZE=0.1
RISK_FREE_RATE=0.05
DEFAULT_VOLATILITY=0.2
"""
        
        with open(self.env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ Secure environment configuration created")
        
        # Store credentials for later display
        self.db_password = db_password
    
    def setup_services(self):
        """Set up required services (PostgreSQL, Redis)"""
        print("\n🛠️ Setting up required services...")
        
        try:
            docker_client = docker.from_env()
            self.setup_docker_services(docker_client)
        except Exception:
            print("⚠️ Docker not available, please install PostgreSQL and Redis manually")
            print("   PostgreSQL: https://www.postgresql.org/download/")
            print("   Redis: https://redis.io/download")
            input("   Press Enter when PostgreSQL and Redis are running...")
    
    def setup_docker_services(self, docker_client):
        """Set up services using Docker"""
        
        # Start PostgreSQL
        try:
            postgres_container = docker_client.containers.get("octopus-postgres")
            if postgres_container.status != "running":
                postgres_container.start()
            print("✅ PostgreSQL container already exists and running")
        except docker.errors.NotFound:
            print("📦 Starting PostgreSQL container...")
            postgres_container = docker_client.containers.run(
                "postgres:15-alpine",
                name="octopus-postgres",
                environment={
                    "POSTGRES_DB": "trading_db",
                    "POSTGRES_USER": "postgres",
                    "POSTGRES_PASSWORD": self.db_password
                },
                ports={"5432/tcp": 5432},
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Wait for PostgreSQL to be ready
            print("⏳ Waiting for PostgreSQL to be ready...")
            for i in range(30):
                try:
                    result = postgres_container.exec_run("pg_isready -U postgres")
                    if result.exit_code == 0:
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise Exception("PostgreSQL failed to start")
            
            print("✅ PostgreSQL container started")
        
        # Start Redis
        try:
            redis_container = docker_client.containers.get("octopus-redis")
            if redis_container.status != "running":
                redis_container.start()
            print("✅ Redis container already exists and running")
        except docker.errors.NotFound:
            print("📦 Starting Redis container...")
            redis_container = docker_client.containers.run(
                "redis:7-alpine",
                name="octopus-redis",
                ports={"6379/tcp": 6379},
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            print("✅ Redis container started")
    
    def setup_database(self):
        """Initialize the database schema"""
        print("\n🗄️ Setting up database...")
        
        try:
            # Run database initialization script
            init_script = self.project_root / "scripts" / "init-db.sql"
            if init_script.exists():
                cmd = [
                    "psql",
                    f"postgresql://postgres:{self.db_password}@localhost:5432/trading_db",
                    "-f", str(init_script)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print("✅ Database schema initialized")
            else:
                print("⚠️ Database init script not found, will create tables on first run")
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"Database setup issue: {e}")
            print("⚠️ Database setup had issues, but continuing...")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("\n📦 Installing Python dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True, capture_output=True)
                print("✅ Python dependencies installed")
            except subprocess.CalledProcessError:
                print("⚠️ Some dependencies failed to install, continuing anyway...")
        else:
            print("⚠️ requirements.txt not found")
    
    def start_application(self):
        """Start the trading platform"""
        print("\n🚀 Starting Octopus Trading Platform...")
        
        # Change to project directory
        os.chdir(self.project_root)
        
        # Load environment
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        # Start FastAPI backend
        print("🔧 Starting FastAPI backend...")
        try:
            backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "src.main_refactored:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], cwd=self.project_root)
            
            # Wait a moment for backend to start
            time.sleep(3)
            
            # Check if backend is running
            if backend_process.poll() is None:
                print("✅ FastAPI backend started")
                self.services['backend'] = backend_process
            else:
                raise Exception("Backend failed to start")
                
        except Exception as e:
            logger.error(f"Failed to start backend: {e}")
            raise
        
        # Start Next.js frontend
        frontend_dir = self.project_root / "frontend-nextjs"
        if frontend_dir.exists():
            print("🎨 Starting Next.js frontend...")
            try:
                # Install npm dependencies if needed
                if not (frontend_dir / "node_modules").exists():
                    subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
                
                frontend_process = subprocess.Popen([
                    "npm", "run", "dev"
                ], cwd=frontend_dir)
                
                time.sleep(3)
                
                if frontend_process.poll() is None:
                    print("✅ Next.js frontend started")
                    self.services['frontend'] = frontend_process
                else:
                    print("⚠️ Frontend failed to start, continuing with backend only...")
                    
            except Exception as e:
                print(f"⚠️ Frontend startup issue: {e}")
    
    def show_success_info(self):
        """Display success information and access details"""
        print("\n🎉 SUCCESS! Octopus Trading Platform is running!")
        print("=" * 60)
        
        print("\n📍 ACCESS URLS:")
        print("   • API Backend:     http://localhost:8000")
        print("   • API Documentation: http://localhost:8000/docs")
        print("   • Frontend:        http://localhost:3000")
        print("   • Health Check:    http://localhost:8000/health")
        
        print("\n🔑 DEMO CREDENTIALS:")
        print("   • Admin:     admin@octopus.trading / SecureAdmin2025!")
        print("   • Trader:    trader@octopus.trading / TraderPro2025!")
        print("   • Demo User: demo@octopus.trading / DemoUser2025!")
        
        print("\n🛠️ SERVICES RUNNING:")
        print("   • PostgreSQL:      localhost:5432")
        print("   • Redis:           localhost:6379")
        print("   • FastAPI:         localhost:8000")
        if 'frontend' in self.services:
            print("   • Next.js:         localhost:3000")
        
        print("\n📊 FREE DATA SOURCES CONFIGURED:")
        print("   • Yahoo Finance (Primary)")
        print("   • CoinGecko (Crypto)")
        print("   • Finnhub Free Tier")
        print("   • Alpha Vantage Free Tier")
        
        print("\n⭐ NEXT STEPS:")
        print("   1. Visit http://localhost:3000 to access the platform")
        print("   2. Log in with demo credentials above")
        print("   3. Get FREE API keys from:")
        print("      - Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        print("      - Finnhub: https://finnhub.io/")
        print("   4. Update API keys in env.local file")
        
        print("\n🔄 TO STOP THE PLATFORM:")
        print("   Press Ctrl+C to stop all services")
        
        # Keep services running
        try:
            print("\n⏳ Platform is running... Press Ctrl+C to stop")
            while True:
                time.sleep(1)
                
                # Check if services are still running
                for service_name, process in self.services.items():
                    if process.poll() is not None:
                        print(f"⚠️ {service_name} stopped unexpectedly")
                        
        except KeyboardInterrupt:
            print("\n🛑 Stopping Octopus Trading Platform...")
            self.stop_services()
    
    def stop_services(self):
        """Stop all running services"""
        for service_name, process in self.services.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {service_name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔥 {service_name} force stopped")
            except Exception as e:
                print(f"⚠️ Error stopping {service_name}: {e}")
    
    def cleanup_on_error(self):
        """Clean up on error"""
        print("\n🧹 Cleaning up...")
        self.stop_services()

if __name__ == "__main__":
    manager = QuickStartManager()
    manager.run() 