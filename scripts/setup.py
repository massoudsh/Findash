#!/usr/bin/env python3
"""
Setup script for Quantum Trading Matrix™
Initializes the development environment and performs necessary setup tasks
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import secrets


def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def create_env_file():
    """Create .env file from template"""
    print("🔄 Creating .env file...")
    
    env_example = Path("env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file already exists. Skipping creation.")
        return True
    
    if not env_example.exists():
        print("❌ env.example file not found")
        return False
    
    # Read template
    with open(env_example, 'r') as f:
        content = f.read()
    
    # Generate secure keys
    secret_key = secrets.token_urlsafe(32)
    jwt_secret_key = secrets.token_urlsafe(32)
    
    # Replace placeholders
    content = content.replace("your-super-secret-key-change-in-production", secret_key)
    content = content.replace("your-jwt-secret-key", jwt_secret_key)
    
    # Write .env file
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ .env file created with secure keys")
    return True


def setup_directories():
    """Create necessary directories"""
    print("🔄 Creating project directories...")
    
    directories = [
        "data",
        "logs",
        "models",
        "uploads",
        "src/alembic/versions"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("🔄 Installing Python dependencies...")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  You might want to activate a virtual environment first")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Install main dependencies
    if not run_command("pip install -r requirements/requirements.txt", "Installing main dependencies"):
        return False
    
    # Install development dependencies
    if not run_command("pip install -r requirements/requirements-dev.txt", "Installing development dependencies"):
        return False
    
    return True


def setup_database():
    """Set up database with migrations"""
    print("🔄 Setting up database...")
    
    # Check if PostgreSQL is running
    if not run_command("pg_isready -h localhost -p 5432", "Checking PostgreSQL connection", check=False):
        print("⚠️  PostgreSQL doesn't seem to be running locally")
        print("   Please ensure PostgreSQL is installed and running")
        print("   You can also use Docker: docker-compose up -d db")
        return False
    
    # Initialize Alembic
    if not run_command("python -m alembic upgrade head", "Running database migrations"):
        print("⚠️  Database migration failed. This might be normal for first setup.")
        
        # Try to create initial migration
        if not run_command("python -m alembic revision --autogenerate -m 'Initial schema'", "Creating initial migration"):
            print("❌ Failed to create initial migration")
            return False
        
        # Try migration again
        if not run_command("python -m alembic upgrade head", "Running migrations again"):
            print("❌ Database migration still failed")
            return False
    
    print("✅ Database setup completed")
    return True


def setup_pre_commit():
    """Set up pre-commit hooks"""
    print("🔄 Setting up pre-commit hooks...")
    
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("⚠️  Pre-commit hook installation failed")
        return False
    
    print("✅ Pre-commit hooks installed")
    return True


def run_tests():
    """Run basic tests to verify setup"""
    print("🔄 Running basic tests...")
    
    if not run_command("python -m pytest tests/test_auth.py -v", "Running authentication tests"):
        print("⚠️  Some tests failed. This might be expected during initial setup.")
        return False
    
    print("✅ Tests completed successfully")
    return True


def create_sample_config():
    """Create sample configuration files"""
    print("🔄 Creating sample configuration files...")
    
    # Create sample API testing file
    api_test_config = {
        "base_url": "http://localhost:8000",
        "test_user": {
            "email": "demo@quantumtrading.com",
            "password": "demo123"
        },
        "endpoints": {
            "login": "/api/auth/login",
            "register": "/api/auth/register",
            "profile": "/api/auth/me",
            "health": "/health"
        }
    }
    
    with open("config/api_test.json", "w") as f:
        json.dump(api_test_config, f, indent=2)
    
    # Create sample environment files for different stages
    environments = ["development", "staging", "production"]
    
    for env in environments:
        env_dir = Path(f"config/{env}")
        env_dir.mkdir(parents=True, exist_ok=True)
        
        env_config = {
            "environment": env,
            "debug": env == "development",
            "log_level": "DEBUG" if env == "development" else "INFO",
            "database": {
                "pool_size": 5 if env == "development" else 20,
                "max_overflow": 10 if env == "development" else 30
            }
        }
        
        with open(env_dir / "config.json", "w") as f:
            json.dump(env_config, f, indent=2)
    
    print("✅ Sample configuration files created")
    return True


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Review and update the .env file with your specific configuration")
    print("2. Start the application:")
    print("   uvicorn src.main:app --reload")
    print("3. Visit the API documentation:")
    print("   http://localhost:8000/docs")
    print("4. Test authentication with demo user:")
    print("   Email: demo@quantumtrading.com")
    print("   Password: demo123")
    print("\n🔧 Development Commands:")
    print("• Run tests: pytest")
    print("• Format code: black .")
    print("• Sort imports: isort .")
    print("• Type checking: mypy .")
    print("• Database migration: alembic revision --autogenerate -m 'description'")
    print("• Apply migrations: alembic upgrade head")
    print("\n🐳 Docker Commands:")
    print("• Start services: docker-compose up -d")
    print("• View logs: docker-compose logs -f")
    print("• Stop services: docker-compose down")
    print("\n📖 Documentation:")
    print("• API Docs: http://localhost:8000/docs")
    print("• ReDoc: http://localhost:8000/redoc")
    print("• Grafana: http://localhost:3001 (admin/admin123)")
    print("• Prometheus: http://localhost:9090")


def main():
    """Main setup function"""
    print("🚀 Quantum Trading Matrix™ Setup")
    print("=" * 40)
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    steps = [
        ("Creating environment file", create_env_file),
        ("Setting up directories", setup_directories),
        ("Installing dependencies", install_dependencies),
        ("Setting up database", setup_database),
        ("Setting up pre-commit hooks", setup_pre_commit),
        ("Creating sample configurations", create_sample_config),
        ("Running basic tests", run_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        if not step_function():
            failed_steps.append(step_name)
    
    if failed_steps:
        print(f"\n⚠️  Setup completed with some issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nYou may need to address these manually.")
    else:
        print_next_steps()
    
    return len(failed_steps) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 