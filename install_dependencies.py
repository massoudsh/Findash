#!/usr/bin/env python3
"""
Install required dependencies for Quantum Trading Matrix™
This script installs the PostgreSQL migration dependencies
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True, check=True)
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Install all required packages"""
    print("🚀 Installing Quantum Trading Matrix™ Dependencies")
    print("=" * 50)
    
    # Core dependencies for PostgreSQL migration
    packages = [
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0", 
        "yfinance>=0.2.24",
        "pandas>=2.1.4",
        "numpy>=1.24.4",
        "scipy>=1.11.4",
        "passlib[bcrypt]>=1.7.4",
        "python-jose[cryptography]>=3.3.0",
        "pydantic[email]>=2.5.1",
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.3.2"
    ]
    
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 50)
    if failed_packages:
        print(f"❌ Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nTry installing manually:")
        for package in failed_packages:
            print(f"  python3 -m pip install {package}")
    else:
        print("✅ All dependencies installed successfully!")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        ("psycopg2", "PostgreSQL driver"),
        ("fastapi", "FastAPI framework"),
        ("pandas", "Data analysis"),
        ("numpy", "Numerical computing"),
        ("yfinance", "Market data"),
        ("passlib", "Password hashing")
    ]
    
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} ({description})")
        except ImportError:
            print(f"❌ {module} ({description}) - not available")
    
    print("\n🎉 Dependency installation complete!")
    print("\nNext steps:")
    print("1. Initialize database: python3 database/postgres_init.py")
    print("2. Run example: python3 examples/postgresql_usage.py")
    print("3. Start API: python3 main.py")

if __name__ == "__main__":
    main() 