#!/usr/bin/env python3
"""
Fix remaining dependencies for Quantum Trading Matrix™
"""

import subprocess
import sys

def install_packages():
    """Install missing visualization packages"""
    packages = [
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0", 
        "plotly>=5.17.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} failed: {e}")

def test_imports():
    """Test if we can import the required modules"""
    test_modules = [
        "matplotlib",
        "seaborn", 
        "plotly",
        "psycopg2",
        "fastapi",
        "pandas",
        "numpy"
    ]
    
    print("\n🧪 Testing imports...")
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - not available")

if __name__ == "__main__":
    print("🔧 Installing missing dependencies...")
    install_packages()
    test_imports()
    print("🎉 Dependencies check complete!") 