#!/usr/bin/env python3
"""
Install missing dependencies for Quantum Trading Matrix™
"""

import subprocess
import sys

def install_missing():
    """Install missing packages"""
    packages = ["passlib", "scipy", "scikit-learn", "bcrypt"]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} failed: {e}")

if __name__ == "__main__":
    install_missing()
    print("🎉 Installation complete!") 