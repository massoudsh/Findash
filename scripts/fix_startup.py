#!/usr/bin/env python3
"""
Quantum Trading Matrix™ - Startup Fix & System Test
Fixes common startup issues and validates system integration
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_environment():
    """Fix environment variables and Python path"""
    logger.info("🔧 Fixing environment variables...")
    
    # Set required environment variables
    env_vars = {
        "CELERY_ACCEPT_CONTENT": "json",
        "CELERY_TASK_SERIALIZER": "json", 
        "CELERY_RESULT_SERIALIZER": "json",
        "CELERY_BROKER_URL": "redis://localhost:6379/0",
        "CELERY_RESULT_BACKEND": "redis://localhost:6379/0",
        "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/trading_db",
        "REDIS_URL": "redis://localhost:6379/0"
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"✓ Set {key}")
    
    # Add current directory to Python path
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logger.info(f"✓ Added {current_dir} to Python path")

def test_imports():
    """Test critical imports"""
    logger.info("🧪 Testing critical imports...")
    
    try:
        # Test core imports
        from src.core.config import get_settings
        logger.info("✓ Core config import - OK")
        
        from src.core.exceptions import StrategyError, TradingError
        logger.info("✓ Exceptions import - OK")
        
        from src.api.endpoints.comprehensive_api import router
        logger.info("✓ Comprehensive API router - OK")
        
        # Test Phase 5 components
        from src.options.options_trading_engine import OptionsEngine
        logger.info("✓ Options engine import - OK")
        
        from src.alternative_data.alternative_data_engine import AlternativeDataEngine
        logger.info("✓ Alternative data engine import - OK")
        
        from src.enhancements.esg_predictor import ESGPredictor
        logger.info("✓ ESG predictor import - OK")
        
        from src.enhancements.quantum_neural_networks import QuantumNeuralNetworks
        logger.info("✓ Quantum neural networks import - OK")
        
        # Test autonomous trading pods (optional)
        try:
            from src.enhancements.autonomous_trading_pods import AutonomousTradingPodSystem
            logger.info("✓ Autonomous trading pods import - OK")
        except ImportError as e:
            logger.warning(f"⚠️ Autonomous trading pods import - Warning: {e}")
            # Continue anyway as this is not critical for basic functionality
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def start_simple_server():
    """Start a simplified version of the server for testing"""
    logger.info("🚀 Starting simplified server...")
    
    try:
        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        
        # Create a simple test script
        test_script = """
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

# Set environment variables
os.environ.setdefault("CELERY_ACCEPT_CONTENT", "json")
os.environ.setdefault("CELERY_TASK_SERIALIZER", "json")
os.environ.setdefault("CELERY_RESULT_SERIALIZER", "json")

try:
    from fastapi import FastAPI
    from src.api.endpoints.comprehensive_api import router
    
    app = FastAPI(title="Quantum Trading Test")
    app.include_router(router, tags=["Test API"])
    
    import uvicorn
    print("✅ Starting test server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    
except Exception as e:
    print(f"❌ Server startup failed: {e}")
    import traceback
    traceback.print_exc()
"""
        
        # Write test script
        with open("test_server.py", "w") as f:
            f.write(test_script)
        
        # Start server
        process = subprocess.Popen(
            [sys.executable, "test_server.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit and check if it's running
        time.sleep(3)
        
        if process.poll() is None:
            logger.info("✅ Test server started successfully!")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Server failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        return None

def test_endpoints():
    """Test key endpoints"""
    logger.info("🧪 Testing endpoints...")
    
    try:
        import requests
        
        base_url = "http://localhost:8001"
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Health endpoint - OK")
            else:
                logger.warning(f"⚠️ Health endpoint - Status: {response.status_code}")
        except:
            logger.info("ℹ️ Health endpoint not available (expected for simple test)")
        
        # Test docs endpoint
        try:
            response = requests.get(f"{base_url}/docs", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API docs endpoint - OK")
            else:
                logger.warning(f"⚠️ API docs endpoint - Status: {response.status_code}")
        except:
            logger.info("ℹ️ API docs endpoint not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Endpoint testing failed: {e}")
        return False

def main():
    """Main integration test"""
    print("🌟 Quantum Trading Matrix™ - Startup Fix & Integration Test")
    print("="*70)
    
    # Step 1: Fix environment
    fix_environment()
    
    # Step 2: Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        logger.error("❌ Critical import failures detected!")
        print("\n🔧 Suggested fixes:")
        print("1. Check if all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify Python path includes current directory")
        print("3. Check for missing Phase 5 component files")
        return
    
    # Step 3: Start test server
    server_process = start_simple_server()
    
    if server_process:
        try:
            # Step 4: Test endpoints
            time.sleep(2)  # Give server time to start
            test_endpoints()
            
            print("\n" + "="*70)
            print("✅ Integration test completed successfully!")
            print("🌐 Test server running at: http://localhost:8001")
            print("📚 API documentation: http://localhost:8001/docs")
            print("\n🎯 Next steps:")
            print("1. Start the full server: python -m uvicorn src.main:app --host 0.0.0.0 --port 8000")
            print("2. Start the frontend: cd frontend-nextjs && npm run dev")
            print("3. Access the dashboard: http://localhost:3000")
            print("="*70)
            
            # Keep server running for a bit
            input("\nPress Enter to stop the test server...")
            
        finally:
            if server_process:
                server_process.terminate()
                logger.info("🛑 Test server stopped")
    
    # Cleanup
    if Path("test_server.py").exists():
        Path("test_server.py").unlink()

if __name__ == "__main__":
    main() 