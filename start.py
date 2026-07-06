#!/usr/bin/env python3
"""
Octopus Trading Platform™ - Production Startup Script
Secure, unified entry point for the trading platform
"""

import asyncio
import os
import sys
import logging
import signal
import argparse
from pathlib import Path
from typing import Optional

# Load environment variables from .env file first
from dotenv import load_dotenv
load_dotenv(override=True)

import uvicorn
from src.core.config import get_settings
from src.core.logging_config import setup_logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class OctopusStartupManager:
    """Manages the startup and shutdown of the Octopus Trading Platform"""
    
    def __init__(self):
        self.settings = get_settings()
        self.server_process: Optional[asyncio.subprocess.Process] = None
        self.shutdown_event = asyncio.Event()
        
    def validate_environment(self) -> bool:
        """Validate that all required environment variables are set"""
        is_dev = os.getenv("ENVIRONMENT", "development").lower() == "development"
        # In production, require these; in development use defaults so backend can start
        required_vars = (
            [] if is_dev
            else ["SECRET_KEY", "JWT_SECRET_KEY", "DATABASE_URL", "REDIS_URL"]
        )
        recommended_vars = [
            "SECRET_KEY",
            "JWT_SECRET_KEY",
            "DATABASE_URL",
            "REDIS_URL",
            "ALPHA_VANTAGE_API_KEY",
            "CORS_ORIGINS",
            "ENVIRONMENT",
        ]
        missing_vars = [v for v in required_vars if not os.getenv(v)]
        missing_recommended = [v for v in recommended_vars if not os.getenv(v)]

        for var in ["SECRET_KEY", "JWT_SECRET_KEY"]:
            val = os.getenv(var)
            if val and len(val) < 32:
                logger.warning(f"⚠️ {var} should be at least 32 characters for security")

        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            logger.error("Please check your .env file or environment configuration")
            return False

        if missing_recommended and is_dev:
            logger.warning(
                "⚠️ Some env vars not set (using dev defaults): %s. Set them in .env for full features.",
                [v for v in missing_recommended if v not in ("ALPHA_VANTAGE_API_KEY", "CORS_ORIGINS", "ENVIRONMENT")],
            )
            
        # Validate format of critical variables
        try:
            from urllib.parse import urlparse
            
            # Validate DATABASE_URL format
            db_url = os.getenv("DATABASE_URL")
            parsed_db = urlparse(db_url)
            if not parsed_db.scheme.startswith('postgresql'):
                logger.warning(f"⚠️ DATABASE_URL should use postgresql:// scheme, got: {parsed_db.scheme}")
            
            # Validate REDIS_URL format
            redis_url = os.getenv("REDIS_URL")
            parsed_redis = urlparse(redis_url)
            if not parsed_redis.scheme.startswith('redis'):
                logger.warning(f"⚠️ REDIS_URL should use redis:// scheme, got: {parsed_redis.scheme}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not validate URL formats: {e}")
            
        logger.info("✅ Environment validation completed")
        return True
        
    def check_dependencies(self) -> bool:
        """Check if all required services are available"""
        logger.info("🔍 Checking dependencies...")
        
        is_development = self.settings.environment == "development"
        all_ok = True
        
        # Check Redis
        try:
            import redis
            redis_client = redis.Redis.from_url(self.settings.redis.url)
            redis_client.ping()
            logger.info("✅ Redis connection successful")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.warning("Some features may be limited without Redis")
            if not is_development:
                all_ok = False
            
        # Check Database
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.settings.database.url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")
        except Exception as e:
            if is_development:
                logger.warning(f"⚠️ Database connection failed: {e}")
                logger.warning("⚠️ Running in development mode - app will start but database features will be limited")
                logger.warning("⚠️ To fix: Start PostgreSQL and Redis services")
            else:
                logger.error(f"❌ Database connection failed: {e}")
                all_ok = False
            
        return all_ok
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(sig, frame):
            logger.info(f"📡 Received signal {sig}, initiating shutdown...")
            self.shutdown_event.set()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def start_server(self, host: str = None, port: int = None, 
                          workers: int = None, reload: bool = None):
        """Start the FastAPI server"""
        
        # Use provided values or defaults from settings
        config = uvicorn.Config(
            app="src.main_refactored:app",
            host=host or self.settings.api.host,
            port=port or self.settings.api.port,
            workers=workers or self.settings.api.workers if not reload else 1,
            reload=reload if reload is not None else self.settings.api.reload,
            log_level=self.settings.log_level.lower(),
            access_log=True,
            server_header=False,  # Security: Don't expose server info
            date_header=False,    # Security: Don't expose date
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"🚀 Starting Octopus Trading Platform on {config.host}:{config.port}")
        logger.info(f"📊 Environment: {self.settings.environment}")
        logger.info(f"🔧 Workers: {config.workers}")
        logger.info(f"🔄 Reload: {config.reload}")
        
        try:
            await server.serve()
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}")
            raise
            
    async def run(self, **kwargs):
        """Main startup routine"""
        
        logger.info("🐙 Octopus Trading Platform™ - Starting...")
        
        # Validate environment
        if not self.validate_environment():
            sys.exit(1)
            
        # Check dependencies
        dependencies_ok = self.check_dependencies()
        if not dependencies_ok:
            if self.settings.environment == "development":
                logger.warning("⚠️ Some dependencies are unavailable, but continuing in development mode")
                logger.warning("⚠️ Please start PostgreSQL and Redis for full functionality")
            else:
                logger.error("❌ Dependency check failed")
                sys.exit(1)
            
        # Setup signal handlers
        self.setup_signal_handlers()
        
        try:
            # Start the server
            await self.start_server(**kwargs)
            
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            sys.exit(1)
        finally:
            logger.info("🛑 Octopus Trading Platform shutdown complete")

def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Octopus Trading Platform™ - Production Startup Script"
    )
    
    parser.add_argument(
        "--host", 
        default=None,
        help="Host to bind to (default: from config)"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=None,
        help="Port to bind to (default: from config)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int,
        default=None,
        help="Number of worker processes (default: from config)"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--no-reload", 
        action="store_true",
        help="Disable auto-reload for production"
    )
    
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Handle reload flags
    reload = None
    if args.reload:
        reload = True
    elif args.no_reload:
        reload = False
        
    try:
        startup_manager = OctopusStartupManager()
        
        if args.validate_only:
            if startup_manager.validate_environment() and startup_manager.check_dependencies():
                print("✅ Configuration validation successful")
                sys.exit(0)
            else:
                print("❌ Configuration validation failed")
                sys.exit(1)
        
        # Run the application
        asyncio.run(startup_manager.run(
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=reload
        ))
        
    except KeyboardInterrupt:
        logger.info("🛑 Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 