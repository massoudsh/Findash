"""
Database Migration Script for Phase 3 Features
Creates tables for Agent Monitoring, Wallet, Security, and Scenarios
"""

import logging
from sqlalchemy import text
from src.database.postgres_connection import init_db_connection, get_db_session, engine
from src.database.models import Base

logger = logging.getLogger(__name__)

def create_phase3_tables():
    """Create all Phase 3 database tables"""
    try:
        init_db_connection()
        
        # Import all models to register them with Base
        from src.database.models import (
            AgentStatus, AgentLog, AgentDecision,
            WalletBalance, WalletTransaction, BankAccount,
            APIKey, UserSession, TradingPermission
        )
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Phase 3 database tables created successfully")
        
        # Create indexes for better performance
        with engine.connect() as conn:
            # Agent logs index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agent_logs_timestamp 
                ON agent_logs(timestamp DESC);
            """))
            
            # Agent logs agent_id index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_id 
                ON agent_logs(agent_id);
            """))
            
            # Agent decisions index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agent_decisions_timestamp 
                ON agent_decisions(timestamp DESC);
            """))
            
            # Wallet transactions index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_wallet_transactions_timestamp 
                ON wallet_transactions(timestamp DESC);
            """))
            
            # Wallet transactions user_id index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_wallet_transactions_user_id 
                ON wallet_transactions(user_id);
            """))
            
            conn.commit()
        
        logger.info("✅ Phase 3 database indexes created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating Phase 3 tables: {e}")
        raise

if __name__ == "__main__":
    create_phase3_tables()

