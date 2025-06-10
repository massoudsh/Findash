"""
Database repositories using raw PostgreSQL queries
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import hashlib
from passlib.context import CryptContext
from .postgres_connection import get_db
import psycopg2.extras

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@dataclass
class User:
    """User data model"""
    id: str
    email: str
    username: str
    hashed_password: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    risk_tolerance: str = 'medium'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class Portfolio:
    """Portfolio data model"""
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    initial_capital: float = 100000.0
    current_value: float = 0.0
    cash_balance: float = 100000.0
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class OptionPosition:
    """Option position data model"""
    id: str
    user_id: str
    portfolio_id: str
    symbol: str
    option_type: str
    strike_price: float
    expiry_date: datetime
    quantity: int
    premium_paid: float
    current_price: Optional[float] = None
    underlying_price: float = 0.0
    implied_volatility: float = 0.2
    risk_free_rate: float = 0.05
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    order_status: str = 'pending'
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    pnl: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UserRepository:
    """Repository for user operations"""
    
    def __init__(self):
        self.db = get_db()
    
    def create_user(self, email: str, username: str, password: str, 
                   full_name: str = None, risk_tolerance: str = 'medium') -> User:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        hashed_password = pwd_context.hash(password)
        
        query = """
        INSERT INTO users (id, email, username, hashed_password, full_name, risk_tolerance)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, email, username, hashed_password, full_name, is_active, 
                  is_verified, risk_tolerance, created_at, updated_at
        """
        
        # Use a dedicated connection to ensure proper transaction handling
        with self.db.get_connection() as conn:
            with self.db.get_cursor(conn, cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                try:
                    cursor.execute(query, (user_id, email, username, hashed_password, full_name, risk_tolerance))
                    result = cursor.fetchone()
                    conn.commit()  # Explicitly commit the transaction
                    
                    return User(**dict(result)) if result else None
                    
                except Exception as e:
                    conn.rollback()
                    raise
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        query = "SELECT * FROM users WHERE id = %s"
        result = self.db.execute_query(query, (user_id,), fetch='one')
        return User(**result) if result else None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        query = "SELECT * FROM users WHERE email = %s"
        result = self.db.execute_query(query, (email,), fetch='one')
        return User(**result) if result else None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        query = "SELECT * FROM users WHERE username = %s"
        result = self.db.execute_query(query, (username,), fetch='one')
        return User(**result) if result else None
    
    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user fields"""
        if not kwargs:
            return self.get_user_by_id(user_id)
        
        # Build dynamic update query
        set_clauses = []
        params = []
        
        for field, value in kwargs.items():
            if field == 'password':
                set_clauses.append("hashed_password = %s")
                params.append(pwd_context.hash(value))
            else:
                set_clauses.append(f"{field} = %s")
                params.append(value)
        
        params.append(user_id)
        
        query = f"""
        UPDATE users 
        SET {', '.join(set_clauses)}
        WHERE id = %s
        RETURNING *
        """
        
        result = self.db.execute_query(query, tuple(params), fetch='one')
        return User(**result) if result else None
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = self.get_user_by_email(email)
        if user and self.verify_password(password, user.hashed_password):
            return user
        return None

class PortfolioRepository:
    """Repository for portfolio operations"""
    
    def __init__(self):
        self.db = get_db()
    
    def create_portfolio(self, user_id: str, name: str, description: str = None,
                        initial_capital: float = 100000.0) -> Portfolio:
        """Create a new portfolio"""
        portfolio_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO portfolios (id, user_id, name, description, initial_capital, cash_balance)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING *
        """
        
        result = self.db.execute_query(
            query, 
            (portfolio_id, user_id, name, description, initial_capital, initial_capital),
            fetch='one'
        )
        
        return Portfolio(**result) if result else None
    
    def get_portfolio_by_id(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get portfolio by ID"""
        query = "SELECT * FROM portfolios WHERE id = %s"
        result = self.db.execute_query(query, (portfolio_id,), fetch='one')
        return Portfolio(**result) if result else None
    
    def get_user_portfolios(self, user_id: str) -> List[Portfolio]:
        """Get all portfolios for a user"""
        query = "SELECT * FROM portfolios WHERE user_id = %s AND is_active = TRUE ORDER BY created_at DESC"
        results = self.db.execute_query(query, (user_id,), fetch='all')
        return [Portfolio(**row) for row in results] if results else []
    
    def update_portfolio_value(self, portfolio_id: str, current_value: float, 
                              cash_balance: float = None) -> Optional[Portfolio]:
        """Update portfolio value and cash balance"""
        if cash_balance is not None:
            query = """
            UPDATE portfolios 
            SET current_value = %s, cash_balance = %s
            WHERE id = %s
            RETURNING *
            """
            params = (current_value, cash_balance, portfolio_id)
        else:
            query = """
            UPDATE portfolios 
            SET current_value = %s
            WHERE id = %s
            RETURNING *
            """
            params = (current_value, portfolio_id)
        
        result = self.db.execute_query(query, params, fetch='one')
        return Portfolio(**result) if result else None

class OptionPositionRepository:
    """Repository for option position operations"""
    
    def __init__(self):
        self.db = get_db()
    
    def create_position(self, user_id: str, portfolio_id: str, symbol: str,
                       option_type: str, strike_price: float, expiry_date: datetime,
                       quantity: int, premium_paid: float, underlying_price: float,
                       implied_volatility: float = 0.2, risk_free_rate: float = 0.05) -> OptionPosition:
        """Create a new option position"""
        position_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO option_positions (
            id, user_id, portfolio_id, symbol, option_type, strike_price,
            expiry_date, quantity, premium_paid, underlying_price,
            implied_volatility, risk_free_rate
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """
        
        result = self.db.execute_query(
            query,
            (position_id, user_id, portfolio_id, symbol, option_type, strike_price,
             expiry_date, quantity, premium_paid, underlying_price,
             implied_volatility, risk_free_rate),
            fetch='one'
        )
        
        return OptionPosition(**result) if result else None
    
    def get_position_by_id(self, position_id: str) -> Optional[OptionPosition]:
        """Get position by ID"""
        query = "SELECT * FROM option_positions WHERE id = %s"
        result = self.db.execute_query(query, (position_id,), fetch='one')
        return OptionPosition(**result) if result else None
    
    def get_portfolio_positions(self, portfolio_id: str) -> List[OptionPosition]:
        """Get all positions for a portfolio"""
        query = """
        SELECT * FROM option_positions 
        WHERE portfolio_id = %s 
        ORDER BY created_at DESC
        """
        results = self.db.execute_query(query, (portfolio_id,), fetch='all')
        return [OptionPosition(**row) for row in results] if results else []
    
    def get_user_positions(self, user_id: str) -> List[OptionPosition]:
        """Get all positions for a user"""
        query = """
        SELECT * FROM option_positions 
        WHERE user_id = %s 
        ORDER BY created_at DESC
        """
        results = self.db.execute_query(query, (user_id,), fetch='all')
        return [OptionPosition(**row) for row in results] if results else []
    
    def update_position_greeks(self, position_id: str, delta: float, gamma: float,
                              theta: float, vega: float, rho: float,
                              current_price: float = None) -> Optional[OptionPosition]:
        """Update position Greeks and current price"""
        if current_price is not None:
            query = """
            UPDATE option_positions 
            SET delta = %s, gamma = %s, theta = %s, vega = %s, rho = %s, current_price = %s
            WHERE id = %s
            RETURNING *
            """
            params = (delta, gamma, theta, vega, rho, current_price, position_id)
        else:
            query = """
            UPDATE option_positions 
            SET delta = %s, gamma = %s, theta = %s, vega = %s, rho = %s
            WHERE id = %s
            RETURNING *
            """
            params = (delta, gamma, theta, vega, rho, position_id)
        
        result = self.db.execute_query(query, params, fetch='one')
        return OptionPosition(**result) if result else None
    
    def update_position_pnl(self, position_id: str, pnl: float) -> Optional[OptionPosition]:
        """Update position P&L"""
        query = """
        UPDATE option_positions 
        SET pnl = %s
        WHERE id = %s
        RETURNING *
        """
        
        result = self.db.execute_query(query, (pnl, position_id), fetch='one')
        return OptionPosition(**result) if result else None
    
    def close_position(self, position_id: str) -> Optional[OptionPosition]:
        """Close a position"""
        query = """
        UPDATE option_positions 
        SET order_status = 'closed', closed_at = CURRENT_TIMESTAMP
        WHERE id = %s
        RETURNING *
        """
        
        result = self.db.execute_query(query, (position_id,), fetch='one')
        return OptionPosition(**result) if result else None
    
    def delete_position(self, position_id: str) -> bool:
        """Delete a position"""
        query = "DELETE FROM option_positions WHERE id = %s"
        self.db.execute_query(query, (position_id,))
        return True

class MarketDataRepository:
    """Repository for market data operations"""
    
    def __init__(self):
        self.db = get_db()
    
    def save_market_data(self, symbol: str, open_price: float, high_price: float,
                        low_price: float, close_price: float, volume: int,
                        timestamp: datetime, implied_volatility: float = None,
                        beta: float = None, market_cap: int = None) -> str:
        """Save market data"""
        data_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO market_data (
            id, symbol, open_price, high_price, low_price, close_price,
            volume, implied_volatility, beta, market_cap, timestamp
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.db.execute_query(
            query,
            (data_id, symbol, open_price, high_price, low_price, close_price,
             volume, implied_volatility, beta, market_cap, timestamp)
        )
        
        return data_id
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price for a symbol"""
        query = """
        SELECT close_price, timestamp 
        FROM market_data 
        WHERE symbol = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        return self.db.execute_query(query, (symbol,), fetch='one')
    
    def get_price_history(self, symbol: str, start_date: datetime, 
                         end_date: datetime = None) -> List[Dict]:
        """Get price history for a symbol"""
        if end_date:
            query = """
            SELECT * FROM market_data 
            WHERE symbol = %s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC
            """
            params = (symbol, start_date, end_date)
        else:
            query = """
            SELECT * FROM market_data 
            WHERE symbol = %s AND timestamp >= %s
            ORDER BY timestamp ASC
            """
            params = (symbol, start_date)
        
        return self.db.execute_query(query, params, fetch='all')

class APIKeyRepository:
    """Repository for API key operations"""
    
    def __init__(self):
        self.db = get_db()
    
    def create_api_key(self, user_id: str, key_name: str, api_key: str,
                      can_read: bool = True, can_write: bool = False,
                      can_trade: bool = False, rate_limit: int = 1000,
                      expires_at: datetime = None) -> str:
        """Create a new API key"""
        key_id = str(uuid.uuid4())
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        query = """
        INSERT INTO api_keys (
            id, user_id, key_name, key_hash, can_read, can_write,
            can_trade, rate_limit, expires_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.db.execute_query(
            query,
            (key_id, user_id, key_name, key_hash, can_read, can_write,
             can_trade, rate_limit, expires_at)
        )
        
        return key_id
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verify API key and return key info"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        query = """
        SELECT ak.*, u.email, u.username
        FROM api_keys ak
        JOIN users u ON ak.user_id = u.id
        WHERE ak.key_hash = %s AND ak.is_active = TRUE
        AND (ak.expires_at IS NULL OR ak.expires_at > CURRENT_TIMESTAMP)
        """
        
        result = self.db.execute_query(query, (key_hash,), fetch='one')
        
        if result:
            # Update usage
            self.db.execute_query(
                "UPDATE api_keys SET last_used = CURRENT_TIMESTAMP, usage_count = usage_count + 1 WHERE id = %s",
                (result['id'],)
            )
        
        return result

class AuditLogRepository:
    """Repository for audit log operations"""
    
    def __init__(self):
        self.db = get_db()
    
    def log_action(self, user_id: str, action: str, resource_type: str = None,
                  resource_id: str = None, old_values: Dict = None,
                  new_values: Dict = None, ip_address: str = None,
                  user_agent: str = None) -> str:
        """Log an action"""
        log_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO audit_logs (
            id, user_id, action, resource_type, resource_id,
            old_values, new_values, ip_address, user_agent
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.db.execute_query(
            query,
            (log_id, user_id, action, resource_type, resource_id,
             json.dumps(old_values) if old_values else None,
             json.dumps(new_values) if new_values else None,
             ip_address, user_agent)
        )
        
        return log_id 