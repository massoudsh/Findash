"""Unify SQLAlchemy Django schema compatibility

Revision ID: unify_schema_001
Revises: 20240610_advanced_trading_schema
Create Date: 2024-12-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'unify_schema_001'
down_revision = '20240610_advanced_trading_schema'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade to unified schema compatible with both SQLAlchemy and Django"""
    
    # Add missing columns to users table
    op.add_column('users', sa.Column('phone', sa.String(20), nullable=True))
    op.add_column('users', sa.Column('is_verified', sa.Boolean(), default=False))
    op.add_column('users', sa.Column('risk_tolerance', sa.String(20), default='moderate'))
    op.add_column('users', sa.Column('is_active', sa.Boolean(), default=True))
    op.add_column('users', sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.func.now()))
    
    # Add missing columns to portfolios table
    op.add_column('portfolios', sa.Column('initial_cash', sa.Numeric(15, 2), default=10000.00))
    op.add_column('portfolios', sa.Column('current_cash', sa.Numeric(15, 2), default=10000.00))
    op.add_column('portfolios', sa.Column('total_value', sa.Numeric(15, 2), default=0.00))
    op.add_column('portfolios', sa.Column('is_active', sa.Boolean(), default=True))
    op.add_column('portfolios', sa.Column('risk_level', sa.String(20), default='moderate'))
    op.add_column('portfolios', sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.func.now()))
    
    # Add missing columns to positions table
    op.add_column('positions', sa.Column('current_price', sa.Numeric(15, 6), default=0.000000))
    op.add_column('positions', sa.Column('market_value', sa.Numeric(15, 2), default=0.00))
    op.add_column('positions', sa.Column('unrealized_pnl', sa.Numeric(15, 2), default=0.00))
    op.add_column('positions', sa.Column('position_type', sa.String(10), default='long'))
    op.add_column('positions', sa.Column('is_active', sa.Boolean(), default=True))
    
    # Modify existing columns in positions table for precision
    op.alter_column('positions', 'quantity', type_=sa.Numeric(15, 6))
    op.alter_column('positions', 'average_price', type_=sa.Numeric(15, 6))
    
    # Add unique constraint for portfolio-symbol combination
    op.create_unique_constraint('unique_portfolio_symbol', 'positions', ['portfolio_id', 'symbol'])
    
    # Add missing columns to trades table
    op.add_column('trades', sa.Column('total_amount', sa.Numeric(15, 2), nullable=False))
    op.add_column('trades', sa.Column('fees', sa.Numeric(10, 2), default=0.00))
    op.add_column('trades', sa.Column('settlement_date', sa.TIMESTAMP(), nullable=True))
    op.add_column('trades', sa.Column('status', sa.String(20), default='pending'))
    op.add_column('trades', sa.Column('order_id', sa.String(100), nullable=True))
    
    # Modify existing columns in trades table for precision
    op.alter_column('trades', 'quantity', type_=sa.Numeric(15, 6))
    op.alter_column('trades', 'price', type_=sa.Numeric(15, 6))
    
    # Rename trade_time to trade_date for consistency
    op.alter_column('trades', 'trade_time', new_column_name='trade_date')
    
    # Modify portfolio_snapshots table
    op.alter_column('portfolio_snapshots', 'snapshot_time', new_column_name='snapshot_date')
    op.alter_column('portfolio_snapshots', 'total_value', type_=sa.Numeric(15, 2))
    op.add_column('portfolio_snapshots', sa.Column('cash_value', sa.Numeric(15, 2), default=0.00))
    op.add_column('portfolio_snapshots', sa.Column('positions_value', sa.Numeric(15, 2), default=0.00))
    op.add_column('portfolio_snapshots', sa.Column('daily_pnl', sa.Numeric(15, 2), default=0.00))
    op.add_column('portfolio_snapshots', sa.Column('daily_pnl_percent', sa.Numeric(8, 4), default=0.0000))
    
    # Add unique constraint for portfolio-snapshot_date combination
    op.create_unique_constraint('unique_portfolio_snapshot', 'portfolio_snapshots', ['portfolio_id', 'snapshot_date'])
    
    # Create new risk_metrics table
    op.create_table('risk_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('value_at_risk_1d', sa.Numeric(15, 2), nullable=True),
        sa.Column('value_at_risk_1w', sa.Numeric(15, 2), nullable=True),
        sa.Column('value_at_risk_1m', sa.Numeric(15, 2), nullable=True),
        sa.Column('sharpe_ratio', sa.Numeric(8, 4), nullable=True),
        sa.Column('beta', sa.Numeric(8, 4), nullable=True),
        sa.Column('volatility', sa.Numeric(8, 4), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(8, 4), nullable=True),
        sa.Column('concentration_risk', sa.String(20), default='medium'),
        sa.Column('last_updated', sa.TIMESTAMP(), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('portfolio_id')
    )
    
    # Add indexes for better performance
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_portfolios_user_active', 'portfolios', ['user_id', 'is_active'])
    op.create_index('idx_positions_symbol', 'positions', ['symbol'])
    op.create_index('idx_positions_portfolio_symbol_active', 'positions', ['portfolio_id', 'symbol', 'is_active'])
    op.create_index('idx_trades_portfolio_symbol', 'trades', ['portfolio_id', 'symbol'])
    op.create_index('idx_trades_date', 'trades', ['trade_date'])
    op.create_index('idx_trades_status', 'trades', ['status'])
    op.create_index('idx_snapshots_portfolio_date', 'portfolio_snapshots', ['portfolio_id', 'snapshot_date'])


def downgrade():
    """Downgrade from unified schema"""
    
    # Drop indexes
    op.drop_index('idx_snapshots_portfolio_date', 'portfolio_snapshots')
    op.drop_index('idx_trades_status', 'trades')
    op.drop_index('idx_trades_date', 'trades')
    op.drop_index('idx_trades_portfolio_symbol', 'trades')
    op.drop_index('idx_positions_portfolio_symbol_active', 'positions')
    op.drop_index('idx_positions_symbol', 'positions')
    op.drop_index('idx_portfolios_user_active', 'portfolios')
    op.drop_index('idx_users_email', 'users')
    
    # Drop risk_metrics table
    op.drop_table('risk_metrics')
    
    # Remove constraints
    op.drop_constraint('unique_portfolio_snapshot', 'portfolio_snapshots')
    op.drop_constraint('unique_portfolio_symbol', 'positions')
    
    # Revert portfolio_snapshots changes
    op.drop_column('portfolio_snapshots', 'daily_pnl_percent')
    op.drop_column('portfolio_snapshots', 'daily_pnl')
    op.drop_column('portfolio_snapshots', 'positions_value')
    op.drop_column('portfolio_snapshots', 'cash_value')
    op.alter_column('portfolio_snapshots', 'snapshot_date', new_column_name='snapshot_time')
    
    # Revert trades changes
    op.alter_column('trades', 'trade_date', new_column_name='trade_time')
    op.drop_column('trades', 'order_id')
    op.drop_column('trades', 'status')
    op.drop_column('trades', 'settlement_date')
    op.drop_column('trades', 'fees')
    op.drop_column('trades', 'total_amount')
    
    # Revert positions changes
    op.drop_column('positions', 'is_active')
    op.drop_column('positions', 'position_type')
    op.drop_column('positions', 'unrealized_pnl')
    op.drop_column('positions', 'market_value')
    op.drop_column('positions', 'current_price')
    
    # Revert portfolios changes
    op.drop_column('portfolios', 'updated_at')
    op.drop_column('portfolios', 'risk_level')
    op.drop_column('portfolios', 'is_active')
    op.drop_column('portfolios', 'total_value')
    op.drop_column('portfolios', 'current_cash')
    op.drop_column('portfolios', 'initial_cash')
    
    # Revert users changes
    op.drop_column('users', 'updated_at')
    op.drop_column('users', 'is_active')
    op.drop_column('users', 'risk_tolerance')
    op.drop_column('users', 'is_verified')
    op.drop_column('users', 'phone') 