"""
Alembic migration for advanced trading schema: users, portfolios, positions, trades, portfolio_snapshots (TimescaleDB)
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20240610_advanced_trading_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Enable TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('username', sa.String(64), unique=True, nullable=False),
        sa.Column('email', sa.String(128), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(256), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
    )

    # Portfolios table
    op.create_table(
        'portfolios',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(128), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
    )

    # Positions table
    op.create_table(
        'positions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('portfolio_id', sa.Integer, sa.ForeignKey('portfolios.id', ondelete='CASCADE'), nullable=False),
        sa.Column('symbol', sa.String(32), nullable=False),
        sa.Column('quantity', sa.Numeric, nullable=False),
        sa.Column('average_price', sa.Numeric, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
    )

    # Trades table
    op.create_table(
        'trades',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('portfolio_id', sa.Integer, sa.ForeignKey('portfolios.id', ondelete='CASCADE'), nullable=False),
        sa.Column('symbol', sa.String(32), nullable=False),
        sa.Column('trade_type', sa.String(4), nullable=False),
        sa.Column('quantity', sa.Numeric, nullable=False),
        sa.Column('price', sa.Numeric, nullable=False),
        sa.Column('trade_time', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
        sa.Column('notes', sa.Text),
        sa.CheckConstraint("trade_type IN ('BUY', 'SELL')", name='trade_type_check'),
    )

    # Portfolio snapshots (time-series, TimescaleDB hypertable)
    op.create_table(
        'portfolio_snapshots',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('portfolio_id', sa.Integer, sa.ForeignKey('portfolios.id', ondelete='CASCADE'), nullable=False),
        sa.Column('snapshot_time', sa.TIMESTAMP(), nullable=False),
        sa.Column('total_value', sa.Numeric, nullable=False),
    )
    # Convert to hypertable
    op.execute("SELECT create_hypertable('portfolio_snapshots', 'snapshot_time', if_not_exists => TRUE);")

def downgrade():
    op.drop_table('portfolio_snapshots')
    op.drop_table('trades')
    op.drop_table('positions')
    op.drop_table('portfolios')
    op.drop_table('users')
    op.execute("DROP EXTENSION IF EXISTS timescaledb;") 