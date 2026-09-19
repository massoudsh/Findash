"""Admin panel, risk policy, and subscription tables

Creates the tables added for issues #12 (admin panel), #13 (subscriptions), and
#22 (risk policy engine). These were previously created only by
`create_tables()` at application startup, so a real deployment had no way to
build them via migrations. `audit_logs` also backs the admin audit trail.

Revision ID: admin_risk_sub_001
Revises: unify_schema_001
Create Date: 2026-09-19 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = 'admin_risk_sub_001'
down_revision = 'unify_schema_001'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('actor_user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('action', sa.String(64), nullable=False),
        sa.Column('target_type', sa.String(32), nullable=True),
        sa.Column('target_id', sa.String(64), nullable=True),
        sa.Column('detail', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now(), index=True),
    )

    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('category', sa.String(32), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('body', sa.Text(), nullable=True),
        sa.Column('is_read', sa.Boolean(), default=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now(), index=True),
    )

    op.create_table(
        'subscription_plans',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('code', sa.String(32), nullable=False, unique=True),
        sa.Column('name_fa', sa.String(128), nullable=False),
        sa.Column('price_toman', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('duration_days', sa.Integer(), nullable=False, server_default='30'),
        sa.Column('features', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.true()),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
    )

    op.create_table(
        'user_subscriptions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('plan_id', sa.Integer(), sa.ForeignKey('subscription_plans.id'), nullable=False),
        sa.Column('status', sa.String(16), nullable=False, server_default='active'),
        sa.Column('start_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        sa.Column('end_at', sa.TIMESTAMP(), nullable=False),
        sa.Column('auto_renew', sa.Boolean(), server_default=sa.false()),
        sa.Column('reminder_sent_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('last_payment_order_id', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
    )

    op.create_table(
        'risk_policies',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('max_daily_drawdown_pct', sa.Numeric(5, 2), nullable=False, server_default='5.0'),
        sa.Column('max_position_concentration_pct', sa.Numeric(5, 2), nullable=False, server_default='30.0'),
        sa.Column('action_on_breach', sa.String(16), nullable=False, server_default='alert'),
        sa.Column('enabled', sa.Boolean(), server_default=sa.true()),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.func.now()),
    )

    op.create_table(
        'risk_policy_breaches',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('rule', sa.String(64), nullable=False),
        sa.Column('value', sa.Numeric(10, 4), nullable=False),
        sa.Column('threshold', sa.Numeric(10, 4), nullable=False),
        sa.Column('action_taken', sa.String(16), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table('risk_policy_breaches')
    op.drop_table('risk_policies')
    op.drop_table('user_subscriptions')
    op.drop_table('subscription_plans')
    op.drop_table('notifications')
    op.drop_table('audit_logs')
