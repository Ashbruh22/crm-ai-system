"""add performance log

Revision ID: 003_performance
Revises: 002_seed_models
Create Date: 2026-07-22 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '003_performance'
down_revision: Union[str, None] = '002_seed_models'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('performance_log',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('model_version', sa.String(length=32), nullable=False),
    sa.Column('evaluation_date', sa.String(), nullable=True),
    sa.Column('window_days', sa.Integer(), nullable=True),
    sa.Column('sample_size', sa.Integer(), nullable=True),
    sa.Column('accuracy', sa.Float(), nullable=True),
    sa.Column('precision_score', sa.Float(), nullable=True),
    sa.Column('recall_score', sa.Float(), nullable=True),
    sa.Column('f1_score', sa.Float(), nullable=True),
    sa.Column('auc_roc', sa.Float(), nullable=True),
    sa.Column('degradation_vs_baseline', sa.Float(), nullable=True),
    sa.Column('retraining_triggered', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('performance_log')
