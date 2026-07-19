"""add drift results and ab flags

Revision ID: 004_drift
Revises: 003_performance
Create Date: 2026-07-22 10:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '004_drift'
down_revision: Union[str, None] = '003_performance'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('drift_results',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('check_date', sa.String(), nullable=True),
    sa.Column('feature_name', sa.String(length=128), nullable=False),
    sa.Column('ks_statistic', sa.Float(), nullable=False),
    sa.Column('p_value', sa.Float(), nullable=False),
    sa.Column('drift_detected', sa.Boolean(), nullable=False),
    sa.Column('training_mean', sa.Float(), nullable=True),
    sa.Column('production_mean', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    
    op.add_column('predictions', sa.Column('is_ab_request', sa.Boolean(), nullable=True))
    op.execute("UPDATE predictions SET is_ab_request = false")
    
    op.add_column('model_versions', sa.Column('is_ab_candidate', sa.Boolean(), nullable=True))
    op.execute("UPDATE model_versions SET is_ab_candidate = false")


def downgrade() -> None:
    op.drop_column('model_versions', 'is_ab_candidate')
    op.drop_column('predictions', 'is_ab_request')
    op.drop_table('drift_results')
