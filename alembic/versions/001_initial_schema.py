"""initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2026-07-18 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # opportunities
    op.create_table('opportunities',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('opportunity_id', sa.String(length=64), nullable=False),
        sa.Column('sales_agent', sa.String(length=128), nullable=False),
        sa.Column('product', sa.String(length=128), nullable=False),
        sa.Column('engage_date', sa.String(length=32), nullable=False),
        sa.Column('crm_source', sa.String(length=32), nullable=True),
        sa.Column('created_at', sa.String(), nullable=True),
        sa.Column('updated_at', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('opportunity_id')
    )
    
    # predictions
    op.create_table('predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('opportunity_id', sa.String(length=64), nullable=False),
        sa.Column('model_version', sa.String(length=32), nullable=False),
        sa.Column('win_probability', sa.Float(), nullable=False),
        sa.Column('outcome_prediction', sa.String(length=8), nullable=False),
        sa.Column('confidence_lower', sa.Float(), nullable=True),
        sa.Column('confidence_upper', sa.Float(), nullable=True),
        sa.Column('predicted_days_remaining', sa.Integer(), nullable=True),
        sa.Column('shap_values', sa.JSON(), nullable=False),
        sa.Column('nl_explanation', sa.Text(), nullable=True),
        sa.Column('inference_latency_ms', sa.Float(), nullable=True),
        sa.Column('cached', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['opportunity_id'], ['opportunities.opportunity_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # recommendations
    op.create_table('recommendations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('opportunity_id', sa.String(length=64), nullable=False),
        sa.Column('action', sa.Text(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column('expected_impact', sa.Text(), nullable=True),
        sa.Column('priority', sa.String(length=8), nullable=False),
        sa.Column('urgency_score', sa.Float(), nullable=False),
        sa.Column('adopted_at', sa.String(), nullable=True),
        sa.Column('outcome_delta', sa.Float(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['prediction_id'], ['predictions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # model_versions
    op.create_table('model_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_type', sa.String(length=32), nullable=False),
        sa.Column('version_tag', sa.String(length=32), nullable=False),
        sa.Column('val_accuracy', sa.Float(), nullable=True),
        sa.Column('val_auc_roc', sa.Float(), nullable=True),
        sa.Column('val_mae', sa.Float(), nullable=True),
        sa.Column('artifact_path', sa.Text(), nullable=False),
        sa.Column('is_production', sa.Boolean(), nullable=True),
        sa.Column('deployed_at', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('version_tag')
    )


def downgrade() -> None:
    op.drop_table('model_versions')
    op.drop_table('recommendations')
    op.drop_table('predictions')
    op.drop_table('opportunities')
