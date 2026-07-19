"""seed model versions

Revision ID: 002_seed_models
Revises: 001_initial
Create Date: 2026-07-18 10:05:00.000000

"""
from typing import Sequence, Union
import uuid
from datetime import datetime

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '002_seed_models'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(f"""
        INSERT INTO model_versions (id, model_type, version_tag, artifact_path, is_production, created_at)
        VALUES 
        ('{uuid.uuid4()}', 'xgboost', 'xgboost-v1.0', 'ml/xgboost_model.joblib', true, '{datetime.now().isoformat()}'),
        ('{uuid.uuid4()}', 'lstm', 'lstm-v1.0', 'ml/lstm_problem2_live_reforecast.keras', true, '{datetime.now().isoformat()}')
    """)


def downgrade() -> None:
    op.execute("DELETE FROM model_versions WHERE version_tag IN ('xgboost-v1.0', 'lstm-v1.0')")
