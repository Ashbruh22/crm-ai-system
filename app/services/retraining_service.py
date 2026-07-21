from datetime import datetime
import json
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ml.tasks.retrain import trigger_retraining
from app.models.db import ModelVersion

class RetrainingService:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def trigger_manual_retrain(self) -> dict:
        """Called by POST /admin/retrain. Returns Celery task ID."""
        task = trigger_retraining.delay(reason='manual_admin')
        return {"task_id": task.id, "status": "queued"}

    def get_retrain_status(self, task_id: str) -> dict:
        """Check Celery task status by ID."""
        from celery.result import AsyncResult
        result = AsyncResult(task_id)
        return {"task_id": task_id, "status": result.status, "result": str(result.result) if result.ready() else None}

    async def get_ab_test_status(self, db: AsyncSession) -> dict:
        """Return current A/B split config from model_versions table."""
        # Query for is_ab_candidate=True model
        result = await db.execute(select(ModelVersion).filter(ModelVersion.is_ab_candidate == True))
        candidate = result.scalars().first()
        
        prod_result = await db.execute(select(ModelVersion).filter(ModelVersion.is_production == True, ModelVersion.model_type == "xgboost"))
        production = prod_result.scalars().first()
        
        if not candidate:
            return {
                "active": False,
                "production_version": production.version_tag if production else None,
                "message": "No active A/B test"
            }
            
        return {
            "active": True,
            "candidate_version": candidate.version_tag,
            "production_version": production.version_tag if production else None,
            "traffic_split": "20% Candidate / 80% Production",
            "candidate_auc": candidate.val_auc_roc
        }

    async def hot_swap_model(self, new_model_path: str, version_tag: str):
        """
        Signal running Gunicorn workers to reload the production model.
        Uses Redis pub/sub: publish 'model_update' event.
        """
        await self.redis_client.publish('model_updates', json.dumps({
            'action': 'reload',
            'model_path': new_model_path,
            'version_tag': version_tag,
            'timestamp': datetime.utcnow().isoformat()
        }))
