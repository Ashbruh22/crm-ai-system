from ml.celery_app import celery_app
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import os
import json
from app.models.db import ModelVersion, PerformanceLog
import redis

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://crm_ai_app:crm_ai_db_password_dev@db:5432/crm_ai")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

sync_engine = create_engine(DATABASE_URL.replace("postgresql+asyncpg", "postgresql"))
redis_client = redis.Redis.from_url(REDIS_URL)

@celery_app.task(bind=True, max_retries=3)
def auto_promote_candidate(self, candidate_version_tag: str):
    """
    After 7-day A/B window:
    1. Check performance of candidate vs production
    2. Promote if candidate beats production
    3. Retire if it fails
    """
    with Session(sync_engine) as session:
        candidate_model = session.query(ModelVersion).filter_by(version_tag=candidate_version_tag).first()
        prod_model = session.query(ModelVersion).filter_by(is_production=True, model_type="xgboost").first()
        
        if not candidate_model or not candidate_model.is_ab_candidate:
            return "Candidate not found or not active A/B candidate."
            
        if not prod_model:
            return "No production model to compare against."

        # Fetch recent performance logs
        cand_perf = session.query(PerformanceLog).filter_by(model_version=candidate_version_tag).order_by(PerformanceLog.evaluation_date.desc()).first()
        prod_perf = session.query(PerformanceLog).filter_by(model_version=prod_model.version_tag).order_by(PerformanceLog.evaluation_date.desc()).first()
        
        # If no A/B evaluation data exists, we'll fall back to validation AUC from training
        cand_score = cand_perf.auc_roc if cand_perf else candidate_model.val_auc_roc
        prod_score = prod_perf.auc_roc if prod_perf else prod_model.val_auc_roc
        
        if not cand_score or not prod_score:
            return "Missing scores for evaluation, aborting promotion."
            
        if cand_score >= prod_score:
            # Promote
            prod_model.is_production = False
            candidate_model.is_production = True
            candidate_model.is_ab_candidate = False
            
            import mlflow
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("model_promotion")
            with mlflow.start_run():
                mlflow.log_param("promoted_version", candidate_version_tag)
                mlflow.log_param("retired_version", prod_model.version_tag)
                mlflow.log_metric("improvement", cand_score - prod_score)
            
            # Hot Swap via Redis Pub/Sub
            redis_client.publish('model_updates', json.dumps({
                'action': 'reload',
                'model_path': candidate_model.artifact_path,
                'version_tag': candidate_model.version_tag
            }))
            
            # Clear predictions cache
            for key in redis_client.scan_iter("pred:*"):
                redis_client.delete(key)
                
            session.commit()
            return f"Promoted {candidate_version_tag} over {prod_model.version_tag} (Score: {cand_score:.3f} >= {prod_score:.3f})"
        else:
            # Reject
            candidate_model.is_ab_candidate = False
            session.commit()
            return f"Rejected {candidate_version_tag}. Score {cand_score:.3f} < {prod_score:.3f}"
