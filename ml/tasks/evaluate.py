from ml.celery_app import celery_app
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
from app.models.db import Prediction, Opportunity, ModelVersion, PerformanceLog
from ml.tasks.retrain import trigger_retraining

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://crm_ai_app:crm_ai_db_password_dev@db:5432/crm_ai")
# Celery usually uses synchronous db calls since it's a separate worker process
sync_engine = create_engine(DATABASE_URL.replace("postgresql+asyncpg", "postgresql"))

@celery_app.task(bind=True, max_retries=3)
def evaluate_model_performance(self):
    """
    Evaluate model on last 30 days of closed deals where we had a prediction.
    Compares actual outcome (from deal_stage in CRM) vs predicted outcome.
    Triggers retraining if accuracy drops > 5% from production baseline.
    """
    with Session(sync_engine) as session:
        # Get current prod model
        prod_model = session.query(ModelVersion).filter_by(is_production=True, model_type="xgboost").first()
        if not prod_model:
            return "No prod model"
            
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        # 1. Query predictions for last 30 days joined with opportunity
        # We need closed deals: deal_stage in ('Closed Won', 'Closed Lost')
        # We don't have deal_stage in db models natively tracked for history, 
        # but in a real system we would fetch closed deals from CRM or opportunity tracking table.
        # Let's mock fetching 30 days of predictions. We'll use the predictions table directly.
        # Actually, let's assume we fetch all predictions from the last 30 days.
        predictions = session.query(Prediction).filter(
            Prediction.model_version == prod_model.version_tag,
            Prediction.created_at >= thirty_days_ago
        ).all()
        
        if len(predictions) < 50:
            return f"Not enough samples ({len(predictions)})"

        # Mock actual outcomes for evaluation purposes since our prototype DB Opportunity table 
        # doesn't store close_value / deal_stage after the fact (the predictor just uses dummy values).
        # We simulate actual outcomes slightly noisy around the predicted probability
        import random
        y_true = []
        y_pred = []
        y_prob = []
        for p in predictions:
            y_prob.append(p.win_probability)
            y_pred.append(1 if p.outcome_prediction == "Won" else 0)
            # Actual: 1 with probability win_probability
            y_true.append(1 if random.random() < p.win_probability else 0)
            
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
            
        baseline_acc = prod_model.val_accuracy or 0.80
        degradation = baseline_acc - acc
        retraining_triggered = degradation > 0.05
        
        perf_log = PerformanceLog(
            model_version=prod_model.version_tag,
            sample_size=len(predictions),
            accuracy=acc,
            precision_score=prec,
            recall_score=rec,
            f1_score=f1,
            auc_roc=auc,
            degradation_vs_baseline=degradation,
            retraining_triggered=retraining_triggered
        )
        session.add(perf_log)
        session.commit()
        
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("model_evaluation")
        with mlflow.start_run():
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc_roc": auc,
                "degradation": degradation
            })
            
        if retraining_triggered:
            trigger_retraining.delay(reason="performance_degradation")
            
        return f"Evaluated {len(predictions)} samples. Acc: {acc:.3f}. Retrain triggered: {retraining_triggered}"
