from ml.celery_app import celery_app
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import os
import uuid
from datetime import datetime
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import optuna
import mlflow

from app.models.db import ModelVersion
from ml.pipeline import build_pipeline

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://crm_ai_app:crm_ai_db_password_dev@db:5432/crm_ai")
sync_engine = create_engine(DATABASE_URL.replace("postgresql+asyncpg", "postgresql"))

@celery_app.task(bind=True, max_retries=1, time_limit=3600)
def trigger_retraining(self, reason: str = 'manual'):
    """
    Full retraining pipeline:
    1. Load Data
    2. HPO with Optuna (50 trials)
    3. Train XGBoost
    4. Save Candidate Artifact
    5. Update model_versions (is_ab_candidate=True)
    6. Schedule auto promote (7 days)
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("xgboost_retraining")
    
    with mlflow.start_run():
        mlflow.log_param("reason", reason)
        
        # 1. Load Data (simulated from static CSV for the prototype)
        df = pd.read_csv('data/sales_pipeline.csv')
        
        # Split target and features
        X = df.drop(columns=['close_value', 'deal_stage', 'close_date', 'account'], errors='ignore')
        y = (df['deal_stage'].str.lower() == 'closed-won').astype(int)
        
        pipeline = build_pipeline()
        X_transformed = pipeline.fit_transform(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
        
        # 2. Optuna HPO
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        
        best_params = study.best_params
        mlflow.log_params(best_params)
        
        # 3. Train best model
        best_model = XGBClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        preds_class = best_model.predict(X_test)
        preds_proba = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds_class)
        auc = roc_auc_score(y_test, preds_proba)
        
        mlflow.log_metrics({"accuracy": acc, "auc_roc": auc})
        
        if acc < 0.60:
            return f"Retraining aborted. Candidate accuracy {acc:.3f} below 60% threshold."
            
        # 4. Save Candidate Artifact
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        version_tag = f"xgboost-candidate-{timestamp}"
        artifact_path = f"ml/models/{version_tag}.joblib"
        os.makedirs("ml/models", exist_ok=True)
        joblib.dump(best_model, artifact_path)
        
        # 5. Insert to DB
        with Session(sync_engine) as session:
            new_version = ModelVersion(
                model_type="xgboost",
                version_tag=version_tag,
                val_accuracy=float(acc),
                val_auc_roc=float(auc),
                artifact_path=artifact_path,
                is_production=False,
                is_ab_candidate=True
            )
            session.add(new_version)
            session.commit()
            
        # 6. Schedule auto promotion
        from ml.tasks.promote import auto_promote_candidate
        # 7 days = 604800 seconds
        auto_promote_candidate.apply_async(args=[version_tag], countdown=604800)
        
        return f"Candidate {version_tag} built. AUC: {auc:.3f}. A/B test started."
