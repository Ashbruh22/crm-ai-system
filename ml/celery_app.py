from celery import Celery
from celery.schedules import crontab
import os

celery_app = Celery(
    'crm_ai',
    broker=os.environ.get('REDIS_URL', 'redis://redis:6379/0'),
    backend=os.environ.get('REDIS_URL', 'redis://redis:6379/0'),
    include=['ml.tasks.retrain', 'ml.tasks.drift', 'ml.tasks.evaluate', 'ml.tasks.promote']
)

celery_app.conf.beat_schedule = {
    'evaluate-model-performance': {
        'task': 'ml.tasks.evaluate.evaluate_model_performance',
        'schedule': crontab(day_of_month='1', hour='2', minute='0'),  # Monthly at 2am
    },
    'run-drift-detection': {
        'task': 'ml.tasks.drift.run_drift_detection',
        'schedule': crontab(day_of_week='1', hour='3', minute='0'),   # Weekly Monday 3am
    },
}

celery_app.conf.timezone = 'UTC'
celery_app.conf.task_serializer = 'json'
celery_app.conf.result_expires = 86400  # 24 hours
