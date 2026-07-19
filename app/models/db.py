from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class Opportunity(Base):
    __tablename__ = "opportunities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    opportunity_id = Column(String(64), unique=True, nullable=False)
    sales_agent = Column(String(128), nullable=False)
    product = Column(String(128), nullable=False)
    engage_date = Column(String(32), nullable=False)
    crm_source = Column(String(32), default='manual')
    created_at = Column(String, default=lambda: datetime.now().isoformat())
    updated_at = Column(String, default=lambda: datetime.now().isoformat())
    
    predictions = relationship("Prediction", back_populates="opportunity")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    opportunity_id = Column(String(64), ForeignKey("opportunities.opportunity_id"), nullable=False)
    model_version = Column(String(32), nullable=False)
    win_probability = Column(Float, nullable=False)
    outcome_prediction = Column(String(8), nullable=False)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    predicted_days_remaining = Column(Integer, nullable=True)
    shap_values = Column(JSON, nullable=False)
    nl_explanation = Column(Text, nullable=True)
    inference_latency_ms = Column(Float, nullable=True)
    cached = Column(Boolean, default=False)
    is_ab_request = Column(Boolean, default=False)
    created_at = Column(String, default=lambda: datetime.now().isoformat())
    
    opportunity = relationship("Opportunity", back_populates="predictions")
    recommendations = relationship("Recommendation", back_populates="prediction")

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=False)
    opportunity_id = Column(String(64), nullable=False)
    action = Column(Text, nullable=False)
    rationale = Column(Text, nullable=False)
    expected_impact = Column(Text, nullable=True)
    priority = Column(String(8), nullable=False)
    urgency_score = Column(Float, nullable=False)
    adopted_at = Column(String, nullable=True)
    outcome_delta = Column(Float, nullable=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())
    
    prediction = relationship("Prediction", back_populates="recommendations")

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_type = Column(String(32), nullable=False)
    version_tag = Column(String(32), unique=True, nullable=False)
    val_accuracy = Column(Float, nullable=True)
    val_auc_roc = Column(Float, nullable=True)
    val_mae = Column(Float, nullable=True)
    artifact_path = Column(Text, nullable=False)
    is_production = Column(Boolean, default=False)
    is_ab_candidate = Column(Boolean, default=False)
    deployed_at = Column(String, nullable=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

class PerformanceLog(Base):
    __tablename__ = "performance_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version = Column(String(32), nullable=False)
    evaluation_date = Column(String, default=lambda: datetime.now().isoformat())
    window_days = Column(Integer, default=30)
    sample_size = Column(Integer, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_roc = Column(Float, nullable=True)
    degradation_vs_baseline = Column(Float, nullable=True)
    retraining_triggered = Column(Boolean, default=False)

class DriftResult(Base):
    __tablename__ = "drift_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    check_date = Column(String, default=lambda: datetime.now().isoformat())
    feature_name = Column(String(128), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    training_mean = Column(Float, nullable=True)
    production_mean = Column(Float, nullable=True)
