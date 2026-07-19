import json
import os
from ml.shap_explainer import XGBoostShapExplainer

# Global explainer instance initialized on first use or at startup
_explainer = None

def get_shap_explainer(state) -> XGBoostShapExplainer:
    """Gets or initializes the SHAP explainer."""
    global _explainer
    if _explainer is None:
        xgb_model = state.xgboost_model
        pipeline = state.pipeline
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        _explainer = XGBoostShapExplainer(xgb_model, feature_names)
    return _explainer

def get_global_importance():
    """Reads the pre-computed global SHAP importance from Phase 3."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(base_dir, 'data', 'global_shap_importance.json')
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return []

def get_local_explanation(prediction_record, state):
    """Reconstructs waterfall and force plot data from stored SHAP values."""
    explainer = get_shap_explainer(state)
    
    shap_values = prediction_record.shap_values
    base_value = explainer.explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = base_value[0]
        
    # We need the original transformed features to get active labels
    # We'll just pass a dummy zero array for now, as reproducing the exact X_row 
    # without storing it requires re-running the pipeline. 
    # In a full implementation, X_row should be saved or re-derived.
    dummy_x_row = [0] * len(explainer.feature_names)
    
    waterfall_data = explainer.local_waterfall_data(shap_values, base_value, dummy_x_row)
    force_plot_data = explainer.force_plot_data(shap_values, base_value, dummy_x_row)
    
    return {
        "waterfall_data": waterfall_data,
        "force_plot_data": force_plot_data
    }
