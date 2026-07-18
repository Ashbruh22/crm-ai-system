import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from pipeline import TemporalFeatureExtractor, AgentPerformanceAggregator, ProductPerformanceAggregator
from shap_explainer import XGBoostShapExplainer

sys.modules['__main__'].TemporalFeatureExtractor = TemporalFeatureExtractor
sys.modules['__main__'].AgentPerformanceAggregator = AgentPerformanceAggregator
sys.modules['__main__'].ProductPerformanceAggregator = ProductPerformanceAggregator

def run():
    print("Loading models and pipeline...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    pipeline = joblib.load(os.path.join(base_dir, 'ml', 'pipeline.joblib'))
    xgb_model = joblib.load(os.path.join(base_dir, 'ml', 'xgboost_model.joblib'))
    
    if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    elif hasattr(pipeline, 'get_feature_names_out'):
        feature_names = pipeline.get_feature_names_out()
    else:
        # Fallback to model's feature names if present
        feature_names = xgb_model.feature_names_in_ if hasattr(xgb_model, 'feature_names_in_') else []

    print("Initializing SHAP explainer (will run leak assertion)...")
    explainer = XGBoostShapExplainer(xgb_model, feature_names)
    
    print("Loading data for test split...")
    df = pd.read_csv(os.path.join(base_dir, 'data', 'sales_pipeline.csv'))
    closed_df = df[df['deal_stage'].isin(['closed-won', 'closed-lost'])].copy()
    
    df_train, df_test = train_test_split(
        closed_df,
        test_size=0.3,
        random_state=42,
        stratify=closed_df['deal_stage']
    )
    
    X_test_cls = pipeline.transform(df_test).astype(np.float32)
    
    print("Generating global importance...")
    global_imp = explainer.global_importance(X_test_cls)
    
    out_dir = os.path.join(base_dir, 'data')
    with open(os.path.join(out_dir, 'global_shap_importance.json'), 'w') as f:
        json.dump(global_imp, f, indent=2)
        
    print("\n--- Feature Importance Comparison (Top 10) ---")
    
    # XGBoost native importance
    booster = xgb_model.get_booster() if hasattr(xgb_model, 'get_booster') else xgb_model
    native_scores = booster.get_score(importance_type='gain')
    
    # Aggregate native scores for one-hot
    agg_native = {}
    for i, fname in enumerate(feature_names):
        # Native scores use keys like 'f0', 'f1', etc if feature names were not set on booster
        key = fname if fname in native_scores else f"f{i}"
        val = native_scores.get(key, 0.0)
        
        if fname.startswith('cat__'):
            prefix = fname[5:]
            if prefix.startswith('product_'):
                parent = 'product'
            elif prefix.startswith('sales_agent_'):
                parent = 'sales_agent'
            else:
                parent = prefix
            agg_native[parent] = agg_native.get(parent, 0.0) + val
        else:
            agg_native[fname] = val
            
    sorted_native = sorted(agg_native.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'SHAP (Aggregated)':<40} | {'XGB Native Gain (Aggregated)':<40}")
    print("-" * 85)
    for i in range(10):
        shap_feat = global_imp[i]['feature'] if i < len(global_imp) else ""
        nat_feat = sorted_native[i][0] if i < len(sorted_native) else ""
        print(f"{shap_feat:<40} | {nat_feat:<40}")

    print("\nBenchmarking latency...")
    latencies = []
    sample_row = X_test_cls[0]
    for i in range(100):
        start = time.perf_counter()
        explainer.explain_single(sample_row)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        if i >= 5: # discard first 5
            latencies.append(elapsed)
            
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    print(f"Latency (ms) -> p50: {p50:.2f} ms | p95: {p95:.2f} ms")
    
    print("Generating sample explanations...")
    samples = {}
    
    # Find low, moderate, high risk by predicting probability
    probs = xgb_model.predict_proba(X_test_cls)[:, 1]
    
    idx_low_risk = np.argmax(probs) # Highest win probability
    idx_high_risk = np.argmin(probs) # Lowest win probability
    idx_mod_risk = np.abs(probs - 0.5).argmin() # Closest to 50%
    
    for label, idx in [('low_risk', idx_low_risk), 
                       ('moderate_risk', idx_mod_risk), 
                       ('high_risk', idx_high_risk)]:
        try:
            row = X_test_cls[idx]
            base_val, shap_vals = explainer.explain_single(row)
            wf_data = explainer.local_waterfall_data(shap_vals, base_val, row)
            fp_data = explainer.force_plot_data(shap_vals, base_val, row)
            nl_exp = explainer.nl_explanation(wf_data)
            
            samples[label] = {
                'win_probability': float(probs[idx]),
                'waterfall': wf_data,
                'force_plot': fp_data,
                'nl_explanation': nl_exp
            }
        except Exception as e:
            print(f"Error generating explanation for {label}: {e}")
        
    with open(os.path.join(out_dir, 'sample_explanations.json'), 'w') as f:
        json.dump(samples, f, indent=2)
        
    print("Checking Validation Gate...")
    top_10_shap = [item['feature'] for item in global_imp[:10]]
    expected = {'num__agent_win_rate', 'num__product_win_rate', 'num__agent_avg_deal_value', 'num__product_category_complexity_score', 'sales_agent', 'product'}
    
    found_count = len(set(top_10_shap).intersection(expected))
    if found_count >= 3:
        print(f"Validation Passed! Found {found_count}/6 expected key features in top 10.")
    else:
        print(f"Validation FAILED! Found only {found_count}/6 expected key features in top 10.")
        sys.exit(1)
        
if __name__ == "__main__":
    run()
