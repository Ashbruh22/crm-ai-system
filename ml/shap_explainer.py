import shap
import numpy as np
import pandas as pd
import math

LEAK_COLUMNS = [
    'deal_stage', 'close_value', 'close_date', 'sales_cycle_days',
    'deal_stage_duration', 'stage_transition_rate', 'days_since_last_activity',
    'activity_frequency', 'response_time_avg', 'urgency_indicators',
    'account_size_category', 'multi_stakeholder_flag', 'competitive_displacement_risk'
]

FRIENDLY_NAMES = {
    'num__agent_win_rate': "agent's historical win rate",
    'num__agent_avg_deal_value': "agent's average deal size",
    'num__agent_deal_velocity': "agent's historical deal velocity",
    'num__product_win_rate': "product's historical win rate",
    'num__product_category_complexity_score': "product complexity",
    'num__day_of_week_engaged': "day of week engaged",
    'num__month_engaged': "month engaged",
    'num__quarter_engaged': "quarter engaged",
    'sales_agent': "assigned sales agent",
    'product': "product category"
}

def _friendly_name(fname):
    """Return friendly label for a feature name if available, otherwise the raw name."""
    return FRIENDLY_NAMES.get(fname, fname)

def assert_no_leaks(feature_names):
    found_leaks = [f for f in feature_names if any(leak in f for leak in LEAK_COLUMNS)]
    if found_leaks:
        raise ValueError(f"CRITICAL: Leak columns detected in feature names: {found_leaks}")

class XGBoostShapExplainer:
    def __init__(self, model, feature_names):
        print(f"DEBUG: Initializing XGBoostShapExplainer with {len(feature_names)} features.")
        assert_no_leaks(feature_names)
        self.model = model
        self.feature_names = list(feature_names)
        print(f"DEBUG: Model type: {type(model)}")
        booster = model.get_booster() if hasattr(model, 'get_booster') else model
        
        import shap.explainers._tree
        if not hasattr(shap.explainers._tree, '_ubj_patched'):
            if hasattr(shap.explainers._tree, 'decode_ubjson_buffer'):
                print("DEBUG: Patching decode_ubjson_buffer for SHAP compatibility.")
                original_decode = shap.explainers._tree.decode_ubjson_buffer
                def patched_decode(*args, **kwargs):
                    obj = original_decode(*args, **kwargs)
                    if isinstance(obj, dict) and 'learner' in obj:
                        try:
                            val = obj['learner']['learner_model_param']['base_score']
                            if isinstance(val, str) and val.startswith('['):
                                obj['learner']['learner_model_param']['base_score'] = val.strip('[]')
                        except KeyError:
                            pass
                    return obj
                shap.explainers._tree.decode_ubjson_buffer = patched_decode
            shap.explainers._tree._ubj_patched = True
            
        self.explainer = shap.TreeExplainer(booster)
        
    def _aggregate_one_hot(self, shap_values, X_row):
        """
        Aggregates one-hot encoded SHAP values.
        Returns a dictionary mapping grouped feature names to their aggregated SHAP value and active category label.
        """
        aggregated_shap = {}
        active_labels = {}
        
        for i, fname in enumerate(self.feature_names):
            val = shap_values[i]
            if fname.startswith('cat__'):
                # e.g., cat__sales_agent_Anita Rao
                parts = fname.split('__')[1].split('_')
                # But it could be cat__product_Analytics Pro.
                # Better split by first underscore after cat__
                prefix = fname[5:] # remove cat__
                # Find the category type by seeing if it starts with 'product' or 'sales_agent'
                if prefix.startswith('product_'):
                    parent = 'product'
                    category = prefix[8:]
                elif prefix.startswith('sales_agent_'):
                    parent = 'sales_agent'
                    category = prefix[12:]
                else:
                    parent = prefix
                    category = prefix
                
                aggregated_shap[parent] = aggregated_shap.get(parent, 0.0) + val
                
                # Check if this category is active (value == 1)
                row_val = X_row[i] if isinstance(X_row, (list, np.ndarray)) else X_row.iloc[i]
                if float(row_val) > 0.5: # it's 1
                    active_labels[parent] = category
            else:
                aggregated_shap[fname] = val
                row_val = X_row[i] if isinstance(X_row, (list, np.ndarray)) else X_row.iloc[i]
                active_labels[fname] = float(row_val) if isinstance(row_val, (float, np.floating, int, np.integer)) else str(row_val)
                
        # For parents that don't have an active label (all 0s), just give 'Unknown'
        for k in aggregated_shap.keys():
            if k not in active_labels:
                active_labels[k] = 'Unknown/Other'
                
        return aggregated_shap, active_labels

    def explain_single(self, X_row):
        # Convert X_row to 2D for shap
        if isinstance(X_row, pd.Series):
            X_2d = X_row.values.reshape(1, -1)
        elif isinstance(X_row, pd.DataFrame):
            X_2d = X_row.values
        else:
            X_2d = np.array(X_row).reshape(1, -1)
            
        shap_vals = self.explainer.shap_values(X_2d)
        # shap_values returns log-odds for binary classification
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
            
        # extract the 1D array
        if isinstance(shap_vals, list):
            shap_array = shap_vals[1][0]
        else:
            shap_array = shap_vals[0]
            
        return float(base_value), shap_array.tolist()

    def global_importance(self, X_test):
        shap_vals = self.explainer.shap_values(X_test)
        if isinstance(shap_vals, list):
            shap_array = shap_vals[1]
        else:
            shap_array = shap_vals
            
        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        
        # Aggregate one-hot
        agg_mean_abs = {}
        for i, fname in enumerate(self.feature_names):
            if fname.startswith('cat__'):
                prefix = fname[5:]
                if prefix.startswith('product_'):
                    parent = 'product'
                elif prefix.startswith('sales_agent_'):
                    parent = 'sales_agent'
                else:
                    parent = prefix
                agg_mean_abs[parent] = agg_mean_abs.get(parent, 0.0) + mean_abs_shap[i]
            else:
                agg_mean_abs[fname] = mean_abs_shap[i]
                
        total_shap = sum(agg_mean_abs.values())
        
        importance = []
        for fname, val in agg_mean_abs.items():
            importance.append({
                'feature': fname,
                'mean_abs_shap': float(val),
                'pct_of_total': float(val / total_shap) if total_shap > 0 else 0.0
            })
            
        importance.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
        return importance

    def local_waterfall_data(self, shap_values, base_value, X_row, top_n=5):
        agg_shap, active_labels = self._aggregate_one_hot(shap_values, X_row)
        
        # Sort by absolute SHAP value
        sorted_features = sorted(agg_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        
        output_value = base_value + sum(agg_shap.values())
        
        features_list = []
        for k, v in sorted_features[:top_n]:
            features_list.append({
                'name': _friendly_name(k),
                'label': active_labels[k],
                'shap': float(v)
            })
            
        other_shap = sum(v for k, v in sorted_features[top_n:])
        
        return {
            'base_value': float(base_value),
            'output_value': float(output_value),
            'features': features_list,
            'other_features_combined_shap': float(other_shap)
        }

    def force_plot_data(self, shap_values, base_value, X_row):
        agg_shap, active_labels = self._aggregate_one_hot(shap_values, X_row)
        output_value = base_value + sum(agg_shap.values())
        
        # Build feature list with friendly names and split into positives/negatives
        features_list = []
        positives = []
        negatives = []
        for k, v in agg_shap.items():
            entry = {
                'name': _friendly_name(k),
                'label': active_labels[k],
                'shap': float(v)
            }
            features_list.append(entry)
            if v > 0:
                positives.append(entry)
            elif v < 0:
                negatives.append(entry)
        
        return {
            'base_value': float(base_value),
            'output_value': float(output_value),
            'features': features_list,
            'positives': positives,
            'negatives': negatives
        }

    def nl_explanation(self, waterfall_data):
        # convert log-odds to probability
        def expit(x):
            return 1 / (1 + math.exp(-x))
            
        prob = expit(waterfall_data['output_value']) * 100
        
        pos_factors = []
        neg_factors = []
        
        for f in waterfall_data['features']:
            # f['name'] is already a friendly label from local_waterfall_data
            friendly_name = f['name']
            # For categorical features, append the active category label
            if f['name'] in ['assigned sales agent', 'product category']:
                friendly_name = f"{friendly_name} ({f['label']})"
            
            if f['shap'] > 0:
                pos_factors.append(friendly_name)
            else:
                neg_factors.append(friendly_name)
                
        pos_str = ", ".join(pos_factors[:3]) if pos_factors else "None"
        neg_str = ", ".join(neg_factors[:3]) if neg_factors else "None"
        
        explanation = f"Win probability {prob:.1f}%. Positive factors: [{pos_str}]. Risk factors: [{neg_str}]."
        return explanation
