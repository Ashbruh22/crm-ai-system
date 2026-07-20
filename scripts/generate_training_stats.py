import json
import numpy as np
import pandas as pd

def save_training_stats():
    df = pd.read_csv('data/processed_dataset.csv')
    
    # We want to skip 'opportunity_id'
    feature_cols = [col for col in df.columns if col != 'opportunity_id']
    X_train_df = df[feature_cols].values
    
    stats = {}
    for i, name in enumerate(feature_cols):
        col = X_train_df[:, i]
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "p25": float(np.percentile(col, 25)),
            "p50": float(np.percentile(col, 50)),
            "p75": float(np.percentile(col, 75)),
            "sample": col[:500].tolist()  # 500-row sample for KS test
        }
    with open('data/training_feature_stats.json', 'w') as f:
        json.dump(stats, f)
        
if __name__ == '__main__':
    save_training_stats()
    print("Successfully generated data/training_feature_stats.json")
