import sys, os
sys.path.append(os.path.dirname(__file__))
from pipeline import TemporalFeatureExtractor, AgentPerformanceAggregator, ProductPerformanceAggregator
sys.modules["__main__"].TemporalFeatureExtractor = TemporalFeatureExtractor
sys.modules["__main__"].AgentPerformanceAggregator = AgentPerformanceAggregator
sys.modules["__main__"].ProductPerformanceAggregator = ProductPerformanceAggregator

import joblib
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(base_dir, "ml", "xgboost_model.joblib"))
pipeline = joblib.load(os.path.join(base_dir, "ml", "pipeline.joblib"))

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = model.feature_importances_

print(f"Number of features: {len(feature_names)}")
print(f"Number of importance values: {len(importances)}")
assert len(feature_names) == len(importances), "MISMATCH"

ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])

print("\nFull ranked feature importance list:")
for name, score in ranked:
    print(f"{name}: {score:.4f}")

total = sum(importances)
top_name, top_score = ranked[0]
print(f"\nTop feature importance as % of total:")
print(f"{top_name}: {top_score/total*100:.1f}%")

print("\nTop 3 combined as % of total:")
top3_sum = sum(s for _, s in ranked[:3])
print(f"{top3_sum/total*100:.1f}%")

print("\n=== LSTM Per-Timestep Variance Check ===")
from sklearn.model_selection import train_test_split

data_path = os.path.join(base_dir, "data", "sales_pipeline.csv")
df = pd.read_csv(data_path)
closed_df = df[df["deal_stage"].isin(["closed-won","closed-lost"])].copy()
df_train, _ = train_test_split(closed_df, test_size=0.3, random_state=42, stratify=closed_df["deal_stage"])

sample = df_train.head(5).copy()
raw_rows = []
for idx, row in sample.iterrows():
    engage_dt = pd.to_datetime(row["engage_date"])
    for t in range(30):
        snap_row = row.copy()
        snap_row["engage_date"] = (engage_dt + pd.Timedelta(days=t)).strftime("%Y-%m-%d")
        raw_rows.append(snap_row)

raw_df = pd.DataFrame(raw_rows)
raw_features = raw_df.drop(columns=["opportunity_id","account","close_value","close_date","deal_stage"], errors="ignore")
transformed = pipeline.transform(raw_features).astype("float32")

feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
print("\nPer-column std across 30 timesteps for deal #0:")
deal0 = transformed[0:30]
stds = deal0.std(axis=0)
for name, s in zip(feat_names, stds):
    print(f"  {name}: std={s:.4f}")

print(f"\nMean std across all columns: {stds.mean():.4f}")
print(f"Min std: {stds.min():.4f}")
print(f"Max std: {stds.max():.4f}")
print(f"Conclusion: {'FROZEN' if stds.max() < 1e-6 else 'VARYING'}")
