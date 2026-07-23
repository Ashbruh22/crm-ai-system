export interface OutcomePredictRequest {
  opportunity_id: string;
  sales_agent: string;
  product: string;
  engage_date: string;
}

export interface TopFactor {
  feature: string;
  shap: number;
  direction: 'positive' | 'negative';
}

export interface ShapExplanation {
  nl_explanation: string;
  top_factors: TopFactor[];
}

export interface RecommendationItem {
  action: string;
  rationale: string;
  expected_impact?: string;
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  urgency_score: number;
}

export interface OutcomePredictResponse {
  opportunity_id: string;
  win_probability: number;
  outcome_prediction: string;
  confidence_interval?: [number, number];
  shap_explanation: ShapExplanation;
  recommendations: RecommendationItem[];
  model_version: string;
  inference_latency_ms: number;
  cached: boolean;
  timestamp?: string; // Appended in history
}

export interface CyclePredictRequest {
  opportunity_id: string;
  sales_agent?: string;
  product?: string;
  engage_date?: string;
  days_elapsed?: number;
  current_stage?: string;
  num_activities?: number;
  days_since_last_activity?: number;
}

export interface CyclePredictResponse {
  opportunity_id: string;
  predicted_days_remaining: number;
  model_version: string;
  cached: boolean;
}

export interface GlobalRanking {
  feature: string;
  mean_abs_shap: number;
}

export interface GlobalImportanceResponse {
  rankings: GlobalRanking[];
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  role: string;
}

export interface DriftResponse {
  ks_stat: number;
  p_value: number;
  drift_detected: boolean;
}

export interface PerformanceResponse {
  accuracy: number;
  auc_roc: number;
}

export interface RecommendationFeedback {
  prediction_id: string;
  adopted_at: string;
  outcome_delta: number;
}

export interface LocalExplainResponse {
  waterfall_data: {
    base_value: number;
    output_value: number;
    features: {
      name: string;
      value: number;
      shap: number;
      cumulative: number;
    }[];
  };
  force_plot_data: {
    base_value: number;
    output_value: number;
    positives: {
      name: string;
      value: number;
      shap: number;
    }[];
    negatives: {
      name: string;
      value: number;
      shap: number;
    }[];
  };
}
