/** Shapes returned by the FastAPI service. Mirrors app/schemas/scoring.py. */

export type Priority = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
export type ActionStatus = "suggested" | "accepted" | "dismissed";

export interface LatencyBreakdown {
  features?: number | null;
  xgb?: number | null;
  lstm?: number | null;
  shap?: number | null;
  nba?: number | null;
  cache?: number | null;
  total: number;
}

export interface ShapDriver {
  feature: string;
  label: string;
  value: number;
  shap: number;
  direction: "increases" | "decreases";
}

export interface DealScore {
  win_prob: number;
  days_to_close: number | null;
  model_version: string;
  scored_at: string;
  shap_top?: ShapDriver[] | null;
  latency_ms?: LatencyBreakdown | null;
  cache_hit?: boolean;
}

export interface DealAction {
  id: string;
  rule_id: string;
  action_type: string;
  reason: string;
  priority: Priority;
  urgency_score: number | null;
  status: ActionStatus;
  created_at: string;
  resolved_at: string | null;
  /** Present on the cross-pipeline queue, which joins the deal. */
  company?: string;
  deal_size?: number;
  stage?: string;
  owner_rep?: string;
  deal_id?: string;
}

export interface DealSummary {
  id: string;
  company: string;
  industry: string;
  region: string;
  deal_size: number;
  stage: string;
  source: string;
  owner_rep: string;
  created_at: string;
  expected_close: string | null;
  score: DealScore | null;
  top_action: Pick<
    DealAction,
    "id" | "action_type" | "reason" | "priority" | "status"
  > | null;
}

export interface DealListResponse {
  items: DealSummary[];
  total: number;
  limit: number;
  offset: number;
  sort: string;
  order: string;
  synthetic_data: boolean;
}

export interface Activity {
  id: string;
  type: string;
  occurred_at: string;
  origin: string;
  payload: Record<string, unknown> | null;
}

export interface DealDetailResponse {
  deal: Omit<DealSummary, "score" | "top_action"> & { closed_at: string | null };
  latest_score: DealScore | null;
  score_history: Array<{
    win_prob: number;
    days_to_close: number | null;
    scored_at: string;
  }>;
  timeline: Activity[];
  actions: DealAction[];
  synthetic_data: boolean;
}

export interface RecommendedAction {
  rule_id: string;
  action_type: string;
  reason: string;
  priority: Priority;
  urgency_score: number;
}

export interface ScoreResponse {
  deal_id: string;
  win_prob: number;
  days_to_close: number | null;
  model_version: string;
  shap_top: ShapDriver[];
  shap_base_value: number;
  feature_hash: string;
  latency_ms: LatencyBreakdown;
  cache_hit: boolean;
  scored_at: string;
  actions: RecommendedAction[];
}

export interface ExplainResponse {
  deal_id: string;
  win_prob: number;
  model_version: string;
  shap_values: ShapDriver[];
  shap_base_value: number;
  margin: number;
  scored_at: string;
}

export interface WhatIfResponse {
  deal_id: string;
  baseline: { win_prob: number; days_to_close: number };
  what_if: { win_prob: number; days_to_close: number };
  delta: { win_prob: number; days_to_close: number };
  applied_overrides: Record<string, { from: number; to: number }>;
  shap_top: ShapDriver[];
  model_version: string;
  latency_ms: LatencyBreakdown;
  persisted: boolean;
}

export interface MetricsResponse {
  model_version: string;
  generated_on: string;
  demo: {
    label: string;
    synthetic_data: boolean;
    win_probability: Record<string, number>;
    days_to_close: Record<string, unknown>;
    data: Record<string, unknown>;
  };
  global_drivers: Array<{ feature: string; label: string; mean_abs_shap: number }>;
  latency_ms: { measured: number | null; note: string };
  note: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  models: Record<string, unknown>;
  db: string;
  redis: string;
  bus: { available?: boolean; error?: string; stream_len?: number; pending?: number };
  synthetic_data: boolean;
}

/** SSE payload for `score_updated`. */
export interface ScoreUpdatedEvent {
  deal_id: string;
  event_id: string;
  activity_type: string;
  win_prob: number;
  days_to_close: number | null;
  model_version: string;
  shap_top: ShapDriver[];
  latency_ms: LatencyBreakdown;
  scored_at: string;
}
