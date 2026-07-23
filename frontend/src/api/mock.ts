import MockAdapter from 'axios-mock-adapter';
import type { AxiosInstance } from 'axios';

export function setupMockApi(api: AxiosInstance) {
  const mock = new MockAdapter(api, { delayResponse: 800 });

  // 1. Auth
  mock.onPost('/auth/token').reply((config) => {
    const data = JSON.parse(config.data);
    if (data.username === 'demo_admin' && data.password === 'admin_pass') {
      return [200, { access_token: 'mock-admin-token', token_type: 'bearer', role: 'admin' }];
    }
    if (data.username === 'demo_rep' && data.password === 'demo_pass') {
      return [200, { access_token: 'mock-rep-token', token_type: 'bearer', role: 'sales_rep' }];
    }
    return [401, { detail: 'Invalid credentials' }];
  });

  // 2. Global SHAP
  mock.onGet('/explain/global').reply(200, {
    rankings: [
      { feature: 'agent_historical_win_rate', mean_abs_shap: 0.85 },
      { feature: 'product_complexity_score', mean_abs_shap: 0.62 },
      { feature: 'days_since_last_activity', mean_abs_shap: 0.41 },
      { feature: 'deal_size_usd', mean_abs_shap: 0.35 },
      { feature: 'competitor_presence', mean_abs_shap: 0.28 }
    ]
  });

  // 3. Admin endpoints
  mock.onGet('/admin/drift').reply(200, { ks_stat: 0.042, p_value: 0.85, drift_detected: false });
  mock.onGet('/admin/performance').reply(200, { accuracy: 0.89, auc_roc: 0.94 });
  mock.onPost('/admin/retrain').reply(200, { task_id: 'mock-task-id-1234' });

  // 4. Team Recommendations (Dashboard)
  mock.onGet(/\/recommendations\/team\/.*/).reply(200, {
    recommendations: [
      { action: 'Schedule executive alignment meeting', priority: 'CRITICAL', urgency_score: 95, rationale: 'Stalled at proposal stage for 14 days.' },
      { action: 'Offer pilot program', priority: 'HIGH', urgency_score: 82, rationale: 'High product complexity requires hands-on validation.' },
      { action: 'Bring in technical sales engineer', priority: 'HIGH', urgency_score: 78, rationale: 'Competitor presence identified, technical differentiation needed.' },
      { action: 'Send ROI case study', priority: 'MEDIUM', urgency_score: 65, rationale: 'Price sensitivity detected in recent interactions.' },
      { action: 'Follow up on security review', priority: 'CRITICAL', urgency_score: 92, rationale: 'Security questionnaire pending for 8 days.' }
    ]
  });

  // 5. Deal specific endpoints
  const oppIdRegex = /\/predict\/history\/(OPP_\d+)/;
  mock.onGet(oppIdRegex).reply(() => {
    return [200, [
      { win_probability: 0.35 },
      { win_probability: 0.42 },
      { win_probability: 0.40 },
      { win_probability: 0.55 },
      { win_probability: 0.68 }
    ]];
  });

  mock.onGet(/\/recommendations\/(OPP_\d+)/).reply(200, {
    recommendations: [
      { action: 'Schedule executive alignment meeting', priority: 'CRITICAL', urgency_score: 95, rationale: 'Stalled at proposal stage for 14 days.', expected_impact: '+15% win probability' }
    ]
  });

  mock.onGet(/\/explain\/local\/(OPP_\d+)/).reply(200, {
    waterfall_data: {
      base_value: 0.25,
      output_value: 0.68,
      features: [
        { name: 'agent_historical_win_rate', value: 0.8, shap: 0.25, cumulative: 0.50 },
        { name: 'product_complexity_score', value: 0.9, shap: -0.15, cumulative: 0.35 },
        { name: 'recent_engagement_score', value: 0.85, shap: 0.33, cumulative: 0.68 }
      ]
    },
    force_plot_data: { base_value: 0.25, output_value: 0.68, positives: [], negatives: [] }
  });

  mock.onPost('/predict/outcome').reply(200, {
    opportunity_id: 'OPP_12345',
    win_probability: 0.68,
    outcome_prediction: 'Won',
    shap_explanation: { nl_explanation: 'Win probability 68%. Strong agent history overcomes product complexity.', top_factors: [] },
    recommendations: [],
    model_version: 'xgboost-v1.0',
    inference_latency_ms: 12,
    cached: false
  });

  mock.onPost('/predict/cycle').reply(200, {
    opportunity_id: 'OPP_12345',
    predicted_days_remaining: 14,
    model_version: 'lstm-v1.0',
    cached: false
  });

  mock.onPost('/crm/webhook/simulate').reply(200, { status: 'simulated' });
  
  // Feedback
  mock.onPost('/recommendations/feedback').reply(200, { status: 'recorded' });

  // Pass-through anything else (e.g. if we are in dev and want real api)
  mock.onAny().passThrough();
}
