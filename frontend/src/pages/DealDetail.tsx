import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import api from '../api/client';
import { WinProbabilityBadge } from '../components/WinProbabilityBadge';
import { SHAPWaterfallChart } from '../components/SHAPWaterfallChart';
import { CycleForecastCard } from '../components/CycleForecastCard';
import { RecommendationCard } from '../components/RecommendationCard';
import type {  OutcomePredictResponse, LocalExplainResponse, RecommendationItem, CyclePredictResponse  } from '../types';
import { Briefcase, User, Calendar, Loader2 } from 'lucide-react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export function DealDetail() {
  const { id } = useParams<{ id: string }>();
  const [outcome, setOutcome] = useState<OutcomePredictResponse | null>(null);
  const [explain, setExplain] = useState<LocalExplainResponse | null>(null);
  const [cycle, setCycle] = useState<CyclePredictResponse | null>(null);
  const [recs, setRecs] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(true);

  // Mock deal context
  const dealContext = {
    agent: 'Sarah Chen',
    product: 'Enterprise Suite',
    stage: 'Proposal/Price Quote',
    engageDate: '2024-01-15'
  };

  useEffect(() => {
    async function loadDeal() {
      if (!id) return;
      setLoading(true);
      try {
        // 1. Trigger live outcome prediction
        const outRes = await api.post('/predict/outcome', {
          opportunity_id: id,
          sales_agent: dealContext.agent,
          product: dealContext.product,
          engage_date: dealContext.engageDate
        });
        setOutcome(outRes.data);

        // 2. Fetch local explain
        const expRes = await api.get(`/explain/local/${id}`);
        setExplain(expRes.data);

        // 3. Fetch cycle forecast
        const cycRes = await api.post('/predict/cycle', {
          opportunity_id: id,
          sales_agent: dealContext.agent,
          product: dealContext.product,
          current_stage: 'proposal',
          days_elapsed: 15
        });
        setCycle(cycRes.data);

        // 4. Fetch recommendations
        const recRes = await api.get(`/recommendations/${id}`);
        // Endpoint returns a single RecommendationItem object if it's the direct return from prediction pipeline,
        // but PRD says `GET /api/v1/recommendations/{opp_id}`.
        // Let's assume it returns { recommendations: [] } based on schemas.
        setRecs(recRes.data.recommendations || []);

      } catch (err) {
        console.error("Failed to load deal data", err);
      } finally {
        setLoading(false);
      }
    }
    loadDeal();
  }, [id]);

  // Fetch history for timeline
  const { data: history } = useQuery({
    queryKey: ['history', id],
    queryFn: async () => {
      const res = await api.get(`/predict/history/${id}`);
      return res.data as OutcomePredictResponse[];
    },
    enabled: !!id
  });

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-brand-600" />
      </div>
    );
  }

  if (!outcome) {
    return <div>Deal not found or error loading predictions.</div>;
  }

  const historyData = history?.map((h, i) => ({
    name: `T-${history.length - i}`,
    prob: Math.round(h.win_probability * 100)
  })) || [{ name: 'Current', prob: Math.round(outcome.win_probability * 100) }];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">{id}</h1>
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <span className="flex items-center gap-1"><User className="w-4 h-4" /> {dealContext.agent}</span>
            <span className="flex items-center gap-1"><Briefcase className="w-4 h-4" /> {dealContext.product}</span>
            <span className="flex items-center gap-1"><Calendar className="w-4 h-4" /> Since {dealContext.engageDate}</span>
          </div>
        </div>
        <div className="px-4 py-2 bg-blue-50 text-blue-700 rounded-lg font-medium text-sm border border-blue-200">
          Stage: {dealContext.stage}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left Column (60%) */}
        <div className="lg:col-span-3 space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col items-center">
            <h2 className="text-lg font-bold text-gray-900 self-start mb-4">Win Probability</h2>
            <WinProbabilityBadge probability={outcome.win_probability} size="lg" />
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-2">Outcome Drivers (SHAP)</h2>
            <p className="text-sm text-gray-600 mb-6">{outcome.shap_explanation.nl_explanation}</p>
            {explain?.waterfall_data && (
              <SHAPWaterfallChart 
                baseValue={explain.waterfall_data.base_value}
                outputValue={explain.waterfall_data.output_value}
                features={explain.waterfall_data.features}
              />
            )}
          </div>

          {cycle && (
            <CycleForecastCard 
              predictedDaysRemaining={cycle.predicted_days_remaining} 
              daysElapsed={15} 
            />
          )}
        </div>

        {/* Right Column (40%) */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4">Action Plan</h2>
            <div className="space-y-4">
              {recs.length === 0 ? (
                <p className="text-sm text-gray-500">No actions required.</p>
              ) : (
                recs.map((rec, i) => (
                  <RecommendationCard key={i} recommendation={rec} opportunityId={id!} />
                ))
              )}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4">Prediction History</h2>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                  <Line type="monotone" dataKey="prob" stroke="#4f46e5" strokeWidth={3} dot={{ r: 4, fill: '#4f46e5' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
