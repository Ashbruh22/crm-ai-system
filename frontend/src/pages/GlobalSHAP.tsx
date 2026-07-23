import { useQuery } from '@tanstack/react-query';
import api from '../api/client';
import type {  GlobalImportanceResponse  } from '../types';
import { FeatureImportanceBar } from '../components/FeatureImportanceBar';
import { Loader2, Info } from 'lucide-react';

export function GlobalSHAP() {
  const { data, isLoading } = useQuery({
    queryKey: ['global_shap'],
    queryFn: async () => {
      const res = await api.get('/explain/global');
      return res.data as GlobalImportanceResponse;
    }
  });

  const rankings = data?.rankings || [];
  const maxImportance = rankings.length > 0 ? rankings[0].mean_abs_shap : 0;
  const topFeature = rankings.length > 0 ? rankings[0].feature.replace(/_/g, ' ') : 'N/A';
  
  // Calculate percentage of total importance for the top feature
  const totalImportance = rankings.reduce((sum, r) => sum + r.mean_abs_shap, 0);
  const topPercentage = totalImportance > 0 ? ((maxImportance / totalImportance) * 100).toFixed(1) : '0.0';

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Feature Impact (Global SHAP)</h1>
        <p className="text-sm text-gray-500 mt-1">
          Understanding which factors drive model predictions across the entire sales pipeline.
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200 bg-gray-50/50">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-brand-600 shrink-0 mt-0.5" />
            <p className="text-sm text-gray-700 leading-relaxed">
              The most influential factor in deal outcome predictions is <span className="font-bold">{topFeature}</span>, 
              contributing <span className="font-bold text-brand-700">{topPercentage}%</span> of total model importance. 
              Agent historical performance and product complexity are the strongest predictors overall.
            </p>
          </div>
        </div>

        <div className="p-6">
          <div className="flex text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4 border-b border-gray-100 pb-2">
            <div className="w-1/3 text-right pr-4">Feature Name</div>
            <div className="w-2/3 pl-3">Mean |SHAP| Impact</div>
          </div>

          {isLoading ? (
            <div className="flex justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-brand-600" />
            </div>
          ) : (
            <div className="space-y-1">
              {rankings.map((r) => (
                <FeatureImportanceBar 
                  key={r.feature} 
                  featureName={r.feature} 
                  importance={r.mean_abs_shap} 
                  maxImportance={maxImportance} 
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
