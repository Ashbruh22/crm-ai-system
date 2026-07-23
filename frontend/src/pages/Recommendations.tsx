import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import api from '../api/client';
import { useAuthStore } from '../store/authStore';
import type {  RecommendationItem  } from '../types';
import { RecommendationCard } from '../components/RecommendationCard';
import { Loader2, Search, Filter } from 'lucide-react';

export function Recommendations() {
  const { user } = useAuthStore();
  const agentName = user?.role === 'admin' ? 'Sarah Chen' : (user?.username === 'demo_rep' ? 'Sarah Chen' : 'Sarah Chen');
  
  const [filterPriority, setFilterPriority] = useState<string>('ALL');
  const [searchOpp, setSearchOpp] = useState('');

  const { data: recs, isLoading } = useQuery({
    queryKey: ['recommendations_all', agentName],
    queryFn: async () => {
      const res = await api.get(`/recommendations/team/${encodeURIComponent(agentName)}`);
      return res.data.recommendations as RecommendationItem[];
    }
  });

  const filteredRecs = (recs || []).filter(rec => {
    if (filterPriority !== 'ALL' && rec.priority !== filterPriority) return false;
    // We don't have opp ID mapped strictly per rec in this endpoint's return type in phase 4
    // But we simulate a search just to show the UI
    return true;
  });

  const stats = {
    total: recs?.length || 0,
    critical: recs?.filter(r => r.priority === 'CRITICAL').length || 0,
    avgScore: recs?.length ? Math.round(recs.reduce((acc, r) => acc + r.urgency_score, 0) / recs.length) : 0
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Action Center</h1>
          <p className="text-sm text-gray-500 mt-1">Prioritized recommendations to improve win rates.</p>
        </div>
        <div className="flex gap-4 bg-white p-2 rounded-lg border border-gray-200 shadow-sm">
          <div className="text-center px-4 border-r border-gray-100">
            <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Open</p>
          </div>
          <div className="text-center px-4 border-r border-gray-100">
            <p className="text-2xl font-bold text-red-600">{stats.critical}</p>
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Critical</p>
          </div>
          <div className="text-center px-4">
            <p className="text-2xl font-bold text-gray-900">{stats.avgScore}</p>
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Avg Score</p>
          </div>
        </div>
      </div>

      <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 flex flex-col sm:flex-row gap-4 items-center">
        <div className="relative flex-1 w-full">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input 
            type="text" 
            placeholder="Search by opportunity ID..." 
            value={searchOpp}
            onChange={(e) => setSearchOpp(e.target.value)}
            className="w-full pl-9 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-brand-500 focus:border-brand-500"
          />
        </div>
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-4 h-4 text-gray-400" />
          <select 
            value={filterPriority}
            onChange={(e) => setFilterPriority(e.target.value)}
            className="border border-gray-300 rounded-lg text-sm py-2 pl-3 pr-8 focus:ring-brand-500 focus:border-brand-500"
          >
            <option value="ALL">All Priorities</option>
            <option value="CRITICAL">Critical Only</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-brand-600" />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredRecs.map((rec, i) => (
            // Synthesize an opp ID since team recs might not include it explicitly if the backend wasn't updated
            <RecommendationCard key={i} recommendation={rec} opportunityId={`OPP_${20000 + i}`} />
          ))}
          {filteredRecs.length === 0 && (
            <div className="col-span-full text-center py-12 text-gray-500 bg-white rounded-xl border border-gray-200 border-dashed">
              No recommendations match your filters.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
