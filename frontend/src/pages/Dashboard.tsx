import { useQuery } from '@tanstack/react-query';
import { useAuthStore } from '../store/authStore';
import api from '../api/client';
import type {  RecommendationItem  } from '../types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Link } from 'react-router-dom';
import { PriorityBadge } from '../components/PriorityBadge';
import { WinProbabilityBadge } from '../components/WinProbabilityBadge';
import { TrendingUp, Users, Target, AlertTriangle } from 'lucide-react';

export function Dashboard() {
  const { user } = useAuthStore();
  const agentName = user?.role === 'admin' ? 'Sarah Chen' : (user?.username === 'demo_rep' ? 'Sarah Chen' : 'Sarah Chen'); // Defaulting for demo

  const { data: recsData, isLoading } = useQuery({
    queryKey: ['recommendations', agentName],
    queryFn: async () => {
      const res = await api.get(`/recommendations/team/${encodeURIComponent(agentName)}`);
      return res.data.recommendations as RecommendationItem[];
    }
  });

  // Mocking some dashboard aggregate data since we don't have a bulk API endpoint
  const stats = {
    totalDeals: recsData?.length || 12,
    avgWinProb: 45.2,
    criticalDeals: recsData?.filter(r => r.priority === 'CRITICAL').length || 3,
    closingThisWeek: 4
  };

  const pipelineData = [
    { name: '0-25%', count: 2, fill: '#ef4444' },
    { name: '25-50%', count: 5, fill: '#f59e0b' },
    { name: '50-75%', count: 3, fill: '#10b981' },
    { name: '75-100%', count: 2, fill: '#059669' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Pipeline Overview</h1>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center text-blue-600">
            <Target className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">Active Deals</p>
            <p className="text-2xl font-bold text-gray-900">{stats.totalDeals}</p>
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-brand-50 flex items-center justify-center text-brand-600">
            <TrendingUp className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">Avg Win Probability</p>
            <p className="text-2xl font-bold text-gray-900">{stats.avgWinProb}%</p>
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-red-50 flex items-center justify-center text-red-600">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">CRITICAL Priority</p>
            <p className="text-2xl font-bold text-gray-900">{stats.criticalDeals}</p>
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-green-50 flex items-center justify-center text-green-600">
            <Users className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">Closing This Week</p>
            <p className="text-2xl font-bold text-gray-900">{stats.closingThisWeek}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pipeline Health Chart */}
        <div className="lg:col-span-1 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-6">Pipeline Health</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pipelineData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6b7280' }} />
                <Tooltip cursor={{fill: '#f9fafb'}} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {pipelineData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Deals Table */}
        <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-bold text-gray-900">Recent Deals Requiring Action</h2>
          </div>
          <div className="flex-1 overflow-x-auto">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-gray-50 text-gray-500 font-medium">
                <tr>
                  <th className="px-6 py-4">Opportunity</th>
                  <th className="px-6 py-4">Agent</th>
                  <th className="px-6 py-4">Win Prob</th>
                  <th className="px-6 py-4">Priority</th>
                  <th className="px-6 py-4">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 text-gray-900">
                {isLoading ? (
                  <tr><td colSpan={5} className="px-6 py-8 text-center text-gray-500">Loading deals...</td></tr>
                ) : (recsData || []).slice(0, 5).map((rec: any, idx) => {
                  // Synthesize opp ID from list since endpoint doesn't return it directly in the item
                  // For demo, we'll use a mocked ID list
                  const oppId = `OPP_${10000 + idx}`;
                  return (
                    <tr key={idx} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 font-medium text-brand-600">
                        <Link to={`/opportunities/${oppId}`}>{oppId}</Link>
                      </td>
                      <td className="px-6 py-4">{agentName}</td>
                      <td className="px-6 py-4">
                        <WinProbabilityBadge 
                          probability={
                            rec.priority === 'CRITICAL' ? 0.25 + (Math.random() * 0.1) :
                            rec.priority === 'HIGH' ? 0.35 + (Math.random() * 0.1) :
                            rec.priority === 'MEDIUM' ? 0.50 + (Math.random() * 0.1) :
                            0.75 + (Math.random() * 0.1)
                          } 
                        />
                      </td>
                      <td className="px-6 py-4">
                        <PriorityBadge priority={rec.priority} />
                      </td>
                      <td className="px-6 py-4">
                        <Link to={`/opportunities/${oppId}`} className="text-brand-600 hover:text-brand-700 font-medium text-xs">
                          View Details &rarr;
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
