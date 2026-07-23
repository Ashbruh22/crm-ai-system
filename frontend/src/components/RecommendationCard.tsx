import { useState } from 'react';
import { PriorityBadge } from './PriorityBadge';
import type {  RecommendationItem  } from '../types';
import { CheckCircle2, ChevronDown, ChevronUp } from 'lucide-react';
import api from '../api/client';

interface Props {
  recommendation: RecommendationItem;
  opportunityId: string;
}

export function RecommendationCard({ recommendation, opportunityId }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleMarkDone = async () => {
    setLoading(true);
    try {
      await api.post('/recommendations/feedback', {
        prediction_id: opportunityId, // Assuming prediction ID maps 1:1 with opportunity for demo
        adopted_at: new Date().toISOString(),
        outcome_delta: 0 // Cannot know true outcome delta at adoption time
      });
      setDone(true);
    } catch (err) {
      console.error('Failed to mark done', err);
    } finally {
      setLoading(false);
    }
  };

  if (done) {
    return (
      <div className="rounded-xl border border-green-200 bg-green-50 p-4 flex items-center gap-3 opacity-60">
        <CheckCircle2 className="w-5 h-5 text-green-600" />
        <span className="text-sm font-medium text-green-800 line-through">{recommendation.action}</span>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm hover:border-gray-300 transition-colors">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <PriorityBadge priority={recommendation.priority} />
            <span className="text-xs text-gray-500 font-medium tracking-wide">SCORE: {Math.round(recommendation.urgency_score)}</span>
          </div>
          <h4 className="text-sm font-semibold text-gray-900 leading-snug">{recommendation.action}</h4>
        </div>
        <button 
          onClick={handleMarkDone}
          disabled={loading}
          className="shrink-0 p-1.5 rounded-full hover:bg-gray-100 text-gray-400 hover:text-green-600 transition-colors focus:outline-none focus:ring-2 focus:ring-brand-500"
          title="Mark as done"
        >
          <CheckCircle2 className="w-5 h-5" />
        </button>
      </div>

      <button 
        onClick={() => setExpanded(!expanded)}
        className="mt-3 flex items-center gap-1 text-xs font-medium text-brand-600 hover:text-brand-700"
      >
        {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        {expanded ? 'Hide details' : 'Show rationale'}
      </button>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-100 space-y-3">
          <div>
            <h5 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-1">Why this action?</h5>
            <p className="text-xs text-gray-600 leading-relaxed">{recommendation.rationale}</p>
          </div>
          {recommendation.expected_impact && (
            <div>
              <h5 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-1">Expected Impact</h5>
              <p className="text-xs text-gray-600 leading-relaxed">{recommendation.expected_impact}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
