

interface Props {
  featureName: string;
  importance: number;
  maxImportance: number;
}

export function FeatureImportanceBar({ featureName, importance, maxImportance }: Props) {
  const percentage = maxImportance > 0 ? (importance / maxImportance) * 100 : 0;
  
  // Categorize colors based on feature name heuristics for demo purposes
  let barColor = 'bg-gray-400';
  if (featureName.includes('agent')) barColor = 'bg-blue-500';
  else if (featureName.includes('product') || featureName.includes('price')) barColor = 'bg-green-500';
  else if (featureName.includes('day') || featureName.includes('month')) barColor = 'bg-purple-500';

  return (
    <div className="flex items-center gap-4 py-2">
      <div className="w-1/3 text-right">
        <span className="text-sm font-medium text-gray-700 capitalize">
          {featureName.replace(/_/g, ' ')}
        </span>
      </div>
      <div className="w-2/3 flex items-center gap-3">
        <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full ${barColor}`} 
            style={{ width: `${percentage}%` }}
          />
        </div>
        <div className="w-16 text-xs text-gray-500 font-mono text-right">
          {importance.toFixed(3)}
        </div>
      </div>
    </div>
  );
}
