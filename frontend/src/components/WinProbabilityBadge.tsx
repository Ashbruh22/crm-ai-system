import { twMerge } from 'tailwind-merge';

interface Props {
  probability: number;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function WinProbabilityBadge({ probability, size = 'sm', className }: Props) {
  const probPercent = Math.round(probability * 100);
  
  let colorClass = '';
  if (probability < 0.4) colorClass = 'bg-red-100 text-red-700 border-red-200';
  else if (probability < 0.65) colorClass = 'bg-amber-100 text-amber-700 border-amber-200';
  else colorClass = 'bg-green-100 text-green-700 border-green-200';

  if (size === 'lg') {
    // Large gauge style
    const strokeDasharray = `${probPercent} 100`;
    let strokeColor = '';
    if (probability < 0.4) strokeColor = 'text-red-500';
    else if (probability < 0.65) strokeColor = 'text-amber-500';
    else strokeColor = 'text-green-500';

    return (
      <div className={twMerge("flex flex-col items-center justify-center", className)}>
        <svg viewBox="0 0 36 36" className="w-32 h-32 transform -rotate-90">
          <path
            className="text-gray-200"
            strokeWidth="3"
            stroke="currentColor"
            fill="none"
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
          />
          <path
            className={strokeColor}
            strokeDasharray={strokeDasharray}
            strokeWidth="3"
            stroke="currentColor"
            fill="none"
            strokeLinecap="round"
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="text-3xl font-bold text-gray-900">{probPercent}%</span>
          <span className="text-xs text-gray-500 uppercase tracking-wider">Win Prob</span>
        </div>
      </div>
    );
  }

  return (
    <span className={twMerge(
      "inline-flex items-center justify-center border font-medium",
      size === 'sm' ? "px-2 py-0.5 rounded text-xs" : "w-10 h-10 rounded-full text-sm",
      colorClass,
      className
    )}>
      {probPercent}%
    </span>
  );
}
