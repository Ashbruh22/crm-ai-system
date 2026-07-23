import { CalendarClock, Zap } from 'lucide-react';

interface Props {
  predictedDaysRemaining: number;
  daysElapsed?: number; // Optional, defaults to 0 for demo if not known
}

export function CycleForecastCard({ predictedDaysRemaining, daysElapsed = 0 }: Props) {
  const totalDays = daysElapsed + predictedDaysRemaining;
  const progressPercent = totalDays > 0 ? Math.min(100, (daysElapsed / totalDays) * 100) : 0;

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm relative overflow-hidden">
      {/* Decorative background element */}
      <div className="absolute -right-4 -top-4 text-gray-50 opacity-50 pointer-events-none">
        <CalendarClock className="w-32 h-32" />
      </div>

      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-2 bg-brand-50 rounded-lg">
            <Zap className="w-5 h-5 text-brand-600" />
          </div>
          <h3 className="text-sm font-semibold text-gray-900 tracking-tight uppercase">AI Cycle Forecast</h3>
        </div>

        <div className="flex items-end gap-2 mb-6">
          <span className="text-4xl font-black text-gray-900 leading-none">{predictedDaysRemaining}</span>
          <span className="text-sm font-medium text-gray-500 pb-1">days remaining</span>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-xs font-medium text-gray-500">
            <span>{daysElapsed} days elapsed</span>
            <span>Est. Total: {totalDays} days</span>
          </div>
          <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden flex">
            <div 
              className="h-full bg-gray-300 rounded-l-full" 
              style={{ width: `${progressPercent}%` }}
            />
            <div 
              className="h-full bg-brand-500 relative" 
              style={{ width: `${100 - progressPercent}%` }}
            >
              <div className="absolute inset-0 bg-white/20 animate-pulse" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
