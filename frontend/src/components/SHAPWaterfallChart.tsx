import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

interface Props {
  baseValue: number;
  outputValue: number;
  features: { name: string; value: number; shap: number; cumulative: number }[];
}

export function SHAPWaterfallChart({ baseValue, outputValue, features }: Props) {
  // Format data for Recharts stacked bar approach
  // To draw floating bars, we use a transparent "filler" bar up to the starting point, 
  // and a colored bar for the actual value.

  const formattedData = [
    {
      name: 'Base Value',
      start: 0,
      size: baseValue,
      shap: baseValue,
      isBase: true,
      fill: '#94a3b8' // gray-400
    },
    ...features.map(f => {
      // Recharts needs positive sizes for the second bar in a stack if we're stacking on top of 'start'.
      // A trick for waterfall in recharts: 
      // bar1 = min(start, end) [transparent]
      // bar2 = abs(end - start) [colored]
      
      const start = f.cumulative - f.shap;
      const end = f.cumulative;
      
      return {
        name: f.name.replace(/_/g, ' '),
        start: Math.min(start, end),
        size: Math.abs(f.shap),
        shap: f.shap,
        fill: f.shap > 0 ? '#ef4444' : '#3b82f6' // red for positive, blue for negative
      };
    }),
    {
      name: 'Final Prediction',
      start: 0,
      size: outputValue,
      shap: outputValue,
      isTotal: true,
      fill: '#4f46e5' // brand-600
    }
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      if (data.isBase || data.isTotal) {
        return (
          <div className="bg-white p-3 border border-gray-200 shadow-sm rounded-lg text-sm">
            <p className="font-semibold text-gray-900">{data.name}</p>
            <p className="text-gray-600">{data.shap.toFixed(3)}</p>
          </div>
        );
      }
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-sm rounded-lg text-sm">
          <p className="font-semibold text-gray-900 capitalize">{data.name}</p>
          <p className="text-gray-600 flex items-center gap-1">
            Impact: 
            <span className={data.shap > 0 ? 'text-red-600 font-medium' : 'text-blue-600 font-medium'}>
              {data.shap > 0 ? '+' : ''}{data.shap.toFixed(3)}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          layout="vertical"
          data={formattedData}
          margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" />
          <XAxis type="number" hide />
          <YAxis 
            dataKey="name" 
            type="category" 
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#4b5563', fontSize: 12, className: 'capitalize' }}
            width={120}
          />
          <Tooltip cursor={{fill: '#f3f4f6'}} content={<CustomTooltip />} />
          
          {/* Transparent spacer bar */}
          <Bar dataKey="start" stackId="a" fill="transparent" />
          {/* Visible size bar */}
          <Bar dataKey="size" stackId="a" radius={[0, 4, 4, 0]}>
            {formattedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
