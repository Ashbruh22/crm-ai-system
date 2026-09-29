/**
 * Win probability over the deal's scoring history.
 *
 * A step line, not a smooth curve: the score changes discretely when an event
 * lands, and interpolating between points would imply readings that were never
 * taken. Lazy-loaded by the deal page because it pulls in Recharts.
 */
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { pct } from "../lib/format";

interface Props {
  history: Array<{ win_prob: number; days_to_close: number | null; scored_at: string }>;
}

export function ScoreHistory({ history }: Props) {
  const data = history.map((point, index) => ({
    index,
    prob: point.win_prob * 100,
    at: new Date(point.scored_at),
  }));

  return (
    <div className="mt-3 h-40">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid stroke="#d3dae2" strokeDasharray="2 3" vertical={false} />
          {/* 50% is the decision boundary, so it earns a rule of its own. */}
          <ReferenceLine y={50} stroke="#b3bec9" strokeWidth={1} />
          <XAxis
            dataKey="index"
            tick={{ fill: "#8b9aa8", fontSize: 11 }}
            stroke="#b3bec9"
            tickFormatter={(i: number) => (i === 0 ? "first" : i === data.length - 1 ? "now" : "")}
          />
          <YAxis
            domain={[0, 100]}
            ticks={[0, 50, 100]}
            tick={{ fill: "#8b9aa8", fontSize: 11 }}
            stroke="#b3bec9"
            tickFormatter={(v: number) => `${v}`}
          />
          <Tooltip
            contentStyle={{
              background: "#f7f9fb",
              border: "1px solid #b3bec9",
              borderRadius: 2,
              fontSize: 12,
            }}
            labelFormatter={(_, payload) => {
              const at = payload?.[0]?.payload?.at as Date | undefined;
              return at ? at.toLocaleString() : "";
            }}
            formatter={(value) => [pct(Number(value) / 100), "win probability"]}
          />
          <Line
            type="stepAfter"
            dataKey="prob"
            stroke="#0f766e"
            strokeWidth={1.75}
            dot={{ r: 2, fill: "#0f766e", strokeWidth: 0 }}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
