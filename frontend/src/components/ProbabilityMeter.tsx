/**
 * Win probability as an instrument reading, not a stat card.
 *
 * The labelled scale is the point: a bare percentage in a big font tells you
 * nothing about where it sits, while a needle against ticks is readable at a
 * glance and comparable between deals. `inline` is the compact form used in the
 * pipeline table, where the same scale lets a whole column be scanned down.
 */
import { pct, points } from "../lib/format";

interface Props {
  value: number;
  /** Previous value, to draw the move. */
  previous?: number | null;
  inline?: boolean;
  animate?: boolean;
}

const TICKS = [0, 25, 50, 75, 100];

function band(value: number): string {
  if (value >= 0.7) return "bg-lift";
  if (value >= 0.45) return "bg-ink-muted";
  return "bg-drag";
}

export function ProbabilityMeter({ value, previous, inline, animate }: Props) {
  const clamped = Math.min(1, Math.max(0, value));
  const moved =
    previous !== null && previous !== undefined && Math.abs(value - previous) >= 0.005;

  if (inline) {
    return (
      <div className="flex items-center gap-2">
        <div className="relative h-1.5 w-20 bg-paper-sunken" aria-hidden="true">
          <div
            className={`absolute inset-y-0 left-0 ${band(clamped)} transition-[width] duration-700 ease-meter`}
            style={{ width: `${clamped * 100}%` }}
          />
        </div>
        <span className="tnum w-9 text-right text-sm">{pct(clamped)}</span>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-end gap-3">
        <span
          className={`tnum text-5xl font-semibold leading-none tracking-tight ${
            animate ? "just-updated" : ""
          }`}
        >
          {pct(clamped)}
        </span>
        {moved && (
          <span
            className={`tnum pb-1 text-sm ${
              value > (previous ?? 0) ? "text-lift" : "text-drag"
            }`}
          >
            {points(value - (previous ?? 0))}
          </span>
        )}
      </div>

      <div className="mt-3">
        <div
          className="relative h-3 bg-paper-sunken"
          role="meter"
          aria-valuenow={Math.round(clamped * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Win probability"
        >
          <div
            className={`absolute inset-y-0 left-0 ${band(clamped)} transition-[width] duration-700 ease-meter`}
            style={{ width: `${clamped * 100}%` }}
          />
          {previous !== null && previous !== undefined && moved && (
            <div
              className="absolute inset-y-0 w-px bg-ink"
              style={{ left: `${Math.min(1, Math.max(0, previous)) * 100}%` }}
              title={`was ${pct(previous)}`}
            />
          )}
        </div>

        {/* Ticks make it a reading rather than a decorative bar. */}
        <div className="relative mt-1 h-4">
          {TICKS.map((t) => (
            <div
              key={t}
              className="absolute top-0 -translate-x-1/2"
              style={{ left: `${t}%` }}
            >
              <div className="mx-auto h-1 w-px bg-rule-strong" />
              <div className="tnum text-micro text-ink-faint">{t}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
