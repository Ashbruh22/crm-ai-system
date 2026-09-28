/**
 * What-if sliders.
 *
 * Nothing here is saved: `/what-if` returns the hypothetical score and the
 * baseline in one response, so the panel can show the move without writing to
 * the deal. The copy says so plainly, because a reviewer dragging a slider on a
 * live demo should not have to wonder whether they just changed data.
 *
 * Requests are debounced — a slider drag would otherwise fire a scoring call
 * per pixel.
 */
import { useEffect, useMemo, useRef, useState } from "react";

import { useWhatIf } from "../api/hooks";
import { days, pct, points } from "../lib/format";
import type { DealDetailResponse } from "../types/api";

interface Knob {
  feature: string;
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  format?: (v: number) => string;
}

/** Bounds mirror WHAT_IF_BOUNDS in app/services/scoring.py. */
const KNOBS: Knob[] = [
  {
    feature: "deal_size_k",
    label: "Deal size",
    min: 5,
    max: 1000,
    step: 5,
    format: (v) => `$${Math.round(v)}k`,
  },
  { feature: "n_stakeholders", label: "Stakeholders", min: 1, max: 12, step: 1 },
  {
    feature: "days_since_last_activity",
    label: "Days since last touch",
    min: 0,
    max: 90,
    step: 1,
    unit: " days",
  },
  { feature: "n_meetings", label: "Meetings held", min: 0, max: 15, step: 1 },
];

interface Props {
  dealId: string;
  detail: DealDetailResponse | undefined;
}

export function WhatIfPanel({ dealId, detail }: Props) {
  const whatIf = useWhatIf(dealId);
  const [values, setValues] = useState<Record<string, number> | null>(null);
  const [touched, setTouched] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  // Seed the sliders from the deal's real features so the starting position is
  // the truth rather than an arbitrary midpoint.
  const initial = useMemo(() => {
    const size = detail?.deal.deal_size ? detail.deal.deal_size / 1000 : 100;
    const drivers = detail?.latest_score?.shap_top ?? [];
    const fromDrivers = (feature: string, fallback: number) =>
      drivers.find((d) => d.feature === feature)?.value ?? fallback;

    return {
      deal_size_k: Math.round(size),
      n_stakeholders: Math.round(fromDrivers("n_stakeholders", 2)),
      days_since_last_activity: Math.round(
        fromDrivers("days_since_last_activity", 7),
      ),
      n_meetings: Math.round(fromDrivers("n_meetings", 1)),
    } as Record<string, number>;
  }, [detail]);

  useEffect(() => {
    setValues(initial);
    setTouched(false);
  }, [initial]);

  useEffect(() => {
    if (!touched || !values) return;
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => whatIf.mutate(values), 250);
    return () => window.clearTimeout(timer.current);
    // whatIf is a stable mutation object; including it would re-fire the debounce.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, touched]);

  if (!values) return null;

  const result = whatIf.data;
  const moved = touched && result;

  return (
    <section>
      <div className="flex items-baseline justify-between border-b border-rule pb-2">
        <h2 className="text-sm font-semibold">Try a change</h2>
        <span className="text-micro text-ink-faint">nothing is saved</span>
      </div>

      <div className="mt-3 space-y-3">
        {KNOBS.map((knob) => {
          const value = values[knob.feature] ?? knob.min;
          const changed = Math.abs(value - (initial[knob.feature] ?? 0)) > 0.001;
          return (
            <label key={knob.feature} className="block">
              <div className="flex items-baseline justify-between">
                <span className="text-sm text-ink-muted">{knob.label}</span>
                <span
                  className={`tnum text-sm ${changed ? "font-semibold" : "text-ink-muted"}`}
                >
                  {knob.format
                    ? knob.format(value)
                    : `${value}${knob.unit ?? ""}`}
                </span>
              </div>
              <input
                type="range"
                className="slider mt-1.5"
                min={knob.min}
                max={knob.max}
                step={knob.step}
                value={value}
                onChange={(e) => {
                  setTouched(true);
                  setValues({ ...values, [knob.feature]: Number(e.target.value) });
                }}
              />
            </label>
          );
        })}
      </div>

      <div className="mt-4 border-t border-rule pt-3">
        {!touched && (
          <p className="text-micro text-ink-faint">
            Move a slider to score a hypothetical version of this deal.
          </p>
        )}

        {whatIf.isError && (
          <p className="text-micro text-drag">
            {whatIf.error instanceof Error
              ? whatIf.error.message
              : "Could not score that change"}
          </p>
        )}

        {moved && (
          <div className="space-y-1.5">
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-ink-muted">Win probability</span>
              <span className="tnum text-sm">
                {pct(result.baseline.win_prob)}
                <span className="mx-1.5 text-ink-faint">to</span>
                <span className="font-semibold">{pct(result.what_if.win_prob)}</span>
                <span
                  className={`ml-2 ${
                    result.delta.win_prob >= 0 ? "text-lift" : "text-drag"
                  }`}
                >
                  {points(result.delta.win_prob)}
                </span>
              </span>
            </div>

            <div className="flex items-baseline justify-between">
              <span className="text-sm text-ink-muted">Days to close</span>
              <span className="tnum text-sm">
                {days(result.baseline.days_to_close)}
                <span className="mx-1.5 text-ink-faint">to</span>
                <span className="font-semibold">
                  {days(result.what_if.days_to_close)}
                </span>
              </span>
            </div>

            <button
              type="button"
              className="btn btn-quiet mt-2 px-0 text-micro text-ink-muted"
              onClick={() => {
                setValues(initial);
                setTouched(false);
                whatIf.reset();
              }}
            >
              Reset to this deal
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
