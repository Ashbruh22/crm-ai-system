/**
 * The additive SHAP ledger.
 *
 * Most dashboards show feature importance as an unordered bar chart, which says
 * "these mattered" and stops there. The service computes exact TreeSHAP, so
 * `base + Σ contributions` reproduces the model's margin to ~3e-07 — this
 * renders that as a running total that visibly lands on the prediction, because
 * the arithmetic closing is the interesting claim.
 *
 * Values are log-odds. They are labelled as such rather than converted to
 * percentages, because contributions are only additive in log-odds space;
 * showing "+12%" per feature would be a comfortable lie.
 */
import { useMemo } from "react";

import type { ShapDriver } from "../types/api";
import { pct } from "../lib/format";

interface Props {
  drivers: ShapDriver[];
  baseValue: number;
  winProb: number;
  /** Features whose contribution changed on the last update, to mark. */
  changed?: Set<string>;
  limit?: number;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function ContributionLedger({
  drivers,
  baseValue,
  winProb,
  changed,
  limit = 8,
}: Props) {
  const rows = useMemo(
    () => [...drivers].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)).slice(0, limit),
    [drivers, limit],
  );

  const shown = rows.reduce((sum, d) => sum + d.shap, 0);
  const rest = drivers.reduce((sum, d) => sum + d.shap, 0) - shown;
  const margin = baseValue + shown + rest;

  // One shared scale so bar lengths are comparable between rows.
  const scale = Math.max(0.25, ...rows.map((d) => Math.abs(d.shap)));

  return (
    <div>
      <div className="flex items-baseline justify-between border-b border-rule pb-2">
        <h2 className="text-sm font-semibold">Why this score</h2>
        <span className="text-micro text-ink-faint">contributions in log-odds</span>
      </div>

      <table className="w-full border-collapse">
        <tbody>
          <tr className="border-b border-rule/60">
            <td className="py-2 pr-3 text-sm text-ink-muted">
              Starting point
              <span className="ml-1.5 text-micro text-ink-faint">
                average deal
              </span>
            </td>
            <td className="w-32" />
            <td className="tnum py-2 pl-3 text-right text-sm text-ink-muted">
              {baseValue.toFixed(2)}
            </td>
          </tr>

          {rows.map((d) => {
            const up = d.shap >= 0;
            const width = `${(Math.abs(d.shap) / scale) * 100}%`;
            return (
              <tr
                key={d.feature}
                className={`border-b border-rule/60 ${
                  changed?.has(d.feature) ? "just-updated" : ""
                }`}
              >
                <td className="py-1.5 pr-3 align-middle">
                  <div className="text-sm leading-tight">{d.label}</div>
                  <div className="tnum text-micro text-ink-faint">
                    {formatFeatureValue(d.feature, d.value)}
                  </div>
                </td>

                {/* Diverging bar: zero is the centre line, so direction is
                    readable without consulting the sign. */}
                <td className="w-32 align-middle">
                  <div className="relative h-4">
                    <div className="absolute inset-y-0 left-1/2 w-px bg-rule-strong" />
                    <div
                      className={`absolute top-1/2 h-2.5 -translate-y-1/2 transition-[width] duration-500 ease-meter ${
                        up ? "left-1/2 bg-lift" : "right-1/2 bg-drag"
                      }`}
                      style={{ width: `calc(${width} / 2)` }}
                    />
                  </div>
                </td>

                <td
                  className={`tnum py-1.5 pl-3 text-right text-sm ${
                    up ? "text-lift" : "text-drag"
                  }`}
                >
                  {up ? "+" : "−"}
                  {Math.abs(d.shap).toFixed(2)}
                </td>
              </tr>
            );
          })}

          {Math.abs(rest) > 0.005 && (
            <tr className="border-b border-rule/60">
              <td className="py-2 pr-3 text-sm text-ink-muted">
                {drivers.length - rows.length} smaller factors
              </td>
              <td className="w-32" />
              <td className="tnum py-2 pl-3 text-right text-sm text-ink-muted">
                {rest >= 0 ? "+" : "−"}
                {Math.abs(rest).toFixed(2)}
              </td>
            </tr>
          )}

          <tr className="border-b-2 border-ink">
            <td className="py-2 pr-3 text-sm font-semibold">Total</td>
            <td className="w-32" />
            <td className="tnum py-2 pl-3 text-right text-sm font-semibold">
              {margin >= 0 ? "+" : "−"}
              {Math.abs(margin).toFixed(2)}
            </td>
          </tr>

          <tr>
            <td className="pt-2 pr-3 text-sm text-ink-muted">
              Win probability
              <span className="ml-1.5 text-micro text-ink-faint">
                from the total
              </span>
            </td>
            <td className="w-32" />
            <td className="tnum pt-2 pl-3 text-right text-base font-semibold">
              {pct(sigmoid(margin))}
            </td>
          </tr>
        </tbody>
      </table>

      {Math.abs(sigmoid(margin) - winProb) > 0.01 && (
        <p className="mt-2 text-micro text-drag">
          Shown total differs from the served score by{" "}
          {pct(Math.abs(sigmoid(margin) - winProb), 1)}; the score was computed
          from a newer feature snapshot.
        </p>
      )}
    </div>
  );
}

/** Render a feature's raw value in units a reader recognises. */
function formatFeatureValue(feature: string, value: number): string {
  if (feature === "has_champion" || feature === "discount_requested_late") {
    return value >= 1 ? "yes" : "no";
  }
  if (feature === "deal_size_k") return `$${Math.round(value)}k`;
  if (feature === "reply_rate") return pct(value);
  if (feature.startsWith("days")) return `${Math.round(value)} days`;
  if (feature.startsWith("industry_") || feature.startsWith("region_")) {
    return value >= 1 ? "yes" : "no";
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}
