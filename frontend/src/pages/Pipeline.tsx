/**
 * The pipeline.
 *
 * A table, not a card grid: a rep scans sixty deals down a column, and the
 * shared probability scale only works if the meters line up. The three figures
 * at the top are set inline rather than in stat cards, so the table stays the
 * centre of the page.
 */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { useDeals } from "../api/hooks";
import { ProbabilityMeter } from "../components/ProbabilityMeter";
import { WakingNotice } from "../components/WakingNotice";
import { days, money } from "../lib/format";
import type { DealSummary, Priority } from "../types/api";

const STAGES = ["Prospecting", "Qualification", "Discovery", "Proposal", "Negotiation"];

const PRIORITY_MARK: Record<Priority, string> = {
  CRITICAL: "text-alarm",
  HIGH: "text-drag",
  MEDIUM: "text-ink-muted",
  LOW: "text-lift",
};

export function Pipeline() {
  const [stage, setStage] = useState<string>("");
  const { data, isLoading, isError, error } = useDeals({
    stage: stage || undefined,
    sort: "win_prob",
    order: "desc",
    limit: 100,
  });

  // Derived inside the memo, not above it: `data?.items ?? []` produces a new
  // array identity on every render, so depending on it meant the summary was
  // recomputed every time regardless.
  const summary = useMemo(() => {
    const items = data?.items ?? [];
    const scored = items.filter((d) => d.score);
    const weighted = scored.reduce(
      (sum, d) => sum + d.deal_size * (d.score?.win_prob ?? 0),
      0,
    );
    const needsAttention = items.filter(
      (d) =>
        d.top_action &&
        (d.top_action.priority === "CRITICAL" || d.top_action.priority === "HIGH"),
    ).length;
    return { open: items.length, weighted, needsAttention, scored: scored.length };
  }, [data]);

  const items = data?.items ?? [];

  if (isLoading) return <WakingNotice />;

  if (isError) {
    return (
      <div className="max-w-prose">
        <h1 className="text-lg font-semibold">The pipeline could not load</h1>
        <p className="mt-2 text-sm text-ink-muted">
          {error instanceof Error ? error.message : "The service did not respond."}
        </p>
        <button
          type="button"
          className="btn mt-4"
          onClick={() => window.location.reload()}
        >
          Try again
        </button>
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Open pipeline</h1>
          <p className="mt-1 flex flex-wrap items-baseline gap-x-5 text-sm text-ink-muted">
            <span>
              <span className="tnum text-ink">{summary.open}</span> deals
            </span>
            <span>
              <span className="tnum text-ink">{summary.scored}</span> scored
            </span>
            <span>
              <span className="tnum text-ink">{money(summary.weighted)}</span>{" "}
              weighted value
              {summary.scored < summary.open && (
                <span className="text-ink-faint"> of those scored</span>
              )}
            </span>
            <span>
              <span className="tnum text-ink">{summary.needsAttention}</span> need
              attention
            </span>
          </p>
        </div>

        <label className="text-sm">
          <span className="sr-only">Filter by stage</span>
          <select
            value={stage}
            onChange={(e) => setStage(e.target.value)}
            className="border border-rule-strong bg-paper-raised px-2 py-1.5 text-sm"
          >
            <option value="">All stages</option>
            {STAGES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
      </div>

      {summary.scored === 0 && summary.open > 0 && (
        <p className="mt-4 border-l-2 border-rule-strong pl-3 text-sm text-ink-muted">
          None of these deals has been scored yet. Open one and score it, or send
          it an event.
        </p>
      )}

      <div className="mt-5 overflow-x-auto">
        <table className="w-full min-w-[46rem] border-collapse text-left">
          <thead>
            <tr className="border-b border-ink text-micro text-ink-muted">
              <th className="py-2 pr-3 font-medium">Deal</th>
              <th className="py-2 pr-3 font-medium">Stage</th>
              <th className="py-2 pr-3 text-right font-medium">Value</th>
              <th className="py-2 pr-3 font-medium">Win probability</th>
              <th className="py-2 pr-3 text-right font-medium">Closes in</th>
              <th className="py-2 font-medium">Next action</th>
            </tr>
          </thead>
          <tbody>
            {items.map((deal) => (
              <Row key={deal.id} deal={deal} />
            ))}
          </tbody>
        </table>
      </div>

      {items.length === 0 && (
        <p className="mt-6 text-sm text-ink-muted">
          No open deals in this stage.
        </p>
      )}
    </div>
  );
}

function Row({ deal }: { deal: DealSummary }) {
  return (
    <tr className="border-b border-rule/60 align-middle hover:bg-paper-raised">
      <td className="py-2.5 pr-3">
        <Link
          to={`/deals/${deal.id}`}
          className="text-sm font-medium underline decoration-rule-strong decoration-1 underline-offset-2 hover:decoration-ink"
        >
          {deal.company}
        </Link>
        <div className="text-micro text-ink-faint">{deal.owner_rep}</div>
      </td>

      <td className="py-2.5 pr-3 text-sm text-ink-muted">{deal.stage}</td>

      <td className="tnum py-2.5 pr-3 text-right text-sm">
        {money(deal.deal_size)}
      </td>

      <td className="py-2.5 pr-3">
        {deal.score ? (
          <ProbabilityMeter value={deal.score.win_prob} inline />
        ) : (
          <span className="text-sm text-ink-faint">not scored</span>
        )}
      </td>

      <td className="tnum py-2.5 pr-3 text-right text-sm text-ink-muted">
        {deal.score?.days_to_close != null ? days(deal.score.days_to_close) : "—"}
      </td>

      <td className="py-2.5 text-sm">
        {deal.top_action ? (
          <span className="flex items-baseline gap-2">
            <span
              className={`text-micro font-semibold ${PRIORITY_MARK[deal.top_action.priority]}`}
            >
              {deal.top_action.priority.toLowerCase()}
            </span>
            <span className="text-ink-muted">
              {deal.top_action.action_type.replace(/_/g, " ")}
            </span>
          </span>
        ) : (
          <span className="text-ink-faint">—</span>
        )}
      </td>
    </tr>
  );
}
