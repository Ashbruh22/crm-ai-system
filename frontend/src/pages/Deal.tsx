/**
 * Deal detail: the readout on the left, the controls on the right.
 *
 * The page holds the previous win probability so an SSE update can draw the
 * move rather than silently swapping the number — seeing 51% become 59% is the
 * whole demonstration, and a number that just changes tells you nothing.
 */
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { useDeal, useExplain, useScoreNow, useScoreStream } from "../api/hooks";
import { ActionList } from "../components/ActionList";
import { ContributionLedger } from "../components/ContributionLedger";
import { ProbabilityMeter } from "../components/ProbabilityMeter";
import { SimulateEvents } from "../components/SimulateEvents";
import { WakingNotice } from "../components/WakingNotice";
import { WhatIfPanel } from "../components/WhatIfPanel";
import { activityLabel, ago, days, money, ms } from "../lib/format";

// Recharts is the heaviest thing on the page and only the history chart needs
// it, so it loads after the score is on screen.
const ScoreHistory = lazy(() =>
  import("../components/ScoreHistory").then((m) => ({ default: m.ScoreHistory })),
);

export function Deal() {
  const { id = "" } = useParams();
  const { data, isLoading, isError, error } = useDeal(id);
  const explain = useExplain(id, Boolean(data?.latest_score));
  const scoreNow = useScoreNow(id);

  const [previous, setPrevious] = useState<number | null>(null);
  const [updateToken, setUpdateToken] = useState(0);
  const [changed, setChanged] = useState<Set<string>>(new Set());
  const lastProb = useRef<number | null>(null);

  // Track the score we last rendered, so an update can be compared against it.
  const current = data?.latest_score?.win_prob ?? null;
  useEffect(() => {
    if (current !== null && lastProb.current === null) lastProb.current = current;
  }, [current]);

  useScoreStream((event) => {
    if (event.deal_id !== id) return;
    setPrevious(lastProb.current);
    lastProb.current = event.win_prob;
    setChanged(new Set(event.shap_top.slice(0, 3).map((d) => d.feature)));
    setUpdateToken((n) => n + 1);
  });

  const drivers = explain.data?.shap_values ?? data?.latest_score?.shap_top ?? [];
  const baseValue = explain.data?.shap_base_value;

  const timeline = useMemo(() => data?.timeline.slice(0, 12) ?? [], [data]);

  if (isLoading) return <WakingNotice label="Loading the deal" />;

  if (isError || !data) {
    return (
      <div className="max-w-prose">
        <h1 className="text-lg font-semibold">This deal could not load</h1>
        <p className="mt-2 text-sm text-ink-muted">
          {error instanceof Error ? error.message : "The service did not respond."}
        </p>
        <Link to="/" className="btn mt-4">
          Back to the pipeline
        </Link>
      </div>
    );
  }

  const { deal, latest_score, score_history, actions } = data;

  return (
    <div>
      <Link
        to="/"
        className="text-sm text-ink-muted underline decoration-rule-strong underline-offset-2 hover:text-ink"
      >
        Pipeline
      </Link>

      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h1 className="text-xl font-semibold tracking-tight">{deal.company}</h1>
        <p className="flex flex-wrap items-baseline gap-x-4 text-sm text-ink-muted">
          <span>{deal.stage}</span>
          <span className="tnum">{money(deal.deal_size)}</span>
          <span>{deal.owner_rep}</span>
          <span>{deal.industry}</span>
        </p>
      </div>

      <div className="mt-6 grid gap-x-10 gap-y-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
        {/* ---------- readout ---------- */}
        <div className="space-y-8">
          <section>
            {latest_score ? (
              <>
                <div className="flex flex-wrap items-start justify-between gap-6">
                  <div className="min-w-[14rem] flex-1">
                    <h2 className="text-sm font-semibold">Win probability</h2>
                    <div className="mt-2">
                      <ProbabilityMeter
                        value={latest_score.win_prob}
                        previous={previous}
                        animate={updateToken > 0}
                      />
                    </div>
                  </div>

                  <dl className="grid shrink-0 grid-cols-2 gap-x-6 gap-y-2 text-sm">
                    <dt className="text-ink-muted">Closes in</dt>
                    <dd className="tnum text-right">
                      {days(latest_score.days_to_close)}
                    </dd>
                    <dt className="text-ink-muted">Scored</dt>
                    <dd className="text-right">{ago(latest_score.scored_at)}</dd>
                    <dt className="text-ink-muted">Took</dt>
                    <dd className="tnum text-right">
                      {ms(latest_score.latency_ms?.total)}
                      {latest_score.cache_hit && (
                        <span className="ml-1 text-micro text-ink-faint">cached</span>
                      )}
                    </dd>
                    <dt className="text-ink-muted">Model</dt>
                    <dd className="tnum text-right text-micro text-ink-muted">
                      {latest_score.model_version}
                    </dd>
                  </dl>
                </div>

                {latest_score.latency_ms && (
                  <p className="mt-4 flex flex-wrap gap-x-4 text-micro text-ink-faint">
                    {(["features", "xgb", "lstm", "shap", "nba"] as const).map(
                      (stage) =>
                        latest_score.latency_ms?.[stage] != null && (
                          <span key={stage} className="tnum">
                            {stage} {ms(latest_score.latency_ms?.[stage])}
                          </span>
                        ),
                    )}
                  </p>
                )}
              </>
            ) : (
              <div>
                <h2 className="text-sm font-semibold">Not scored yet</h2>
                <p className="mt-1 max-w-prose text-sm text-ink-muted">
                  This deal has activity but no score. Scoring builds its
                  features, runs both models, and generates recommendations.
                </p>
                <button
                  type="button"
                  className="btn mt-3"
                  disabled={scoreNow.isPending}
                  onClick={() => scoreNow.mutate()}
                >
                  {scoreNow.isPending ? "Scoring…" : "Score this deal"}
                </button>
                {scoreNow.isError && (
                  <p className="mt-2 text-micro text-drag">
                    {scoreNow.error instanceof Error
                      ? scoreNow.error.message
                      : "Scoring failed"}
                  </p>
                )}
              </div>
            )}
          </section>

          {latest_score && drivers.length > 0 && baseValue !== undefined && (
            <ContributionLedger
              drivers={drivers}
              baseValue={baseValue}
              winProb={latest_score.win_prob}
              changed={changed}
            />
          )}

          {score_history.length > 1 && (
            <section>
              <h2 className="border-b border-rule pb-2 text-sm font-semibold">
                Score over time
              </h2>
              <Suspense
                fallback={<div className="mt-3 h-40 animate-pulse bg-paper-sunken" />}
              >
                <ScoreHistory history={score_history} />
              </Suspense>
            </section>
          )}

          <section>
            <h2 className="border-b border-rule pb-2 text-sm font-semibold">
              Recommended next actions
            </h2>
            <ActionList actions={actions} dealId={id} />
          </section>
        </div>

        {/* ---------- controls ---------- */}
        <div className="space-y-8">
          <SimulateEvents dealId={id} updateToken={updateToken} />
          <WhatIfPanel dealId={id} detail={data} />

          <section>
            <h2 className="border-b border-rule pb-2 text-sm font-semibold">
              Activity
            </h2>
            <ol className="mt-1">
              {timeline.map((a) => (
                <li
                  key={a.id}
                  className="flex items-baseline justify-between gap-3 border-b border-rule/60 py-1.5"
                >
                  <span className="text-sm">{activityLabel(a.type)}</span>
                  <span className="shrink-0 text-micro text-ink-faint">
                    {ago(a.occurred_at)}
                  </span>
                </li>
              ))}
            </ol>
            {data.timeline.length > timeline.length && (
              <p className="mt-2 text-micro text-ink-faint">
                {data.timeline.length - timeline.length} earlier events not shown
              </p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
