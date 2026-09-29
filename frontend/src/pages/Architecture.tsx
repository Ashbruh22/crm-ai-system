/**
 * How it works, and what the numbers do and do not mean.
 *
 * Every figure on this page was measured on synthetic data and says so. The
 * design decisions listed below are the interesting part: each one names what
 * the original system did, what this demo does instead, and why the swap was
 * necessary to host it on a free tier.
 */
import { useMetrics } from "../api/hooks";
import { ArchitectureDiagram } from "../components/ArchitectureDiagram";
import { WakingNotice } from "../components/WakingNotice";

const SWAPS = [
  {
    layer: "Ingestion",
    original: "REST, webhooks, Apache Kafka",
    demo: "REST + signed webhook onto Redis Streams",
    why: "Kafka has no free hosting. Both sit behind one MessageBus interface, and the Kafka adapter still runs locally.",
  },
  {
    layer: "Feature engineering",
    original: "Streaming feature pipeline",
    demo: "The same feature code, run by the stream consumer",
    why: "Unchanged. Training and serving import one module, so they cannot drift.",
  },
  {
    layer: "Inference",
    original: "XGBoost + LSTM, Redis cache",
    demo: "XGBoost native + LSTM via ONNX Runtime",
    why: "TensorFlow alone is ~600 MB against a 512 MB budget. The LSTM is exported to ONNX and checked for parity.",
  },
  {
    layer: "Explainability and actions",
    original: "SHAP + hierarchical decision tree",
    demo: "XGBoost TreeSHAP + the same rule tree",
    why: "Same algorithm, computed by the library that built the trees, so contributions sum exactly to the prediction.",
  },
  {
    layer: "Presentation",
    original: "Write back to the CRM",
    demo: "This dashboard plus an action ledger",
    why: "A public demo should not write to anyone's CRM.",
  },
];

export function Architecture() {
  const { data, isLoading } = useMetrics();

  if (isLoading) return <WakingNotice label="Loading the metrics" />;

  const demo = data?.demo.win_probability ?? {};
  const cycle = (data?.demo.days_to_close ?? {}) as Record<string, number>;

  return (
    <div className="max-w-3xl">
      <h1 className="text-lg font-semibold tracking-tight">How it works</h1>
      <p className="mt-2 max-w-prose text-sm leading-relaxed text-ink-muted">
        A deal's activity history becomes a feature vector, two models score it,
        TreeSHAP attributes the score to individual features, and a rule tree
        turns those attributions into next actions. Every deal, company and rep
        in this demo is generated.
      </p>

      <ArchitectureDiagram />

      {/* ---------- metrics ---------- */}
      <section className="mt-8">
        <h2 className="border-b border-ink pb-2 text-sm font-semibold">
          What the models score
        </h2>

        <div className="mt-4 grid gap-6 sm:grid-cols-2">
          <div>
            <h3 className="text-sm font-medium">Win probability</h3>
            <p className="text-micro text-ink-faint">XGBoost classifier</p>
            <dl className="mt-3 space-y-1.5 text-sm">
              <Metric label="Accuracy" value={fmt(demo.accuracy)} />
              <Metric label="AUC-ROC" value={fmt(demo.auc_roc)} />
              <Metric label="Precision" value={fmt(demo.precision)} />
              <Metric label="Recall" value={fmt(demo.recall)} />
              <Metric label="Brier score" value={fmt(demo.brier)} />
            </dl>
          </div>

          <div>
            <h3 className="text-sm font-medium">Days to close</h3>
            <p className="text-micro text-ink-faint">LSTM, served as ONNX</p>
            <dl className="mt-3 space-y-1.5 text-sm">
              <Metric
                label="Mean error"
                value={cycle.mae_days ? `${cycle.mae_days.toFixed(1)} days` : "—"}
              />
              <Metric
                label="Baseline error"
                value={
                  cycle.baseline_mae_days
                    ? `${cycle.baseline_mae_days.toFixed(1)} days`
                    : "—"
                }
              />
              <Metric
                label="Better than baseline"
                value={
                  cycle.improvement_vs_baseline
                    ? `${(cycle.improvement_vs_baseline * 100).toFixed(0)}%`
                    : "—"
                }
              />
            </dl>
          </div>
        </div>

        <p className="mt-5 max-w-prose border-l-2 border-rule-strong pl-3 text-sm leading-relaxed text-ink-muted">
          Held out from {"2,000"} generated deals. The generator carries
          deliberate noise, so these are the numbers a model can reach on data
          with real uncertainty in it — a near-perfect score here would mean the
          generator had leaked the answer, not that the model was good.
        </p>
      </section>

      {/* ---------- swaps ---------- */}
      <section className="mt-10">
        <h2 className="border-b border-ink pb-2 text-sm font-semibold">
          What changed to make it hostable
        </h2>
        <dl className="mt-1">
          {SWAPS.map((row) => (
            <div key={row.layer} className="border-b border-rule/60 py-3">
              <dt className="text-sm font-medium">{row.layer}</dt>
              <dd className="mt-1 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
                <span className="text-ink-muted">
                  <span className="text-micro text-ink-faint">was</span>{" "}
                  {row.original}
                </span>
                <span className="text-ink-muted">
                  <span className="text-micro text-ink-faint">now</span>{" "}
                  {row.demo}
                </span>
                <span className="max-w-prose text-ink-faint sm:col-span-2">
                  {row.why}
                </span>
              </dd>
            </div>
          ))}
        </dl>
      </section>

      {/* ---------- drivers ---------- */}
      {data?.global_drivers?.length ? (
        <section className="mt-10">
          <h2 className="border-b border-ink pb-2 text-sm font-semibold">
            What the win model relies on most
          </h2>
          <p className="mt-2 text-micro text-ink-faint">
            mean absolute SHAP contribution across the held-out set
          </p>
          <ul className="mt-3">
            {data.global_drivers.slice(0, 8).map((d) => {
              const top = data.global_drivers[0].mean_abs_shap || 1;
              return (
                <li key={d.feature} className="flex items-center gap-3 py-1">
                  <span className="w-48 shrink-0 text-sm">{d.label}</span>
                  <span className="h-2 flex-1 bg-paper-sunken">
                    <span
                      className="block h-2 bg-lift"
                      style={{ width: `${(d.mean_abs_shap / top) * 100}%` }}
                    />
                  </span>
                  <span className="tnum w-12 text-right text-micro text-ink-muted">
                    {d.mean_abs_shap.toFixed(3)}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      {data?.latency_ms && (
        <p className="mt-10 max-w-prose text-sm text-ink-muted">
          {data.latency_ms.measured
            ? `Single-deal scoring takes about ${data.latency_ms.measured} ms on the hosted service.`
            : "Scoring latency on the hosted service has not been measured yet, so no figure is quoted here."}
        </p>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-rule/60 pb-1">
      <dt className="text-ink-muted">{label}</dt>
      <dd className="tnum">{value}</dd>
    </div>
  );
}

function fmt(value: number | undefined): string {
  return value === undefined || value === null ? "—" : value.toFixed(3);
}
