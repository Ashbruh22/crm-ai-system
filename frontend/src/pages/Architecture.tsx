/**
 * How it works, and what the numbers do and do not mean.
 *
 * The metrics comparison is the reason this page exists. The paper's figures were
 * measured on real pilot data that cannot be published; the demo's models are
 * retrained on generated data and score lower. Presenting them side by side with
 * both labelled is the honest treatment, and conflating them would be the easy
 * dishonest one.
 */
import { useMetrics } from "../api/hooks";
import { WakingNotice } from "../components/WakingNotice";

const SWAPS = [
  {
    layer: "Ingestion",
    paper: "REST, webhooks, Apache Kafka",
    demo: "REST + signed webhook onto Redis Streams",
    why: "Kafka has no free hosting. Both sit behind one MessageBus interface, and the Kafka adapter still runs locally.",
  },
  {
    layer: "Feature engineering",
    paper: "Streaming feature pipeline",
    demo: "The same feature code, run by the stream consumer",
    why: "Unchanged. Training and serving import one module, so they cannot drift.",
  },
  {
    layer: "Inference",
    paper: "XGBoost + LSTM, Redis cache",
    demo: "XGBoost native + LSTM via ONNX Runtime",
    why: "TensorFlow alone is ~600 MB against a 512 MB budget. The LSTM is exported to ONNX and checked for parity.",
  },
  {
    layer: "Explainability and actions",
    paper: "SHAP + hierarchical decision tree",
    demo: "XGBoost TreeSHAP + the same rule tree",
    why: "Same algorithm, computed by the library that built the trees, so contributions sum exactly to the prediction.",
  },
  {
    layer: "Presentation",
    paper: "Write back to the CRM",
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

      {/* ---------- metrics ---------- */}
      <section className="mt-8">
        <h2 className="border-b border-ink pb-2 text-sm font-semibold">
          What the models score
        </h2>

        <div className="mt-4 grid gap-6 sm:grid-cols-2">
          <div>
            <h3 className="text-sm font-medium">Demo models</h3>
            <p className="text-micro text-ink-faint">
              trained on synthetic data, reproducible from this repository
            </p>
            <dl className="mt-3 space-y-1.5 text-sm">
              <Metric label="Accuracy" value={fmt(demo.accuracy)} />
              <Metric label="AUC-ROC" value={fmt(demo.auc_roc)} />
              <Metric label="Precision" value={fmt(demo.precision)} />
              <Metric label="Recall" value={fmt(demo.recall)} />
              <Metric
                label="Days-to-close error"
                value={cycle.mae_days ? `${cycle.mae_days.toFixed(1)} days` : "—"}
              />
            </dl>
          </div>

          <div>
            <h3 className="text-sm font-medium">Published paper</h3>
            <p className="text-micro text-ink-faint">
              measured on the real pilot CRM data, which is not in this repository
            </p>
            <dl className="mt-3 space-y-1.5 text-sm">
              <Metric
                label="Accuracy"
                value={fmt(data?.paper.win_model_accuracy)}
              />
              <Metric label="AUC-ROC" value={fmt(data?.paper.win_model_auc_roc)} />
              <Metric label="Reproducible here" value="No" />
            </dl>
          </div>
        </div>

        <p className="mt-5 max-w-prose border-l-2 border-rule-strong pl-3 text-sm leading-relaxed text-ink-muted">
          These two sets are not comparable and are never averaged. The demo
          models see a deliberately noisy generator with fewer features and none
          of the firmographic context the real CRM carried. A synthetic model
          matching the paper's 0.92 AUC would mean the generator had leaked the
          label, not that the model was good.
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
                  <span className="text-micro text-ink-faint">paper</span>{" "}
                  {row.paper}
                </span>
                <span className="text-ink-muted">
                  <span className="text-micro text-ink-faint">here</span>{" "}
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
