/**
 * The five layers, drawn as the path one event actually takes.
 *
 * Deliberately not a box-and-arrow decoration: each band names the component
 * that runs, and the right-hand column names what the original used where the demo
 * differs. The vertical flow is what makes it work on a phone — five layers
 * side by side would be illegible below ~700px.
 *
 * Inline SVG with a viewBox rather than an image: it scales without a second
 * request, and the text stays selectable and readable by a screen reader.
 */

interface Layer {
  n: number;
  name: string;
  here: string;
  original?: string;
}

const LAYERS: Layer[] = [
  {
    n: 1,
    name: "Ingestion",
    here: "REST + signed webhook → Redis Streams",
    original: "Kafka",
  },
  {
    n: 2,
    name: "Feature engineering",
    here: "app/features/build.py — 32 features",
  },
  {
    n: 3,
    name: "Inference",
    here: "XGBoost + LSTM (ONNX), Redis cache",
    original: "TensorFlow",
  },
  {
    n: 4,
    name: "Explainability + agent",
    here: "TreeSHAP → 12 next-best-action rules",
  },
  {
    n: 5,
    name: "Action",
    here: "Dashboard + action ledger",
    original: "CRM write-back",
  },
];

const BAND_H = 72;
const GAP = 14;
const TOP = 46;
const LEFT = 92;
const WIDTH = 640;
const BAND_W = 384;

export function ArchitectureDiagram() {
  const height = TOP + LAYERS.length * (BAND_H + GAP) + 44;

  return (
    <figure className="mt-4">
      <svg
        viewBox={`0 0 ${WIDTH} ${height}`}
        className="h-auto w-full max-w-2xl"
        role="img"
        aria-labelledby="arch-title arch-desc"
      >
        <title id="arch-title">
          How a CRM event becomes a score and a recommendation
        </title>
        <desc id="arch-desc">
          Five layers in sequence: ingestion over Redis Streams, feature
          engineering, inference with XGBoost and an ONNX LSTM, explainability
          with TreeSHAP feeding a rule engine, and finally the dashboard and
          action ledger.
        </desc>

        <defs>
          <marker
            id="arrow"
            viewBox="0 0 8 8"
            refX="6"
            refY="4"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 z" fill="#5a6a7a" />
          </marker>
        </defs>

        {/* The event's path down the left, one continuous line. */}
        <line
          x1={LEFT - 30}
          y1={TOP - 14}
          x2={LEFT - 30}
          y2={TOP + LAYERS.length * (BAND_H + GAP) - GAP + 6}
          stroke="#b3bec9"
          strokeWidth="1.5"
          markerEnd="url(#arrow)"
        />

        <text x={LEFT - 30} y={TOP - 24} textAnchor="middle" className="fill-ink-faint" fontSize="11">
          one event
        </text>

        {LAYERS.map((layer, i) => {
          const y = TOP + i * (BAND_H + GAP);
          return (
            <g key={layer.n}>
              {/* Node on the path */}
              <circle cx={LEFT - 30} cy={y + BAND_H / 2} r="4" fill="#16202b" />

              <rect
                x={LEFT}
                y={y}
                width={BAND_W}
                height={BAND_H}
                fill="#f7f9fb"
                stroke="#d3dae2"
                strokeWidth="1"
              />

              <text
                x={LEFT + 14}
                y={y + 20}
                fontSize="11"
                className="fill-ink-faint"
                fontFamily="'IBM Plex Mono', monospace"
              >
                layer {layer.n}
              </text>
              <text
                x={LEFT + 14}
                y={y + 40}
                fontSize="14"
                fontWeight="600"
                className="fill-ink"
              >
                {layer.name}
              </text>
              <text x={LEFT + 14} y={y + 60} fontSize="12" className="fill-ink-muted">
                {layer.here}
              </text>

              {/* What the original design used, where it differs. */}
              {layer.original && (
                <>
                  <line
                    x1={LEFT + BAND_W}
                    y1={y + BAND_H / 2}
                    x2={LEFT + BAND_W + 20}
                    y2={y + BAND_H / 2}
                    stroke="#d3dae2"
                    strokeWidth="1"
                    strokeDasharray="3 3"
                  />
                  <text
                    x={LEFT + BAND_W + 26}
                    y={y + BAND_H / 2 - 2}
                    fontSize="11"
                    className="fill-ink-faint"
                  >
                    was
                  </text>
                  <text
                    x={LEFT + BAND_W + 26}
                    y={y + BAND_H / 2 + 12}
                    fontSize="12"
                    className="fill-ink-muted"
                  >
                    {layer.original}
                  </text>
                </>
              )}
            </g>
          );
        })}

        <text
          x={LEFT - 30}
          y={TOP + LAYERS.length * (BAND_H + GAP) + 18}
          textAnchor="middle"
          fontSize="11"
          className="fill-ink-faint"
        >
          score
        </text>
      </svg>

      <figcaption className="mt-2 max-w-prose text-micro text-ink-faint">
        Sending an event from a deal page runs this whole path. The new score
        comes back over a live connection, which is why the number moves without
        a refresh.
      </figcaption>
    </figure>
  );
}
