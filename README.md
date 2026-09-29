# CRM AI Core

Scores open sales deals in real time, explains every score, and turns the
explanation into a next action. Peer-reviewed research (ICIDS 2026), rebuilt as
a public demo that runs entirely on synthetic data.

**[Live demo](https://REPLACE-ME.vercel.app)** · **[Paper](https://REPLACE-ME)** ·
[Model card](artifacts/model_card.md) · [Deploying it yourself](DEPLOY.md)

> Every deal, company and sales rep in the demo is generated. No real CRM data,
> customer names, or anything from the original pilot appears in this
> repository.

<!-- TODO: 60-90s demo recording. Show the pipeline, open a deal, point at the
     contribution ledger, then send an event and let the score move on camera. -->

---

## The two-minute tour

1. **The pipeline.** Sixty open deals with a win probability each, on a shared
   scale you can read down the column, and the highest-priority recommended
   action for each.
2. **Why a deal scores what it does.** Open one. The contribution ledger starts
   at the population base rate, adds each driver, and lands on the prediction.
   The arithmetic closes because the attributions are exact.
3. **What to do about it.** Recommendations name the rule that produced them
   and cite the numbers: *"No contact for 33 days with win probability at 41%
   and a champion already identified — reach out to them directly."*
4. **Watch it change.** Press **Champion identified**. The event goes onto a
   Redis stream, a consumer re-scores the deal, and the new score arrives over
   SSE. The number moves without a refresh.

## How it works

```
Dashboard (Vercel, React + Vite)
      │  REST + Server-Sent Events
      ▼
ML service (Render, Docker) ──── Redis (Upstash): event stream + score cache
      │
      └── PostgreSQL: deals, activities, scores, action ledger
```

A deal's activity history becomes a 32-feature vector. XGBoost scores win
probability; an LSTM served through ONNX Runtime estimates days to close;
TreeSHAP attributes the score to individual features; a rule tree turns those
attributions into next actions. The whole path runs on every ingested event.

### What changed to make it hostable

| Layer | Paper | Here | Why |
|---|---|---|---|
| Ingestion | REST, webhooks, Apache Kafka | REST + signed webhook → Redis Streams | Kafka has no free hosting. Both sit behind one `MessageBus` interface, and the Kafka adapter still runs via `docker compose --profile kafka up`. |
| Features | Streaming pipeline | The same feature code, run by the stream consumer | Unchanged. Training and serving import one module, so they cannot drift. |
| Inference | XGBoost + LSTM, Redis cache | XGBoost native + LSTM via ONNX Runtime | TensorFlow alone is ~600 MB against a 512 MB budget. The LSTM is exported to ONNX and parity-checked. |
| Explainability + agent | SHAP + hierarchical decision tree | XGBoost TreeSHAP + the same rule tree | Same algorithm, computed by the library that built the trees, so contributions sum exactly to the prediction. |
| Action | CRM write-back | Dashboard + action ledger | A public demo should not write to anyone's CRM. |

## Metrics

Two separate sets. They are not comparable and are never averaged.

### Demo models — synthetic data, reproducible from this repository

| Win probability (XGBoost) | | Days to close (LSTM → ONNX) | |
|---|---|---|---|
| Accuracy | 0.738 | MAE | 11.1 days |
| AUC-ROC | 0.788 | RMSE | 14.3 days |
| Precision | 0.709 | Mean-prediction baseline | 15.2 days |
| Recall | 0.539 | Improvement over baseline | +27% |
| Brier | 0.181 | ONNX parity (max abs diff) | 1.2e-07 days |

Held out from 2,000 generated deals and 46,135 activity events.

### Paper — real pilot data, not in this repository

| Metric | Value |
|---|---|
| Accuracy | 87.3% |
| AUC-ROC | 0.92 |
| Reproducible here | **No** |

The pilot data cannot be published, so those figures cannot be regenerated from
this repo. The demo models score lower because the generator is deliberately
noisy, has fewer features, and carries none of the firmographic or relationship
context the real CRM had. A synthetic model matching 0.92 AUC would mean the
generator had leaked the label, not that the model was good.

**Scoring latency is measured, not quoted.** `artifacts/metrics.json` carries
`latency_ms.measured: null` until someone runs
`python training/measure_latency.py --url <service>` against a real deployment.
The paper's figure was measured on different hardware and is not reused.

## API

| Method & path | Purpose |
|---|---|
| `GET /healthz` | Model, database, Redis and bus status |
| `GET /api/deals` | Open pipeline with each deal's latest score (filter, sort, paginate) |
| `GET /api/deals/{id}` | Deal, activity timeline, score history, actions |
| `POST /api/deals/{id}/score` | Re-score now; returns a per-stage latency breakdown |
| `GET /api/deals/{id}/explain` | Full SHAP attribution, additive in log-odds |
| `POST /api/deals/{id}/what-if` | Score a hypothetical change without saving it |
| `POST /api/events` | Simulate a CRM event (the dashboard's buttons) |
| `POST /webhooks/crm` | Signed webhook ingestion (HMAC-SHA256, 5-minute window) |
| `GET /api/actions` | The recommendation queue |
| `PATCH /api/actions/{id}` | Accept or dismiss a recommendation |
| `GET /api/stream` | Server-Sent Events: `score_updated`, `action_created` |
| `GET /api/meta/metrics` | Demo metrics, with the paper's kept separate |
| `POST /api/admin/reset` | Re-seed the demo (admin token) |

Interactive docs at `/docs`.

## Running it locally

```bash
docker compose up          # api + postgres + redis, the hosted topology
```

Then <http://localhost:8000/healthz>. Optional profiles: `--profile proxy`
(nginx), `--profile mlops` (Celery drift/retrain workers), `--profile kafka`
(the paper's ingestion path).

Dashboard:

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

Retraining everything from the seed:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-train.txt
make train            # or .\train.ps1 on Windows
```

`make train` regenerates the synthetic data, both models, the ONNX export with
its parity check, `metrics.json` and the model card — deterministically from
seed 42.

### Tests

```bash
pytest -q                    # 175 backend tests
cd frontend && npx vitest run # 16 unit tests
cd frontend && npm run e2e    # Playwright, needs the service running
```

CI runs the backend suite against both SQLite and PostgreSQL, because the
PostgreSQL run is what catches dialect-specific mistakes.

## Security

- Webhooks are HMAC-SHA256 over `{timestamp}.{body}`, compared in constant time,
  with a five-minute freshness window. Both halves are needed: signing the body
  alone leaves a captured request valid forever.
- The consumer claims each event id in `processed_events` before doing any work,
  so a redelivery after a crash is a no-op rather than a duplicate.
- Rate limits are tiered by cost: reads 120/min, scoring 30/min, what-if 60/min,
  ingestion 20/min. The client key honours `X-Forwarded-For`, or everyone behind
  the proxy would be limited as one visitor.
- CORS origins are explicit. `*` is rejected at startup — it cannot be combined
  with credentialed requests.
- Admin actions need a bearer token of at least 16 characters. Unset means
  disabled, never open.
- The container runs as a non-root user. Secrets come from the environment;
  `.env` is gitignored and `.env.example` documents every variable.

## Limitations

- Trained entirely on generated data. The models have learned the generator's
  structure, not any real market's. Do not use them to forecast real revenue or
  rank real opportunities.
- Probabilities are reported but not explicitly calibrated (Brier 0.181); treat
  them as ordinal.
- A rep unseen at training time falls back to the global base rate.
- The days-to-close model sees activity cadence and deal size only — not
  seasonality, quota pressure, or procurement cycles.
- Free tier: the service sleeps after ~15 minutes idle, so the first request
  wakes it and takes 30–60 seconds. The dashboard says so rather than spinning.

Full detail in the [model card](artifacts/model_card.md).

## Credits

Research and system by **Ashriwad Behera**, with co-author **REPLACE-ME** and
faculty mentor **REPLACE-ME**. Published at ICIDS 2026.

```bibtex
@inproceedings{REPLACE-ME,
  title     = {REPLACE-ME},
  author    = {REPLACE-ME},
  booktitle = {ICIDS},
  year      = {2026}
}
```
