# GAP_ANALYSIS.md — existing repo vs. `CRM_AI_CORE_DEMO_SPEC.md`

**Repo:** `crm-ai-system` (`github.com/Ashbruh22/crm-ai-system`)
**Branch:** `demo/hosted` (cut from `main` @ `b29a079`)
**Date:** 2026-09-28
**Scope:** read-only analysis. No code written.

---

## 0. Headline findings

The repo is **substantially further along than the spec assumes**. A FastAPI service, Alembic
migrations, a React/Vite dashboard, Docker Compose, CI, SHAP, a recommendation engine and a
closed-loop MLOps stack (drift → Optuna retrain → A/B promote) already exist across 7 commits.

Three corrections to the spec's assumptions in §A.4:

1. **There is no Kafka code.** "kafka" appears only in a docstring and a JSON disclosure string
   (`ml/train_problem2_lstm.py:487`, `models/problem2_results.json:18`), both describing Kafka as
   *future* work from the PRD. The async path that exists is **Celery + Redis broker**. So §5's
   "wrap existing Kafka as `KafkaBus`" has nothing to wrap — the `MessageBus` work is BUILD NEW,
   and `KafkaBus` is optional (build it only as an interview talking point, or drop it and talk
   about Celery→Streams instead, which is the honest story).
2. **The database is PostgreSQL, not MySQL**, and `app/models/db.py:4` imports
   `sqlalchemy.dialects.postgresql.UUID` on every table's PK. Moving to MySQL 8 is a real
   migration (UUID → `CHAR(36)`/`BINARY(16)`, plus `asyncpg` → `aiomysql`/PyMySQL and rewriting
   4 Alembic revisions). The spec explicitly says PostgreSQL works unchanged — **recommendation:
   stay on PostgreSQL** unless the MySQL line on the résumé is worth ~a day of migration.
3. **The service cannot boot from a fresh clone.** `app/main.py` loads `ml/xgboost_model.joblib`
   and `ml/lstm_problem2_live_reforecast.keras` at startup; both are **gitignored** (`*.joblib`,
   `*.keras`) and absent from history. Only `ml/pipeline.joblib` is tracked, and it is *stale* —
   the working tree has it modified from 22,534 → 5,251 bytes. This is the single biggest
   hosting blocker and the reason Phase 1 cannot be skipped.

---

## 1. Safety check (§A.1)

| Check | Result |
|---|---|
| Data-like files in history | 3 CSVs + 1 `.joblib` + 6 JSONs, all added in `b26266b`/`c134403`/`c2508f2` |
| Real company / client names | **None found.** Accounts are `Company_0…999`; reps are 8 invented names (`Sarah Chen`, `Mike Torres`, `Priya Nair`, …) hard-coded in `ml/generate_data.py:12` |
| Pilot exports | **None found.** `data/sales_pipeline.csv` (8,800 rows) and `data/processed_dataset.csv` are both reproducible from `ml/generate_data.py`, which is seeded (`np.random.seed(42)`, `random.seed(42)`) |
| Notebooks with printed output | **None.** Repo contains zero `.ipynb` files |
| Secrets | **None leaked.** Only dev-only in-memory logins in `app/routers/auth.py:11-15` (`test_pass`, `demo_pass`, `admin_pass`) and placeholders in `.env.example`. `gitleaks` is **not installed** on this machine — run it before publishing (see Action S1) |

**Verdict: history appears clean. No `git filter-repo` purge needed, no keys to rotate.**

### Items still needing a decision

- **`data/crm_training_dataset.csv` (200 rows) — provenance unverified.** Unlike the other CSVs
  it is *not* produced by `generate_data.py`: different schema (`deal_id` `#7243`, `account`
  `Nexus Corp` / `Apex Systems`, `exec_sponsor`, `competition`, `risk_level`) and a different
  ID scheme. Names read as invented, but **confirm it wasn't hand-built from pilot deals before
  linking the repo anywhere.** If provenance is uncertain, the spec's own rule applies: leave it
  out. → Action **S2**.
- **`.gitignore` does not match reality.** It ignores `data/*.csv` and `*.joblib`, yet 3 CSVs and
  `ml/pipeline.joblib` are tracked (committed before the rules landed). `.gitignore` does not
  stop already-tracked files. → Action **S3**.

### Actions

- **S1** — install and run `gitleaks detect --source . --log-opts="--all"` before the repo is linked publicly.
- **S2** — confirm provenance of `data/crm_training_dataset.csv`; if pilot-derived, `git filter-repo` it out and force-push.
- **S3** — `git rm --cached` the tracked CSVs/joblib, or (better, per Phase 1) replace them with `make train` regeneration from a seed.

---

## 2. Spec-section mapping

| Spec § | Existing files | Status | Notes |
|---|---|---|---|
| **§4** Synthetic generator | `ml/generate_data.py`, `ml/event_simulator.py`, `scripts/crm_simulator.py` | **ADAPT** (spec said BUILD NEW) | Already seeded and reproducible. Missing: the **~60 open-deal live set** for the demo, Faker-based company names (currently `Company_N`), and the §4 activity-event vocabulary (`champion_identified`, `discount_requested`, `no_activity_7d`, …). Current signal is agent/product-driven, not activity-driven — the §4 "realistic signal" list needs building in or SHAP explanations stay shallow |
| **§5** XGBoost training | `ml/train.py`, `ml/pipeline.py` | **REUSE / ADAPT** | Artifacts are untracked; retrain + commit under `artifacts/`. Save native `xgb_model.json` instead of `.joblib` so `TreeExplainer` loads it directly and the image drops scikit-learn/joblib pickle-version coupling (`requirements.txt` has a comment pinning `scikit-learn==1.9.0` *pre-release* to match the pickle — a hosting hazard) |
| **§5** LSTM + **ONNX export** | `ml/train_problem2_lstm.py`, `ml/lstm_problem2_live_reforecast.keras` (untracked) | **ADAPT — critical** | **No ONNX anywhere in the repo.** `app/main.py:3` does `import tensorflow as tf` at module scope; `requirements.txt` pins `tensorflow==2.21.0`. This alone blows the ~512 MB budget. Must export `(None, 60, 7)` → `lstm.onnx` + parity test, then remove TF from the runtime image |
| **§5** `feature_schema.json` | — (`data/training_feature_stats.json` is adjacent but not this) | **BUILD NEW** | No ordered-feature contract and no startup validation. `data/processed_dataset.csv` header is the de-facto order — freeze it |
| **§5** `metrics.json` + model card | `models/phase2_results.json`, `models/problem2_results.json`, `models/baseline_comparison.csv`, `models/training_logs.txt` | **ADAPT** | Metrics exist in bespoke shapes; no `evaluate.py`, no `metrics.json`, **no `model_card.md`**. `problem2_results.json:18` already contains an honest synthetic-data disclosure — good raw material for the card |
| **§6** DB schema + Alembic | `app/models/db.py`, `alembic/versions/001…004` | **ADAPT** | Tables `opportunities`, `predictions`, `recommendations`, `model_versions`, `performance_log`, drift results. Maps onto spec's `deals`/`scores`/`actions`. **Missing: `activities` (no event log at all) and `processed_events` (no idempotency).** Two real defects: all timestamps are `Column(String)` not `DateTime` (breaks score-over-time ordering and indexing), and `recommendations` has no `status ENUM(suggested, accepted, dismissed)` — only `adopted_at` |
| **§6** seed command | — | **BUILD NEW** | No seeding path; the demo has nothing to show on first boot |
| **§7** FastAPI service | `app/main.py`, 6 routers, 1,033 lines | **REUSE / ADAPT** | Solid base: JWT auth, request-ID middleware, Prometheus, `/health` + `/ready` (satisfies `/healthz`) |
| §7 `GET /api/deals`, `/deals/{id}` | — | **BUILD NEW** | No pipeline-list endpoint. Closest is `POST /api/v1/predict/outcome` (single) and `/predict/history/{opp_id}` |
| §7 `POST /score` | `app/routers/predict.py:18` | **REUSE** | Needs a cache-bypass flag |
| §7 `POST /what-if` | — | **BUILD NEW** | No trace of what-if anywhere |
| §7 `GET /explain` | `app/routers/explain.py` (`/global`, `/local/{opp_id}`) | **REUSE** | Already both global and local |
| §7 `POST /api/events` | `app/routers/crm.py:110` `/webhook/simulate` | **ADAPT** | Exists but writes synchronously; needs to publish to the bus |
| §7 `POST /webhooks/crm` | `app/routers/crm.py:57,83` + `app/utils/hmac_validator.py` | **ADAPT** | HMAC-SHA256 with `hmac.compare_digest` ✅. **Missing the 5-minute timestamp tolerance** — `validate_hubspot_signature` receives `timestamp` and folds it into the digest but never checks freshness, so **replays are accepted**. Salesforce variant takes no timestamp at all |
| §7 `PATCH /api/actions/{id}` | `app/routers/recommendations.py:61` `POST /feedback` | **ADAPT** | Feedback path exists; needs accept/dismiss status semantics |
| §7 `GET /api/stream` (SSE) | — | **BUILD NEW** | **No SSE/`StreamingResponse`/`EventSource` anywhere.** This is the "watch it update live" moment in the 2-minute tour — the highest-value missing piece after the artifacts |
| §7 `GET /api/meta/metrics` | — | **BUILD NEW** | Depends on `metrics.json` |
| §7 `POST /api/admin/reset` | `app/routers/admin.py` (7 endpoints, none a reset) | **BUILD NEW** | No reset, no nightly job |
| §7 Redis cache + latency | `app/services/prediction_service.py:29-39,130-146` | **ADAPT — has a bug** | Cache + `inference_latency_ms` exist. But `feature_hash = hash(feature_string)` (line 32) uses Python's **per-process-salted** `hash()`, so **every restart and every worker computes different keys** — the cache silently never hits across processes. Must switch to `hashlib`. Also latency is a single total, not the §7 features/xgb/lstm/shap breakdown |
| §7 consumer | `ml/celery_app.py`, `ml/tasks/*` | **ADAPT** | Celery worker + beat exist (drift/retrain/evaluate/promote). The scoring consumer is new work; Celery is a heavier dependency than one Render service wants |
| **§8** NBA rule engine | `app/services/recommendation_service.py` (59 lines) | **ADAPT — biggest rewrite** | Currently **4 mutually exclusive `win_prob` bands**, one canned action each, `rationale` = the model's `nl_explanation`. Per §8 it needs: rule **ids**, conditions over **features and SHAP** (not just probability), multiple rules firing, and reasons that **cite drivers** ("fell 18 pts after 9 days of no activity →"). **Zero unit tests.** The band comment claims contiguity — it is in fact contiguous, so that part is fine |
| **§9** Dashboard | `frontend/` — React + **Vite** + TS + Tailwind, 6 pages, 7 components incl. `SHAPWaterfallChart.tsx` | **ADAPT (not BUILD NEW)** | Spec says Next.js App Router; repo has a Vite SPA with `vercel.json` already present. **Recommendation: keep Vite.** The spec's Next.js-specific asks (server components, lazy routes) are achievable in Vite via `React.lazy`, and a rewrite buys nothing a reviewer can see. Existing `Dashboard`, `DealDetail`, `GlobalSHAP`, `Recommendations`, `AdminRetrain`, `Login` cover most of §9 |
| §9 Pipeline page | `frontend/src/pages/Dashboard.tsx` | **ADAPT** | Needs the synthetic-data banner and top-action column |
| §9 SHAP chart | `frontend/src/components/SHAPWaterfallChart.tsx` | **REUSE** | |
| §9 Simulate buttons / what-if sliders | — | **BUILD NEW** | |
| §9 `/architecture` page | `docs/crm_integration.md`, `crm_ai_system_architecture.svg` (in `~/Downloads`, **not in repo**) | **BUILD NEW** | Copy the SVG in; add the paper-vs-demo table and the two labelled metric sets |
| §9 Cold-start UX | `frontend/src/api/mock.ts` | **BUILD NEW** | Mock layer exists but no "waking the ML service" state |
| **§10** CORS | `app/main.py:57` `allow_origins=["*"]` **with** `allow_credentials=True` | **ADAPT — fix** | Wildcard + credentials is both a spec violation and rejected by browsers. Must come from `ALLOWED_ORIGINS` |
| §10 Rate limiting | `nginx/nginx.conf` only | **ADAPT** | Nginx-level limiting won't exist on Render. Add `slowapi` in-app |
| §10 Non-root container | `Dockerfile` | verify | Not yet inspected line-by-line — check for a `USER` directive in Phase 8 |
| **§11** Tests | `tests/test_api_smoke.py` — **4 tests, 35 lines** | **BUILD NEW** | No feature-builder, NBA, HMAC, ONNX-parity, integration, or Playwright tests. `vitest` not in `frontend/package.json` scripts (uses `oxlint`) |
| **§11** CI | `.github/workflows/ci.yml` | **ADAPT** | Pytest + GHCR image build. No lint, no type-check, no DB service container |
| **§12** Deploy | `Dockerfile`, `docker-compose.yml`, `nginx/nginx.conf`, `frontend/vercel.json` | **ADAPT** | Compose has `api`, `db` (postgres:14-alpine), `redis`, `nginx`, `celery-worker`, `celery-beat` — **6 services, no `kafka` profile**. Render needs a slimmer single-service target. `frontend/vercel.json` already present |
| **§13** README | `README.md` | **ADAPT** | Exists; needs live link, demo GIF, paper-vs-demo table, the two labelled metric sets, credits |

---

## 3. Dependencies that won't fit a ~512 MB CPU runtime

From `requirements.txt` — these are all in the **runtime** install today:

| Package | Problem |
|---|---|
| `tensorflow==2.21.0` | ~600 MB installed, alone over budget. **Must leave the runtime image** (spec §0). Replace with `onnxruntime` (~15 MB) |
| `mlflow` (unpinned) | Heavy, pulls a large transitive tree; also the only unpinned line in the file |
| `optuna==4.4.0` | Training-time only |
| `celery==5.4.0` | Only needed if the Celery MLOps loop is hosted; not needed to serve scores |
| `scikit-learn==1.9.0` | A **pre-release pin**, kept only to match the `pipeline.joblib` pickle. Fragile. Re-export the preprocessing as plain code or a versioned artifact and drop the hard pin |
| `psycopg2-binary` **and** `asyncpg` | Both drivers installed; keep one |
| `prometheus-fastapi-instrumentator`, `structlog`, `gunicorn` | Keep — small and genuinely useful |

**Split into `requirements.txt` (runtime) and `requirements-train.txt` (dev/training)** — that one
change plus the ONNX swap is most of the budget fight.

---

## 4. Where `MessageBus` goes

There is no Kafka to adapt, so this is a clean build:

```
app/bus/base.py            # MessageBus: publish(topic, event) / consume(group, handler)
app/bus/redis_streams.py   # hosted demo path (Upstash) — XADD / XREADGROUP / XACK
app/bus/kafka.py           # OPTIONAL, local only, docker compose --profile kafka
app/consumer.py            # event → activity → features → score → SHAP → NBA → persist → SSE
```

Current ingestion is `app/routers/crm.py` → `app/services/crm_mapper.py` → synchronous predict.
The seam is at the end of each webhook handler: today it maps and scores inline; it should
instead `bus.publish(...)` and return `202`, with the consumer doing the rest. `crm_mapper.py`
(72 lines) is reusable as-is inside the consumer.

Celery stays for the **MLOps** loop (drift/retrain/promote) or gets dropped for the demo — it is
not the right tool for per-event scoring and it costs a second Render service.

---

## 5. Revised phase plan

| # | Phase | Status | Revised scope |
|---|---|---|---|
| 0 | Safety, branch, gap analysis | **DONE** (pending S1/S2) | History clean; `demo/hosted` cut; this file. Still: run `gitleaks`, confirm `crm_training_dataset.csv` provenance |
| 1 | Generator + training + **ONNX** + model card | **ADAPT — do not skip** | Biggest single win. Add §4 activity events + 60 open deals + Faker names; retrain; **export `lstm.onnx` + parity test**; `xgb_model.json`; `feature_schema.json`; `evaluate.py` → `metrics.json` + `model_card.md`; commit all under `artifacts/`; `make train` |
| 2 | FastAPI + config + schema + Alembic + seed + compose | **ADAPT (~60% done)** | Add `activities` + `processed_events`; `String` → `DateTime` timestamps; `status` enum on actions; seed command. **Decide Postgres vs MySQL first** (recommend Postgres) |
| 3 | Inference + SHAP + cache + latency | **ADAPT (~70% done)** | Swap TF→ONNX at `main.py`; **fix the `hash()` cache-key bug**; split latency into features/xgb/lstm/shap/total; add `/what-if`; `feature_schema` startup check |
| 4 | NBA + actions ledger | **ADAPT — near-rewrite** | Rule ids, SHAP/feature conditions, driver-citing reasons, multi-rule firing, unit tests |
| 5 | `MessageBus` + consumer + webhook + idempotency + **SSE** | **BUILD NEW** | No Kafka to wrap. Add **replay protection** to HMAC (5-min tolerance) — currently missing. SSE is entirely new |
| 6 | Dashboard: pipeline, detail, SHAP, simulate, what-if | **ADAPT (~50% done)** | Keep **Vite**, don't port to Next.js. Add simulate buttons, what-if sliders, SSE subscription, synthetic banner |
| 7 | Architecture page, metrics comparison, cold start | **BUILD NEW** | Bring the architecture SVG into the repo |
| 8 | Security pass | **ADAPT** | Fix `allow_origins=["*"]` + `allow_credentials=True`; add `slowapi`; admin reset + nightly job; verify non-root `USER` in Dockerfile; retire the hard-coded demo logins or scope them clearly |
| 9 | Deploy: Render + Vercel + hosted DB + Upstash | **BUILD NEW** | Slim single-service Render target; **measure real latency**, don't reuse 180 ms |
| 10 | README, video, merge, résumé links | **ADAPT** | |

**Critical path:** Phase 1 (ONNX + committed artifacts) → Phase 3 (serve them) → Phase 5 (SSE).
Nothing is hostable until the artifacts are in the repo and TensorFlow is out of the image.

---

## 6. Housekeeping found en route

- **Uncommitted work carried from `main`:** `app/routers/auth.py`, `app/services/shap_service.py`,
  `ml/pipeline.py` (±182 lines), and a regenerated `data/processed_dataset.csv` +
  `ml/pipeline.joblib` (22,534 → 5,251 bytes). **Review and commit or discard deliberately** —
  the shrunken `pipeline.joblib` may be why the tracked pipeline is stale.
- **6 stray untracked root files:** `app.py` (a 146-line **Streamlit** app — a second, undocumented
  UI), `pipeline.py` (a 6-line "stub module for joblib unpickling" — a smell confirming the pickle
  fragility above), `test_phase3.py`, `test_probe.py`, `test_probe_static.py`, `verify_ml.py`
  (probe scripts, two with BOM markers). Decide: move the probes to `training/` or delete;
  `app.py` is either a real alternate demo worth keeping or dead weight to remove.
- **`ml/lstm_model.h5` (5.8 MB)** on disk and unreferenced by `main.py` — superseded by the
  `.keras` file. Drop it.
- **Paper/spec assets live in `~/Downloads`, not the repo:** `crm_ai_system_architecture.svg`,
  `CRM_AI_CORE_DEMO_SPEC.md`, the IEEE paper drafts. Bring the SVG and the spec in (Phase 7/10).

---

## 7. Two questions that change the work

1. **Postgres or MySQL?** Recommendation: **keep PostgreSQL.** The spec permits it, and the
   Postgres-dialect UUID PKs + 4 Alembic revisions make the swap ~a day of work that no reviewer
   will see. Take MySQL only if a specific job posting makes it a must-have.
2. **Next.js or keep Vite?** Recommendation: **keep Vite.** Six working pages and a SHAP waterfall
   already exist, `vercel.json` is in place, and §9's real asks (live SSE update, what-if sliders,
   lazy charts) are all reachable in Vite. A rewrite is invisible to a reviewer.

Both are logged as assumptions, not decisions — say the word and either flips.
