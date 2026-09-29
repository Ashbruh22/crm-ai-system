# Deploying the demo

Four pieces: a Postgres database, a Redis instance, the ML service on Render,
and the dashboard on Vercel. All free tier.

The order matters in one place only — the service and the dashboard each need
the other's URL, so the last two steps are a round trip.

---

## 1. Redis (Upstash)

The event bus uses Redis Streams, which need **Redis 5 or newer**. Upstash is
current; a self-hosted Redis 3 or 4 will pass a health check, serve the score
cache happily, and then fail every `XGROUP` call — `/healthz` reports this
explicitly rather than leaving you to find it in the logs.

1. Create a database at <https://console.upstash.com> (choose a region near
   your Render region).
2. Copy the **`rediss://`** connection string. Keep it for step 3.

## 2. Database and service (Render)

1. Render dashboard → **New → Blueprint**, point it at this repository.
   `render.yaml` declares the web service and a free Postgres instance, and
   generates `SECRET_KEY`, `ADMIN_TOKEN` and `CRM_WEBHOOK_SECRET`.
2. Render will ask for the two values marked `sync: false`:
   - `REDIS_URL` — the Upstash string from step 1
   - `ALLOWED_ORIGINS` — leave a placeholder for now; step 4 corrects it
3. Deploy. The first build takes a few minutes (the image is ~950 MB).
4. Check it came up:

   ```
   curl https://<your-service>.onrender.com/healthz
   ```

   Expect `"status": "ok"` with `db`, `redis` and `bus.available` all good. If
   `bus.available` is `false`, the message names the reason.

## 3. Dashboard (Vercel)

1. Vercel → **Add New → Project**, import the repository.
2. Set **Root Directory** to `frontend`.
3. Add an environment variable:

   ```
   VITE_API_URL = https://<your-service>.onrender.com
   ```

   No trailing slash, no `/api` suffix — the client appends paths itself. Vite
   inlines this at build time, so it must be set *before* the build. A
   production build without it throws rather than shipping a guessed URL.
4. Deploy.

## 4. Close the loop

Back on Render, set `ALLOWED_ORIGINS` to the Vercel URL and redeploy:

```
ALLOWED_ORIGINS=https://<your-project>.vercel.app
```

Add the preview domain too if you want preview deployments to work. `*` is
rejected at startup: it cannot be combined with credentialed requests.

## 5. Scheduled jobs (GitHub)

Two workflows need repository secrets (**Settings → Secrets and variables →
Actions**):

| Secret | Value |
|---|---|
| `DEMO_API_URL` | `https://<your-service>.onrender.com` |
| `DEMO_ADMIN_TOKEN` | the `ADMIN_TOKEN` Render generated (Environment tab) |

- `nightly-reset.yml` — puts the pipeline back to a clean 60 deals each night
  and re-anchors their timeline, so the board does not drift into looking
  abandoned.
- `keep-warm.yml` — pings `/healthz` every 10 minutes during waking hours, so a
  reviewer does not land on a 50-second cold start.

## 6. Measure the latency

`artifacts/metrics.json` ships with `latency_ms.measured: null` on purpose: no
figure is quoted until one has actually been observed. Fill it in from the real
deployment:

```
python training/measure_latency.py --url https://<your-service>.onrender.com
```

It scores a sample of deals, reports the per-stage breakdown, and writes the
median into `metrics.json`. Commit the result, and the architecture page will
quote it.

---

## Notes

**Cold starts.** Render's free tier sleeps after ~15 minutes idle; the first
request then takes 30–60 seconds. The dashboard shows a "waking the model
service" state rather than a spinner, and the keep-warm workflow reduces how
often anyone sees it.

**Free Postgres expires.** Render's free database is deleted after 90 days.
Nothing in it is precious — it holds generated data and is re-seeded on boot —
but the service will fail its health check until `DATABASE_URL` points
somewhere. Create a fresh one and update the variable.

**One worker is deliberate.** `WEB_CONCURRENCY=1`. Each worker loads its own
copy of the models and starts its own stream consumer, so raising it multiplies
memory use and duplicates scoring work.
