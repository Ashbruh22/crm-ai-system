/**
 * HTTP client for the ML service.
 *
 * The previous version installed axios-mock-adapter whenever
 * `import.meta.env.PROD` was true, so the deployed build served invented
 * numbers instead of talking to the model. For a live demo whose entire point
 * is that a reviewer can poke the real thing, that is worse than being down:
 * the page looks fine and every figure on it is fiction. It is gone.
 *
 * It also no longer attaches a bearer token. Reads are public — the data is
 * synthetic, and making a reviewer log in before they can see a pipeline costs
 * more than it protects.
 */
import axios from "axios";

/**
 * Base URL of the ML service, with any trailing slash trimmed so
 * `${BASE}/api/deals` never doubles up.
 *
 * Vite inlines this at build time, so a wrong value cannot be corrected at
 * runtime — it ships. A stale local `.env` once baked in `/api/v1`, which made
 * every request resolve against the dashboard's own origin and 404 silently.
 * A production build without the variable set now fails loudly instead.
 */
const configured = import.meta.env.VITE_API_URL?.trim();

if (import.meta.env.PROD && !configured) {
  throw new Error(
    "VITE_API_URL is not set. Point it at the ML service before building, " +
      "or the dashboard will request its own origin and find nothing.",
  );
}

export const API_BASE = (configured || "http://localhost:8000").replace(
  /\/+$/,
  "",
);

const api = axios.create({
  baseURL: API_BASE,
  // The free tier cold-starts. A short timeout would turn a wake-up into an
  // error toast; the UI shows a waking state instead.
  timeout: 90_000,
  headers: { "Content-Type": "application/json" },
});

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail =
      err.response?.data?.detail ??
      err.response?.data?.message ??
      err.message ??
      "Request failed";
    const message = Array.isArray(detail)
      ? detail.map((d: { msg?: string }) => d.msg ?? String(d)).join("; ")
      : String(detail);
    return Promise.reject(new ApiError(message, err.response?.status));
  },
);

export default api;

/** URL for the SSE stream. EventSource cannot go through axios. */
export const streamUrl = () => `${API_BASE}/api/stream`;
