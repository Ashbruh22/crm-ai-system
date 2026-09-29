/**
 * App shell.
 *
 * The synthetic-data notice lives in the header, not a dismissible banner: it is
 * the single most important thing to know about every number on the page, and
 * the spec requires it be unmissable.
 */
import { Link, NavLink, Outlet } from "react-router-dom";

import { useHealth, useScoreStream } from "../api/hooks";
import type { StreamState } from "../api/hooks";

const NAV = [
  { to: "/", label: "Pipeline", end: true },
  { to: "/architecture", label: "How it works" },
];

function StreamDot({ state }: { state: StreamState }) {
  const style =
    state === "open"
      ? "bg-lift"
      : state === "connecting"
        ? "bg-drag animate-pulse"
        : "bg-rule-strong";
  const label =
    state === "open"
      ? "Live"
      : state === "connecting"
        ? "Connecting"
        : "Not live";
  return (
    <span className="flex items-center gap-1.5 text-micro text-ink-muted">
      <span className={`h-1.5 w-1.5 rounded-full ${style}`} />
      {label}
    </span>
  );
}

export function Layout() {
  // One stream for the whole app; components read cache invalidations from it.
  const { state } = useScoreStream();
  const health = useHealth();
  const busDown = health.data?.bus?.available === false;

  return (
    <div className="min-h-screen">
      <header className="border-b border-rule bg-paper-raised">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <Link to="/" className="font-semibold tracking-tight">
            CRM AI Core
          </Link>

          <nav className="flex gap-4">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `text-sm ${
                    isActive
                      ? "font-medium text-ink"
                      : "text-ink-muted hover:text-ink"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-4">
            <StreamDot state={state} />
            <span className="border border-rule-strong px-1.5 py-0.5 text-micro text-ink-muted">
              synthetic data
            </span>
          </div>
        </div>
      </header>

      {busDown && (
        <div className="border-b border-drag-soft bg-drag-wash">
          <p className="mx-auto max-w-6xl px-4 py-2 text-micro text-ink-muted sm:px-6">
            Event ingestion is unavailable, so sending events will not update
            scores. Everything else works.
          </p>
        </div>
      )}

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        <Outlet />
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-10 pt-4 text-micro text-ink-faint sm:px-6">
        Every deal, company and rep on this page is generated. No real CRM data
        is used anywhere in this project.
      </footer>
    </div>
  );
}
