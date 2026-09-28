/**
 * Cold-start state (spec section 9).
 *
 * Render's free tier spins the container down when idle, so the first request
 * after a quiet period takes 30–60 seconds. A bare spinner for a minute reads
 * as broken; saying what is happening and why costs one sentence and buys the
 * reviewer's patience.
 *
 * The explanation only appears after a few seconds, so a warm service does not
 * apologise for being fast.
 */
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

export function WakingNotice({ label = "Loading the pipeline" }: { label?: string }) {
  const [slow, setSlow] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => setSlow(true), 3000);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <div className="flex items-start gap-3 py-10">
      <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-ink-muted" />
      <div>
        <p className="text-sm">{label}…</p>
        {slow && (
          <p className="mt-1 max-w-prose text-sm text-ink-muted">
            The model service sleeps when nobody is using it, so the first
            request wakes it up. This usually takes 30 to 60 seconds.
          </p>
        )}
      </div>
    </div>
  );
}

/** Row-shaped placeholder, for keeping table layout stable while refetching. */
export function SkeletonRows({ rows = 6 }: { rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i} className="border-b border-rule/60">
          <td colSpan={6} className="py-3">
            <div className="h-3 w-full animate-pulse bg-paper-sunken" />
          </td>
        </tr>
      ))}
    </>
  );
}
