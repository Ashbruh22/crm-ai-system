/**
 * Fire a CRM event at the deal and watch the score respond.
 *
 * The buttons name what happens in the sales world ("Prospect replied"), not
 * the enum the model consumes. Ingestion is asynchronous — the API returns 202
 * and the score arrives over SSE — so the button reports "queued" and then
 * clears when the update lands, rather than pretending to be synchronous.
 */
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import { useSimulateEvent } from "../api/hooks";
import { activityLabel } from "../lib/format";

/** The events worth demonstrating: two that help, two that hurt, one neutral. */
const CHOICES = [
  "email_replied",
  "meeting_held",
  "demo_done",
  "champion_identified",
  "discount_requested",
  "no_activity_7d",
] as const;

const HURTS = new Set(["discount_requested", "no_activity_7d"]);

interface Props {
  dealId: string;
  /** Bumped by the page when an SSE update for this deal arrives. */
  updateToken?: number;
  disabled?: boolean;
  disabledReason?: string;
}

export function SimulateEvents({
  dealId,
  updateToken,
  disabled,
  disabledReason,
}: Props) {
  const simulate = useSimulateEvent(dealId);
  const [queued, setQueued] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The score arriving is what "done" means, so clear on the update, not on
  // the 202 response.
  useEffect(() => {
    if (updateToken) setQueued(null);
  }, [updateToken]);

  async function fire(type: string) {
    setError(null);
    setQueued(type);
    try {
      await simulate.mutateAsync(type);
    } catch (err) {
      setQueued(null);
      setError(err instanceof Error ? err.message : "Could not send the event");
    }
  }

  return (
    <section>
      <div className="flex items-baseline justify-between border-b border-rule pb-2">
        <h2 className="text-sm font-semibold">Send an event</h2>
        {queued && (
          <span className="flex items-center gap-1.5 text-micro text-ink-muted">
            <Loader2 className="h-3 w-3 animate-spin" />
            scoring {activityLabel(queued).toLowerCase()}
          </span>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {CHOICES.map((type) => (
          <button
            key={type}
            type="button"
            onClick={() => fire(type)}
            disabled={disabled || queued !== null}
            title={disabled ? disabledReason : undefined}
            className={`btn ${
              HURTS.has(type)
                ? "border-drag-soft text-drag hover:bg-drag-wash"
                : "border-lift-soft text-lift hover:bg-lift-wash"
            }`}
          >
            {activityLabel(type)}
          </button>
        ))}
      </div>

      {disabled && disabledReason && (
        <p className="mt-3 border-l-2 border-drag pl-2 text-micro text-ink-muted">
          {disabledReason}
        </p>
      )}
      {error && <p className="mt-3 text-micro text-drag">{error}</p>}
      {!disabled && !error && (
        <p className="mt-3 text-micro text-ink-faint">
          The event goes onto the stream, the consumer re-scores the deal, and
          the new score arrives here over a live connection.
        </p>
      )}
    </section>
  );
}
