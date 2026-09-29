/**
 * The recommendation ledger.
 *
 * Each row shows the rule that fired, so a reviewer can see the advice is
 * derived rather than generated prose. Resolved rows stay visible and struck
 * through rather than disappearing: the table is an audit trail, and a rep
 * wants to see what they already dismissed.
 */
import { Check, X } from "lucide-react";

import { useResolveAction } from "../api/hooks";
import { actionLabel, ago } from "../lib/format";
import type { DealAction, Priority } from "../types/api";

const PRIORITY_STYLE: Record<Priority, string> = {
  CRITICAL: "border-alarm text-alarm",
  HIGH: "border-drag text-drag",
  MEDIUM: "border-rule-strong text-ink-muted",
  LOW: "border-lift text-lift",
};

interface Props {
  actions: DealAction[];
  dealId?: string;
  emptyMessage?: string;
}

export function ActionList({ actions, dealId, emptyMessage }: Props) {
  const resolve = useResolveAction(dealId);

  if (!actions.length) {
    return (
      <p className="mt-3 text-sm text-ink-muted">
        {emptyMessage ?? "No recommendations yet. Score the deal to generate them."}
      </p>
    );
  }

  return (
    <ul className="mt-1">
      {actions.map((action) => {
        const open = action.status === "suggested";
        return (
          <li key={action.id} className="border-b border-rule/60 py-3">
            <div className="flex items-start gap-3">
              <span
                className={`mt-0.5 shrink-0 border-l-2 pl-2 text-micro font-semibold ${
                  PRIORITY_STYLE[action.priority]
                } ${open ? "" : "opacity-50"}`}
              >
                {action.priority.toLowerCase()}
              </span>

              <div className="min-w-0 flex-1">
                <div
                  className={`text-sm font-medium ${open ? "" : "text-ink-faint line-through"}`}
                >
                  {actionLabel(action.action_type)}
                </div>
                <p
                  className={`mt-0.5 max-w-prose text-sm leading-snug ${
                    open ? "text-ink-muted" : "text-ink-faint"
                  }`}
                >
                  {action.reason}
                </p>
                <div className="mt-1 flex flex-wrap items-center gap-x-3 text-micro text-ink-faint">
                  <span className="tnum">{action.rule_id}</span>
                  <span>{ago(action.created_at)}</span>
                  {!open && <span>{action.status}</span>}
                </div>
              </div>

              {open && (
                <div className="flex shrink-0 gap-1">
                  <button
                    type="button"
                    className="btn btn-quiet px-2 py-1"
                    aria-label="Accept this recommendation"
                    disabled={resolve.isPending}
                    onClick={() =>
                      resolve.mutate({ actionId: action.id, status: "accepted" })
                    }
                  >
                    <Check className="h-4 w-4 text-lift" />
                  </button>
                  <button
                    type="button"
                    className="btn btn-quiet px-2 py-1"
                    aria-label="Dismiss this recommendation"
                    disabled={resolve.isPending}
                    onClick={() =>
                      resolve.mutate({ actionId: action.id, status: "dismissed" })
                    }
                  >
                    <X className="h-4 w-4 text-ink-faint" />
                  </button>
                </div>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
