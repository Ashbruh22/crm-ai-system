/** Data hooks. Query keys are shaped so an SSE event can invalidate precisely. */
import { useEffect, useRef, useState } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from "@tanstack/react-query";

import api, { streamUrl } from "./client";
import type {
  DealDetailResponse,
  DealListResponse,
  ExplainResponse,
  HealthResponse,
  MetricsResponse,
  ScoreResponse,
  ScoreUpdatedEvent,
  WhatIfResponse,
} from "../types/api";

export const keys = {
  health: ["health"] as const,
  metrics: ["metrics"] as const,
  deals: (params?: Record<string, unknown>) => ["deals", params ?? {}] as const,
  deal: (id: string) => ["deal", id] as const,
  explain: (id: string) => ["explain", id] as const,
  actions: (status?: string) => ["actions", status ?? "all"] as const,
};

export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: keys.health,
    queryFn: async () => (await api.get("/healthz")).data,
    // Polls only while the answer is bad news, so a cold start resolves itself.
    refetchInterval: (query) => (query.state.data?.status === "ok" ? false : 4000),
    retry: 6,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
  });
}

export function useMetrics(options?: Partial<UseQueryOptions<MetricsResponse>>) {
  return useQuery<MetricsResponse>({
    queryKey: keys.metrics,
    queryFn: async () => (await api.get("/api/meta/metrics")).data,
    staleTime: Infinity,
    ...options,
  });
}

export function useDeals(params: {
  stage?: string;
  sort?: string;
  order?: string;
  limit?: number;
}) {
  return useQuery<DealListResponse>({
    queryKey: keys.deals(params),
    queryFn: async () => (await api.get("/api/deals", { params })).data,
  });
}

export function useDeal(id: string | undefined) {
  return useQuery<DealDetailResponse>({
    queryKey: keys.deal(id ?? ""),
    queryFn: async () => (await api.get(`/api/deals/${id}`)).data,
    enabled: Boolean(id),
  });
}

export function useExplain(id: string | undefined, enabled = true) {
  return useQuery<ExplainResponse>({
    queryKey: keys.explain(id ?? ""),
    queryFn: async () => (await api.get(`/api/deals/${id}/explain`)).data,
    enabled: Boolean(id) && enabled,
  });
}

export function useScoreNow(id: string) {
  const qc = useQueryClient();
  return useMutation<ScoreResponse>({
    mutationFn: async () =>
      (await api.post(`/api/deals/${id}/score`, { bypass_cache: true })).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.deal(id) });
      qc.invalidateQueries({ queryKey: keys.explain(id) });
      qc.invalidateQueries({ queryKey: ["deals"] });
    },
  });
}

export function useWhatIf(id: string) {
  return useMutation<WhatIfResponse, Error, Record<string, number>>({
    mutationFn: async (overrides) =>
      (await api.post(`/api/deals/${id}/what-if`, { overrides })).data,
  });
}

export function useSimulateEvent(id: string) {
  return useMutation<{ event_id: string }, Error, string>({
    mutationFn: async (type) =>
      (await api.post("/api/events", { deal_id: id, type })).data,
  });
}

export function useResolveAction(dealId?: string) {
  const qc = useQueryClient();
  return useMutation<
    unknown,
    Error,
    { actionId: string; status: "accepted" | "dismissed" }
  >({
    mutationFn: async ({ actionId, status }) =>
      (await api.patch(`/api/actions/${actionId}`, { status })).data,
    onSuccess: () => {
      if (dealId) qc.invalidateQueries({ queryKey: keys.deal(dealId) });
      qc.invalidateQueries({ queryKey: ["actions"] });
      qc.invalidateQueries({ queryKey: ["deals"] });
    },
  });
}

export type StreamState = "connecting" | "open" | "closed";

/**
 * Subscribe to the service's SSE channel.
 *
 * EventSource reconnects on its own, so there is no retry logic here. The
 * connection is opened once for the whole app rather than per component: two
 * components mounting would otherwise hold two streams open, and the free tier
 * has better things to do.
 */
export function useScoreStream(onScore?: (event: ScoreUpdatedEvent) => void) {
  const [state, setState] = useState<StreamState>("connecting");
  const [lastEvent, setLastEvent] = useState<ScoreUpdatedEvent | null>(null);
  const qc = useQueryClient();
  // Kept in a ref so a new callback identity does not tear down the stream.
  const handler = useRef(onScore);
  handler.current = onScore;

  useEffect(() => {
    const source = new EventSource(streamUrl());

    source.addEventListener("connected", () => setState("open"));
    source.onopen = () => setState("open");

    source.addEventListener("score_updated", (raw) => {
      try {
        const payload: ScoreUpdatedEvent = JSON.parse((raw as MessageEvent).data);
        setLastEvent(payload);
        handler.current?.(payload);
        qc.invalidateQueries({ queryKey: keys.deal(payload.deal_id) });
        qc.invalidateQueries({ queryKey: keys.explain(payload.deal_id) });
        qc.invalidateQueries({ queryKey: ["deals"] });
      } catch {
        /* a malformed frame is not worth tearing the stream down for */
      }
    });

    source.addEventListener("action_created", () => {
      qc.invalidateQueries({ queryKey: ["actions"] });
    });

    source.onerror = () => {
      // EventSource sets readyState CLOSED only when it has given up.
      setState(source.readyState === EventSource.CLOSED ? "closed" : "connecting");
    };

    return () => source.close();
  }, [qc]);

  return { state, lastEvent };
}
