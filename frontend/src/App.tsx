/**
 * Routes.
 *
 * Two things the previous version did that are gone:
 *
 * - A login gate in front of everything. Reads are public now; the data is
 *   synthetic and a sign-in wall on a portfolio demo only loses reviewers.
 * - A `DemoSeeder` that POSTed 15 simulated events from the browser on first
 *   load to populate the pipeline. The service seeds its own live set at
 *   startup, so this was both redundant and a client doing a server's job.
 */
import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { Layout } from "./components/Layout";
import { WakingNotice } from "./components/WakingNotice";
import { Deal } from "./pages/Deal";
import { Pipeline } from "./pages/Pipeline";

// Reference material rather than the working surface, so it loads on demand.
const Architecture = lazy(() =>
  import("./pages/Architecture").then((m) => ({ default: m.Architecture })),
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Scores change on ingestion, and the stream invalidates them, so
      // refetching on every window focus is noise.
      refetchOnWindowFocus: false,
      staleTime: 10_000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />} path="/">
            <Route index element={<Pipeline />} />
            <Route path="deals/:id" element={<Deal />} />
            <Route
              path="architecture"
              element={
                <Suspense fallback={<WakingNotice label="Loading" />}>
                  <Architecture />
                </Suspense>
              }
            />
            <Route
              path="*"
              element={
                <div className="max-w-prose">
                  <h1 className="text-lg font-semibold">No such page</h1>
                  <p className="mt-2 text-sm text-ink-muted">
                    Check the address, or go back to the pipeline.
                  </p>
                </div>
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
