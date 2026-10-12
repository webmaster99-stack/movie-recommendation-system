"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { AuthProvider } from "@/components/auth-provider";
import { Toaster } from "@/components/ui/sonner";
import { isWakingUp } from "@/lib/api/client";

// The free host stops the API after 15 idle minutes and needs up to a minute to start it
// again. Requests made meanwhile are retried for about two minutes before giving up.
const WAKE_UP_RETRIES = 40;
const WAKE_UP_RETRY_MS = 3000;

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60_000,
        refetchOnWindowFocus: false,
        retry: (failures, error) => isWakingUp(error) && failures < WAKE_UP_RETRIES,
        retryDelay: WAKE_UP_RETRY_MS,
      },
    },
  });
}

let browserQueryClient: QueryClient | undefined;

function getQueryClient() {
  // A new client for each server render; one for the whole visit in the browser.
  if (typeof window === "undefined") return makeQueryClient();
  browserQueryClient ??= makeQueryClient();
  return browserQueryClient;
}

export function Providers({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={getQueryClient()}>
      <AuthProvider>{children}</AuthProvider>
      <Toaster position="bottom-center" />
    </QueryClientProvider>
  );
}
