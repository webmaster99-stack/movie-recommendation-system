"use client";

import { useSyncExternalStore, type ReactNode } from "react";

const subscribe = () => () => {};

/**
 * False on the server and while the server's HTML is being hydrated, true from then on.
 *
 * The pages are prerendered without any data: who is signed in and what the API answers are
 * only known in the browser. A page can be hydrated after the layout around it has already
 * started loading those, and would then render something other than the HTML it is given.
 * Components that show such state use this to render their empty state first.
 */
export function useHydrated(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}

/** Renders `fallback` in the prerendered HTML and `children` in the browser. */
export function BrowserOnly({
  children,
  fallback = null,
}: {
  children: ReactNode;
  fallback?: ReactNode;
}) {
  return useHydrated() ? children : fallback;
}
