"use client";

import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";
import { useAuth } from "@/components/auth-provider";
import { useHydrated } from "@/components/browser-only";
import { MovieGridSkeleton } from "@/components/movie-card";

/**
 * Shows its children to a signed-in user and sends anyone else to the login page, which
 * brings them back afterwards. This is for navigation only: what protects the data is the
 * API, which checks the access token on every request.
 */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const router = useRouter();
  // Who is signed in is only known in the browser: the prerendered page is the skeleton.
  const hydrated = useHydrated();

  useEffect(() => {
    if (status === "signedOut") {
      router.replace(`/login?next=${encodeURIComponent(window.location.pathname)}`);
    }
  }, [status, router]);

  if (!hydrated || status !== "signedIn") return <MovieGridSkeleton />;
  return children;
}

/** Where to go after signing in: the `next` page if it is one of ours, else the home page. */
export function destinationAfterLogin(search: string): string {
  const next = new URLSearchParams(search).get("next");
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/for-you";
}
