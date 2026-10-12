"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { useHealth } from "@/lib/api/hooks";

// A healthy server answers in well under this, so the notice never flashes on a normal visit.
const PATIENCE_MS = 2500;

/**
 * Explains a slow first load. The API's free host stops it after 15 idle minutes; the
 * next request starts it again, which takes up to a minute. Every page's own requests are
 * retried meanwhile (see `providers.tsx`), so the page fills in by itself.
 */
export function WakeUpNotice() {
  const health = useHealth();
  const [waited, setWaited] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setWaited(true), PATIENCE_MS);
    return () => clearTimeout(timer);
  }, []);

  if (health.isSuccess || (health.isPending && !waited)) return null;

  return (
    <div role="status" className="border-b bg-card">
      <div className="container-page flex flex-wrap items-center gap-x-4 gap-y-2 py-3 text-sm">
        {health.isError ? (
          <>
            <p>The server can&apos;t be reached, so movies and ratings won&apos;t load.</p>
            <Button variant="outline" size="sm" onClick={() => health.refetch()}>
              Try again
            </Button>
          </>
        ) : (
          <>
            <span className="size-2 animate-pulse rounded-full bg-primary" aria-hidden />
            <p>
              Waking up the server. It sleeps when nobody has visited for a while and takes up to a
              minute to start; this page fills in by itself.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
