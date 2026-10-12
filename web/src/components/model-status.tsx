"use client";

import { useHydrated } from "@/components/browser-only";
import { useHealth } from "@/lib/api/hooks";

/** What the API is serving right now, from its `/health` endpoint. */
export function ModelStatus() {
  const health = useHealth();
  const hydrated = useHydrated();
  const data = hydrated ? health.data : undefined;
  if (hydrated && health.isError)
    return <p className="text-muted-foreground">The server can&apos;t be reached.</p>;

  const facts = [
    ["Model", data?.model],
    ["Registry version", data?.model_version],
    ["Movies in the catalog", data?.n_movies.toLocaleString("en")],
    ["Trained on ratings up to", data?.trained_through],
    ["Built from commit", data?.commit.slice(0, 7)],
  ];
  return (
    <dl className="grid max-w-xl grid-cols-[auto_1fr] gap-x-8 gap-y-2">
      {facts.map(([name, value]) => (
        <div key={name} className="col-span-2 grid grid-cols-subgrid border-b py-2">
          <dt className="text-muted-foreground">{name}</dt>
          <dd className="font-medium">{value ?? "…"}</dd>
        </div>
      ))}
    </dl>
  );
}
