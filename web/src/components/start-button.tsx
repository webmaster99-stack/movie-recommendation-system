"use client";

import Link from "next/link";
import { useHydrated } from "@/components/browser-only";
import { useAuth } from "@/components/auth-provider";
import { Button } from "@/components/ui/button";

/** The landing page's main action, which depends on whether the visitor has an account open. */
export function StartButton() {
  const { status } = useAuth();
  const signedIn = useHydrated() && status === "signedIn";
  return (
    <Button className="h-12 px-6 text-base" asChild>
      <Link href={signedIn ? "/for-you" : "/signup"}>
        {signedIn ? "See my recommendations" : "Create an account"}
      </Link>
    </Button>
  );
}
