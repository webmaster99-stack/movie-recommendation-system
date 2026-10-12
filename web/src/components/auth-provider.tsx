"use client";

import { useQueryClient } from "@tanstack/react-query";
import type { Session } from "@supabase/supabase-js";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { supabase } from "@/lib/supabase";

export type AuthUser = { id: string; email: string | undefined };

type Auth = {
  /** `loading` until the stored session has been read, which happens only in the browser. */
  status: "loading" | "signedIn" | "signedOut";
  user: AuthUser | undefined;
};

const AuthContext = createContext<Auth>({ status: "loading", user: undefined });

export function useAuth(): Auth {
  return useContext(AuthContext);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [session, setSession] = useState<Session | null | undefined>(undefined);

  useEffect(() => {
    let client;
    try {
      client = supabase();
    } catch (error) {
      // Without the Supabase settings the public pages still work; signing in reports it.
      console.error(error);
      queueMicrotask(() => setSession(null));
      return;
    }
    // Fires once with the stored session, then on every sign-in, sign-out and token refresh.
    const { data } = client.auth.onAuthStateChange((event, next) => {
      setSession(next);
      if (event === "SIGNED_OUT") {
        queryClient.removeQueries({ queryKey: ["ratings"] });
        queryClient.removeQueries({ queryKey: ["recommendations"] });
      }
    });
    return () => data.subscription.unsubscribe();
  }, [queryClient]);

  const userId = session?.user.id;
  const email = session?.user.email;
  const auth = useMemo<Auth>(() => {
    if (session === undefined) return { status: "loading", user: undefined };
    if (userId === undefined) return { status: "signedOut", user: undefined };
    return { status: "signedIn", user: { id: userId, email } };
  }, [session, userId, email]);

  return <AuthContext.Provider value={auth}>{children}</AuthContext.Provider>;
}
