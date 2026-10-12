import { createClient, type SupabaseClient } from "@supabase/supabase-js";

let client: SupabaseClient | undefined;

/**
 * The browser's Supabase client, used for sign-in only: every other request goes to the
 * API with the access token. Created on first use so that a build without the two
 * variables still succeeds.
 */
export function supabase(): SupabaseClient {
  if (!client) {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
    if (!url || !key) {
      throw new Error(
        "Sign-in is not configured: set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY (see web/.env.example).",
      );
    }
    client = createClient(url, key, { auth: { flowType: "pkce" } });
  }
  return client;
}
