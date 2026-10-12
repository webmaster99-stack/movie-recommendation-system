"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";
import { useAuth } from "@/components/auth-provider";
import { destinationAfterLogin } from "@/components/require-auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/lib/supabase";

const MIN_PASSWORD_LENGTH = 8;

const COPY = {
  login: {
    title: "Log in",
    submit: "Log in",
    submitting: "Logging in…",
    other: { question: "No account yet?", label: "Sign up", href: "/signup" },
  },
  signup: {
    title: "Create your account",
    submit: "Create account",
    submitting: "Creating account…",
    other: { question: "Already have an account?", label: "Log in", href: "/login" },
  },
};

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const copy = COPY[mode];
  const router = useRouter();
  const { status } = useAuth();
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | undefined>(undefined);
  const [confirmationSentTo, setConfirmationSentTo] = useState<string | undefined>(undefined);

  // Covers both a visitor who is already signed in and the moment signing in succeeds.
  useEffect(() => {
    if (status === "signedIn") router.replace(destinationAfterLogin(window.location.search));
  }, [status, router]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const email = String(form.get("email")).trim();
    const password = String(form.get("password"));
    setBusy(true);
    setProblem(undefined);
    try {
      if (mode === "login") {
        const { error } = await supabase().auth.signInWithPassword({ email, password });
        if (error) throw error;
      } else {
        const { data, error } = await supabase().auth.signUp({
          email,
          password,
          options: { emailRedirectTo: `${window.location.origin}/for-you` },
        });
        if (error) throw error;
        // No session yet means the project asks new users to confirm their address first.
        if (!data.session) setConfirmationSentTo(email);
      }
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  if (confirmationSentTo) {
    return (
      <div className="container-page max-w-md py-16">
        <h1 className="text-5xl font-bold">Check your inbox</h1>
        <p className="mt-4 text-muted-foreground">
          We sent a confirmation link to {confirmationSentTo}. Open it to finish creating your
          account; it brings you straight back here.
        </p>
      </div>
    );
  }

  return (
    <div className="container-page max-w-md py-16">
      <h1 className="text-5xl font-bold">{copy.title}</h1>
      <form onSubmit={submit} className="mt-8 flex flex-col gap-5">
        <div className="flex flex-col gap-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            required
            className="h-11"
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            name="password"
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            minLength={mode === "signup" ? MIN_PASSWORD_LENGTH : undefined}
            aria-describedby={mode === "signup" ? "password-hint" : undefined}
            required
            className="h-11"
          />
          {mode === "signup" && (
            <p id="password-hint" className="text-sm text-muted-foreground">
              At least {MIN_PASSWORD_LENGTH} characters.
            </p>
          )}
        </div>
        {problem && (
          <p role="alert" className="text-sm text-destructive">
            {problem}
          </p>
        )}
        <Button type="submit" disabled={busy} className="h-11 text-base">
          {busy ? copy.submitting : copy.submit}
        </Button>
      </form>
      <p className="mt-6 text-sm text-muted-foreground">
        {copy.other.question}{" "}
        <Link href={copy.other.href} className="text-foreground underline">
          {copy.other.label}
        </Link>
      </p>
    </div>
  );
}
