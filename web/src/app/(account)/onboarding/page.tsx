import type { Metadata } from "next";
import { RequireAuth } from "@/components/require-auth";
import { Onboarding } from "./onboarding";

export const metadata: Metadata = { title: "Rate movies" };

export default function OnboardingPage() {
  return (
    <RequireAuth>
      <Onboarding />
    </RequireAuth>
  );
}
