import type { Metadata } from "next";
import { RequireAuth } from "@/components/require-auth";
import { ForYou } from "./for-you";

export const metadata: Metadata = { title: "For you" };

export default function ForYouPage() {
  return (
    <RequireAuth>
      <ForYou />
    </RequireAuth>
  );
}
