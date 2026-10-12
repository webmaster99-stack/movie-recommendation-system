import type { Metadata } from "next";
import { RequireAuth } from "@/components/require-auth";
import { MyRatings } from "./my-ratings";

export const metadata: Metadata = { title: "My ratings" };

export default function RatingsPage() {
  return (
    <RequireAuth>
      <MyRatings />
    </RequireAuth>
  );
}
