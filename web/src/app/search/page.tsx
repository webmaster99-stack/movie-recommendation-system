import type { Metadata } from "next";
import { Suspense } from "react";
import { BrowserOnly } from "@/components/browser-only";
import { Search } from "./search";

export const metadata: Metadata = { title: "Search" };

export default function SearchPage() {
  return (
    <div className="container-page py-10">
      {/* The query string is only known in the browser, so the page's shell renders without it. */}
      <Suspense>
        <BrowserOnly>
          <Search />
        </BrowserOnly>
      </Suspense>
    </div>
  );
}
