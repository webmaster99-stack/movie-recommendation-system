import type { Metadata } from "next";
import { Big_Shoulders, Hanken_Grotesk } from "next/font/google";
import { Providers } from "@/components/providers";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { WakeUpNotice } from "@/components/wake-up-notice";
import { APP_NAME } from "@/lib/config";
import "./globals.css";

// Headings in a condensed poster face, the kind used for a film's billing block.
const display = Big_Shoulders({
  variable: "--font-display",
  subsets: ["latin"],
  // next/font has no metrics for this face to size a fallback with.
  adjustFontFallback: false,
});
const body = Hanken_Grotesk({ variable: "--font-body", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: APP_NAME, template: `%s | ${APP_NAME}` },
  description:
    "Rate ten movies you have seen and get a list of what to watch next, from a recommender trained on 32 million MovieLens ratings.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} dark h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <Providers>
          <SiteHeader />
          <WakeUpNotice />
          <main className="flex-1">{children}</main>
          <SiteFooter />
        </Providers>
      </body>
    </html>
  );
}
