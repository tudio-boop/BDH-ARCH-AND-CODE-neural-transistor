import type { Metadata, Viewport } from "next";

import { SiteFooter } from "./components/SiteFooter";
import { SiteNav } from "./components/SiteNav";
import "./globals.css";

/**
 * This is an internal, team-only demo. It is deliberately not discoverable:
 * no indexing, no keywords, and no Open Graph card, since those exist only to
 * make a page circulate publicly. See also app/robots.ts.
 */
export const metadata: Metadata = {
  title: {
    default: "Baby Dragon Hatchling — a brain-shaped language model, explained",
    template: "%s — Baby Dragon Hatchling",
  },
  description:
    "Internal demo. BDH keeps its working memory in synapses instead of a KV cache.",
  robots: {
    index: false,
    follow: false,
    nocache: true,
    googleBot: { index: false, follow: false, noimageindex: true },
  },
};

export const viewport: Viewport = {
  themeColor: "#0a0705",
  colorScheme: "dark",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <SiteNav />
        <main>{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
