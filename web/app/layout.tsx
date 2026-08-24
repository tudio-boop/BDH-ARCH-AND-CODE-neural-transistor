import type { Metadata, Viewport } from "next";

import { SiteFooter } from "./components/SiteFooter";
import { SiteNav } from "./components/SiteNav";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Baby Dragon Hatchling — a brain-shaped language model, explained",
    template: "%s — Baby Dragon Hatchling",
  },
  description:
    "BDH keeps its working memory in synapses instead of a KV cache. Plain-English explainer, a walk through one BDH-GPU layer, the real numbers from a local CPU training run, and a tiny BDH-GPU you can run in your browser.",
  keywords: [
    "BDH",
    "Dragon Hatchling",
    "BDH-GPU",
    "linear attention",
    "state space model",
    "Hebbian learning",
    "interpretability",
    "Pathway",
  ],
  openGraph: {
    title: "Baby Dragon Hatchling — a brain-shaped language model, explained",
    description:
      "Working memory as synapses, not a KV cache. With a tiny BDH-GPU that runs in your browser.",
    type: "website",
  },
  robots: { index: true, follow: true },
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
