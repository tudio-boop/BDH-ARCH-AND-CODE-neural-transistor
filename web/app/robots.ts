import type { MetadataRoute } from "next";

/**
 * Team-only demo: ask every crawler to stay out. Deployment protection is what
 * actually keeps this private; this is just belt and braces so the site cannot
 * end up in a search index if it is ever served without protection.
 */
export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: "*", disallow: "/" }],
  };
}
