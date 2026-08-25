import Link from "next/link";

import { PAPER, UPSTREAM_REPO } from "@/lib/facts";
import styles from "./SiteNav.module.css";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/architecture", label: "One layer" },
  { href: "/lab", label: "Lab" },
];

export function SiteNav() {
  return (
    <header className={styles.header}>
      <div className={`shell ${styles.inner}`}>
        <Link href="/" className={styles.brand}>
          <svg
            className={styles.mark}
            viewBox="0 0 24 30"
            aria-hidden="true"
            focusable="false"
          >
            <path
              d="M12 1.6C7.6 1.6 3.4 8.4 3.4 15.4c0 7 3.9 13 8.6 13s8.6-6 8.6-13C20.6 8.4 16.4 1.6 12 1.6Z"
              fill="rgba(255,122,47,0.14)"
              stroke="var(--ember)"
              strokeWidth="1.3"
            />
            <path
              d="M16 6l-2 3.2 2.6 1.5-3 2.6"
              fill="none"
              stroke="var(--ember-pale)"
              strokeWidth="1.3"
              strokeLinecap="round"
            />
            <circle cx="10" cy="17.5" r="2" fill="var(--ember-pale)" />
            <circle cx="15" cy="21.5" r="1.4" fill="var(--ember)" />
            <path
              d="M10 17.5l5 4"
              stroke="var(--ember)"
              strokeWidth="1"
              opacity="0.8"
            />
          </svg>
          <span className={styles.wordmark}>
            Baby Dragon <em>Hatchling</em>
          </span>
        </Link>

        <nav className={styles.nav} aria-label="Sections">
          {LINKS.map((link) => (
            <Link key={link.href} href={link.href} className={styles.link}>
              {link.label}
            </Link>
          ))}
          <a
            className={`${styles.link} ${styles.external}`}
            href={PAPER.url}
            target="_blank"
            rel="noreferrer"
          >
            arXiv {PAPER.arxiv}
          </a>
          <a
            className={`${styles.link} ${styles.external}`}
            href={UPSTREAM_REPO}
            target="_blank"
            rel="noreferrer"
          >
            Code
          </a>
        </nav>
      </div>
    </header>
  );
}
