import type { ReactNode } from "react";

export function Section({
  index,
  kicker,
  title,
  id,
  children,
}: {
  index: string;
  kicker?: string;
  title: string;
  id?: string;
  children: ReactNode;
}) {
  return (
    <section className="section" id={id}>
      <div className="shell">
        <div className="section-head">
          <div className="section-index" aria-hidden="true">
            {index}
          </div>
          <div className="section-title-wrap">
            {kicker ? <p className="label label-ember section-kicker">{kicker}</p> : null}
            <h2>{title}</h2>
          </div>
        </div>
        {children}
      </div>
    </section>
  );
}
