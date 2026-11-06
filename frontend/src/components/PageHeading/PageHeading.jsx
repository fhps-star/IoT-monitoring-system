import "./PageHeading.css";

export function SectionTitle({ title, subtitle, children }) {
  return (
    <div className="section-title-row">
      <div>

        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="section-actions">{children}</div>
    </div>
  )
}

export function PanelHeading({ title, meta, action }) {
  return (
    <div className="panel-heading">
      <div>
        <h2>{title}</h2>
        {meta && <span className="panel-meta">{meta}</span>}
      </div>
      {action && <div className="panel-action">{action}</div>}
    </div>
  )
}
