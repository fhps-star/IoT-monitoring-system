import "./StatusSwitch.css";

const DEFAULT_OPTIONS = [
  { value: "Active", label: "Active" },
  { value: "Acknowledged", label: "Acknowledged" },
  { value: "Resolved", label: "Resolved" },
];

export function StatusSwitch({ value, onChange, label, options = DEFAULT_OPTIONS }) {
  return (
    <div className="status-switch" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          className={`status-switch-option tone-${option.value.toLowerCase()}${value === option.value ? " selected" : ""}`}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
