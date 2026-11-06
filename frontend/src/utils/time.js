export function alertAgeMinutes(age) {
  if (typeof age !== "string") return Number.POSITIVE_INFINITY;
  const [, amount, unit] = age.match(/(\d+)\s*(min|hr|day)/i) ?? [];
  if (!amount) return Number.POSITIVE_INFINITY;
  const multiplier = unit.toLowerCase() === "min" ? 1 : unit.toLowerCase() === "hr" ? 60 : 1440;
  return Number(amount) * multiplier;
}
