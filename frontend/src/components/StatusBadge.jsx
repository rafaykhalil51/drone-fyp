export default function StatusBadge({ label, tone = "pending" }) {
  const tones = {
    active: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
    offline: "border-amber-500/40 bg-amber-500/10 text-amber-200",
    pending: "border-slate-600 bg-slate-800/60 text-slate-400",
  };

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-semibold tracking-[0.14em] ${tones[tone] || tones.pending}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          tone === "active"
            ? "bg-emerald-400 shadow-[0_0_8px_#34d399]"
            : tone === "offline"
              ? "bg-amber-400"
              : "bg-slate-500"
        }`}
      />
      {label}
    </span>
  );
}
