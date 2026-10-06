export default function StatCard({ kicker, label, value, accent = "cyan" }) {
  const accents = {
    cyan: "text-cyan-300 shadow-[inset_0_0_0_1px_rgba(34,211,238,0.25)]",
    amber: "text-amber-300 shadow-[inset_0_0_0_1px_rgba(251,191,36,0.28)]",
    orange: "text-orange-300 shadow-[inset_0_0_0_1px_rgba(251,146,60,0.28)]",
    purple: "text-purple-300 shadow-[inset_0_0_0_1px_rgba(192,132,252,0.3)]",
    slate: "text-slate-200 shadow-[inset_0_0_0_1px_rgba(148,163,184,0.25)]",
  };

  const shown = value === null || value === undefined ? "—" : value;

  return (
    <article className={`rounded-2xl border border-white/10 bg-[#10141c]/90 p-4 ${accents[accent] || accents.cyan}`}>
      <p className="text-[10px] font-semibold tracking-[0.22em] text-slate-500">{kicker}</p>
      <p className="mt-3 text-3xl font-semibold tabular-nums">{shown}</p>
      <p className="mt-1 text-xs tracking-wide text-slate-400">{label}</p>
    </article>
  );
}
