const TAGS = [
  ["cap", "CAP"],
  ["mask", "MASK"],
  ["glasses", "GLASSES"],
  ["headphones", "HEADPHONES"],
];

function tagsFor(track) {
  const present = TAGS.filter(([key]) => track[key] === true).map(([, label]) => label);
  if (present.length > 0) return present;
  const decided = TAGS.every(([key]) => track[key] === false);
  return decided ? ["PLAIN"] : ["UNCONFIRMED"];
}

export default function TargetMatrix({ tracks }) {
  if (!tracks) {
    return (
      <section className="flex min-h-48 items-center justify-center rounded-2xl border border-dashed border-white/10 bg-black/30 p-6 text-sm tracking-wide text-slate-500">
        Awaiting surveillance feed.
      </section>
    );
  }

  return (
    <section className="rounded-2xl border border-purple-400/20 bg-[#0d1117] p-5">
      <p className="text-[11px] font-semibold tracking-[0.22em] text-purple-300/90">TARGET MATRIX</p>
      {tracks.length === 0 ? (
        <p className="mt-4 text-sm text-slate-400">No tracked people were returned.</p>
      ) : (
        <ul className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {tracks.map((track) => (
            <li key={track.track_id} className="rounded-xl border border-white/10 bg-black/40 px-4 py-3">
              <p className="text-xs tracking-[0.2em] text-cyan-200/80">ID {track.track_id}</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {tagsFor(track).map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full border border-amber-300/30 bg-amber-300/10 px-2 py-1 text-[10px] font-semibold tracking-[0.16em] text-amber-100"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
