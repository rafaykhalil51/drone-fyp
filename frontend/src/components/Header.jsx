import StatusBadge from "@/components/StatusBadge";

function toneFor(active) {
  if (active === null) return "pending";
  return active ? "active" : "offline";
}

export default function Header({ personActive, accessoryActive, backendOnline }) {
  return (
    <header className="flex flex-col gap-5 rounded-2xl border border-cyan-400/20 bg-gradient-to-r from-black via-[#0c1218] to-[#0b0d11] p-5 shadow-[0_0_42px_rgba(0,210,255,0.08)] lg:flex-row lg:items-center lg:justify-between">
      <div className="flex items-center gap-4">
        <img
          src="/branding/avi_logo.jpg"
          alt="A.E.G.I.S"
          className="h-16 w-16 rounded-xl border border-cyan-400/30 object-cover shadow-[0_0_24px_rgba(0,210,255,0.25)]"
        />
        <div>
          <h1 className="text-lg font-semibold leading-tight tracking-[0.12em] text-white sm:text-xl lg:whitespace-nowrap lg:text-[1.35rem]">
            A.E.G.I.S <span className="text-cyan-300">//</span> VISION INTELLIGENCE
          </h1>
          <p className="mt-1 max-w-xl text-[11px] tracking-[0.16em] text-slate-400 uppercase">
            Autonomous multi-target tracking & cranial/facial accessory telemetry
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <StatusBadge
          label={personActive ? "PERSON AI ACTIVE" : personActive === null ? "PERSON AI" : "PERSON AI OFFLINE"}
          tone={toneFor(personActive)}
        />
        <StatusBadge
          label={
            accessoryActive
              ? "ACCESSORY AI ACTIVE"
              : accessoryActive === null
                ? "ACCESSORY AI"
                : "ACCESSORY AI OFFLINE"
          }
          tone={toneFor(accessoryActive)}
        />
        <StatusBadge
          label={backendOnline ? "BACKEND ONLINE" : backendOnline === null ? "BACKEND" : "BACKEND OFFLINE"}
          tone={toneFor(backendOnline)}
        />
      </div>
    </header>
  );
}
