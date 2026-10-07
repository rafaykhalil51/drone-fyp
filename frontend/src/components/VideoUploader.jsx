import { useRef, useState } from "react";
import { startAnalysis } from "@/lib/api";

const ALLOWED = [".mp4", ".mov", ".avi", ".mkv"];

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function VideoUploader({ backendOnline, processing, progressText, onStart, completed }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function chooseFile(next) {
    setError("");
    if (!next) {
      setFile(null);
      return;
    }
    if (!ALLOWED.includes(extensionOf(next.name))) {
      setFile(null);
      setError("Unsupported file. Use MP4, MOV, AVI, or MKV.");
      return;
    }
    setFile(next);
  }

  async function onAnalyze() {
    if (!file || busy || processing) return;
    if (backendOnline === false) {
      setError("Backend offline. Start the API, then try again.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const started = await startAnalysis(file);
      onStart(started);
    } catch (err) {
      console.error("Analysis error:", err);
      if (err.code === "offline") setError("Backend offline. The video was not sent.");
      else if (err.code === "unsupported") setError(err.message);
      else if (err.code === "model") setError(err.message || "A required model is unavailable.");
      else setError(err.message || "Analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="flex h-full flex-col gap-4 rounded-2xl border border-cyan-400/15 bg-[#0d1117] p-5">
      <div>
        <p className="text-[11px] font-semibold tracking-[0.22em] text-cyan-300/80">
          FEED CONTROL
        </p>
        <p className="mt-2 text-sm text-slate-400">Upload a surveillance clip. Supported formats: MP4, MOV, AVI, MKV.</p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.mov,.avi,.mkv,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
        className="hidden"
        disabled={busy}
        onChange={(event) => chooseFile(event.target.files?.[0] || null)}
      />

      <button
        type="button"
        disabled={busy || processing}
        onClick={() => inputRef.current?.click()}
        className="rounded-xl border border-cyan-400/30 bg-cyan-400/10 px-4 py-3 text-sm font-semibold tracking-[0.16em] text-cyan-100 transition hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        UPLOAD VIDEO
      </button>

      {file ? (
        <div className="rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm">
          <p className="truncate text-slate-100">{file.name}</p>
          <p className="mt-1 text-xs tracking-wide text-slate-500">{formatSize(file.size)}</p>
        </div>
      ) : (
        <p className="text-xs tracking-wide text-slate-500">No video selected.</p>
      )}

      <button
        type="button"
        disabled={!file || busy || processing}
        onClick={onAnalyze}
        className="rounded-xl bg-cyan-400 px-4 py-3 text-sm font-semibold tracking-[0.18em] text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
      >
        {busy || processing ? "ANALYZING" : completed ? "ANALYZE AGAIN" : "ANALYZE VIDEO"}
      </button>

      {busy || processing ? (
        <div className="overflow-hidden rounded-xl border border-cyan-400/20 bg-black/50 px-4 py-4">
          <p className="text-xs font-semibold tracking-[0.22em] text-cyan-200">ANALYSIS IN PROGRESS</p>
          <p className="mt-2 truncate text-sm text-slate-300">{file?.name}</p>
          <p className="mt-1 text-xs text-slate-500">{progressText || "Waiting for the first annotated frame."}</p>
          <div className="aegis-scan mt-4 h-1.5 overflow-hidden rounded-full bg-slate-800">
            <span />
          </div>
        </div>
      ) : null}

      {error ? (
        <p className="rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-sm text-amber-100" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}
