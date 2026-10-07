import { useRef } from "react";
import ReportActions from "@/components/ReportActions";
import TargetMatrix from "@/components/TargetMatrix";

export default function AnalysisResults({ result, streamUrl, videoUrl, videoDetail, liveTracks }) {
  const videoRef = useRef(null);
  const tracks = result ? result.tracks || [] : liveTracks;

  function replay() {
    const node = videoRef.current;
    if (!node) return;
    node.currentTime = 0;
    const started = node.play();
    if (started && typeof started.catch === "function") started.catch(() => {});
  }

  return (
    <div className="flex flex-col gap-4">
      <section className="overflow-hidden rounded-2xl border border-cyan-400/20 bg-black/40 p-4">
        <p className="text-[11px] font-semibold tracking-[0.22em] text-cyan-300/80">
          SURVEILLANCE FEED // TACTICAL VIEWPORT
        </p>
        {streamUrl ? (
          <img
            src={streamUrl}
            alt="Live annotated surveillance feed"
            className="mt-4 max-h-[640px] w-full rounded-xl bg-black object-contain"
          />
        ) : videoUrl ? (
          <div className="mt-4">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              playsInline
              preload="metadata"
              className="max-h-[640px] w-full rounded-xl bg-black object-contain"
            />
            <button
              type="button"
              onClick={replay}
              className="mt-3 rounded-xl border border-cyan-400/30 px-4 py-2 text-xs font-semibold tracking-[0.16em] text-cyan-100 hover:bg-cyan-400/10"
            >
              REPLAY ANALYSIS
            </button>
          </div>
        ) : (
          <p className="mt-8 mb-6 text-center text-sm tracking-wide text-slate-500">
            {videoDetail || "Awaiting surveillance feed."}
          </p>
        )}
      </section>
      {result ? (
        <p className="text-xs tracking-[0.18em] text-slate-400">
          ANALYSIS COMPLETE // {result.video || "VIDEO"}
        </p>
      ) : null}
      {tracks ? <TargetMatrix tracks={tracks} /> : null}
      <ReportActions analysisId={result?.analysis_id} videoUrl={videoUrl} />
    </div>
  );
}
