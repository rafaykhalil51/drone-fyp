import ReportActions from "@/components/ReportActions";
import TargetMatrix from "@/components/TargetMatrix";

export default function AnalysisResults({ result }) {
  return (
    <div className="flex flex-col gap-4">
      {result ? (
        <p className="text-xs tracking-[0.18em] text-slate-400">
          ANALYSIS COMPLETE // {result.video || "VIDEO"}
        </p>
      ) : null}
      <TargetMatrix tracks={result ? result.tracks || [] : null} />
      <ReportActions analysisId={result?.analysis_id} />
    </div>
  );
}
