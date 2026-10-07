import { reportCsvUrl, reportJsonUrl } from "@/lib/api";

export default function ReportActions({ analysisId, videoUrl }) {
  if (!analysisId) return null;

  const jsonUrl = reportJsonUrl(analysisId);
  const csvUrl = reportCsvUrl(analysisId);
  const downloadUrl = videoUrl ? `${videoUrl}${videoUrl.includes("?") ? "&" : "?"}download=1` : "";

  return (
    <section className="flex flex-wrap gap-3">
      <a
        href={jsonUrl}
        target="_blank"
        rel="noreferrer"
        className="rounded-xl border border-cyan-400/30 px-4 py-2 text-xs font-semibold tracking-[0.16em] text-cyan-100 hover:bg-cyan-400/10"
      >
        DOWNLOAD JSON
      </a>
      <a
        href={csvUrl}
        className="rounded-xl border border-purple-400/30 px-4 py-2 text-xs font-semibold tracking-[0.16em] text-purple-100 hover:bg-purple-400/10"
      >
        DOWNLOAD CSV
      </a>
      {downloadUrl ? (
        <a
          href={downloadUrl}
          className="rounded-xl border border-cyan-400/30 px-4 py-2 text-xs font-semibold tracking-[0.16em] text-cyan-100 hover:bg-cyan-400/10"
        >
          DOWNLOAD ANALYZED VIDEO
        </a>
      ) : null}
    </section>
  );
}
