"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnalysisResults from "@/components/AnalysisResults";
import Header from "@/components/Header";
import StatCard from "@/components/StatCard";
import VideoUploader from "@/components/VideoUploader";
import { absoluteApiUrl, analysisStreamUrl, getAnalysisStatus, getHealth, getModelStatus } from "@/lib/api";

const ANALYSIS_KEY = "aegis.analysisId";

const INITIAL_SUMMARY = {
  total_unique_persons: 0,
  wearing_cap: 0,
  wearing_mask: 0,
  wearing_glasses: 0,
  wearing_headphones: 0,
  plain: 0,
};

export default function Home() {
  const [backendOnline, setBackendOnline] = useState(null);
  const [personActive, setPersonActive] = useState(null);
  const [accessoryActive, setAccessoryActive] = useState(null);
  const [summary, setSummary] = useState(INITIAL_SUMMARY);
  const [result, setResult] = useState(null);
  const [statusNote, setStatusNote] = useState("");
  const [analysisId, setAnalysisId] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [streamUrl, setStreamUrl] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [videoDetail, setVideoDetail] = useState("");
  const [liveTracks, setLiveTracks] = useState(null);
  const [progressText, setProgressText] = useState("");
  const statusInFlight = useRef(false);

  const refreshStatus = useCallback(async () => {
    if (statusInFlight.current) return;
    statusInFlight.current = true;
    try {
      const health = await getHealth();
      setBackendOnline(health?.status === "online");
    } catch (error) {
      console.error("Health error:", error);
      setBackendOnline(false);
    }

    try {
      const models = await getModelStatus();
      setPersonActive(models?.person?.status === "active");
      setAccessoryActive(models?.accessory?.status === "active");
      if (models?.person?.status !== "active") {
        setStatusNote(models?.person?.detail || "Person model is unavailable.");
      } else if (models?.accessory?.status !== "active") {
        setStatusNote(models?.accessory?.detail || "Accessory model is unavailable. Person tracking can still run.");
      } else {
        setStatusNote("");
      }
    } catch (error) {
      console.error("Model error:", error);
      setPersonActive(false);
      setAccessoryActive(false);
    } finally {
      statusInFlight.current = false;
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    const timer = setInterval(refreshStatus, 20000);
    return () => clearInterval(timer);
  }, [refreshStatus]);

  function applySummary(next) {
    setSummary({
      total_unique_persons: next?.total_unique_persons ?? null,
      wearing_cap: next?.wearing_cap ?? null,
      wearing_mask: next?.wearing_mask ?? null,
      wearing_glasses: next?.wearing_glasses ?? null,
      wearing_headphones: next?.wearing_headphones ?? null,
      plain: next?.plain ?? null,
    });
  }

  function showCompleted(id, status) {
    setAnalysisId(id);
    setResult(status.result);
    applySummary(status.result.summary);
    setLiveTracks(null);
    setStreamUrl("");
    setVideoUrl(status.video_ready ? absoluteApiUrl(status.video_url) : "");
    setVideoDetail(status.video_detail || "");
    setProcessing(false);
    setProgressText("");
    window.sessionStorage.setItem(ANALYSIS_KEY, id);
  }

  function onStart(started) {
    const id = started?.analysis_id;
    if (!id) return;
    setAnalysisId(id);
    setProcessing(true);
    setResult(null);
    setLiveTracks(null);
    setStreamUrl(analysisStreamUrl(id));
    setVideoUrl("");
    setVideoDetail("");
    applySummary(INITIAL_SUMMARY);
    setProgressText("Waiting for the first annotated frame.");
    setStatusNote("");
    window.sessionStorage.setItem(ANALYSIS_KEY, id);
  }

  useEffect(() => {
    const saved = window.sessionStorage.getItem(ANALYSIS_KEY);
    if (!saved) return undefined;
    let stopped = false;

    getAnalysisStatus(saved)
      .then((status) => {
        if (stopped) return;
        if (status.status === "completed" && status.result) {
          showCompleted(saved, status);
          return;
        }
        if (status.status === "processing") {
          setAnalysisId(saved);
          setProcessing(true);
          setStreamUrl(analysisStreamUrl(saved));
          setVideoUrl("");
          if (status.summary) applySummary(status.summary);
          if (Array.isArray(status.tracks)) setLiveTracks(status.tracks);
        }
      })
      .catch(() => {
        window.sessionStorage.removeItem(ANALYSIS_KEY);
      });

    return () => {
      stopped = true;
    };
  }, []);

  useEffect(() => {
    if (!analysisId || !processing) return undefined;
    let stopped = false;

    async function poll() {
      try {
        const status = await getAnalysisStatus(analysisId);
        if (stopped) return;
        if (status.summary) applySummary(status.summary);
        if (Array.isArray(status.tracks)) setLiveTracks(status.tracks);
        if (status.total_frames) {
          setProgressText(`Frame ${status.current_frame || 0} of ${status.total_frames}.`);
        } else if (status.current_frame) {
          setProgressText(`Frame ${status.current_frame}.`);
        }
        if (status.status === "completed" && status.result) {
          showCompleted(analysisId, status);
        } else if (status.status === "failed") {
          setProcessing(false);
          setProgressText("");
          setStatusNote(status.detail || "Analysis failed.");
        }
      } catch (error) {
        console.error("Live status error:", error);
      }
    }

    poll();
    const timer = setInterval(poll, 800);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [analysisId, processing]);

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
      <Header
        personActive={personActive}
        accessoryActive={accessoryActive}
        backendOnline={backendOnline}
      />

      {statusNote ? (
        <p className="rounded-xl border border-amber-400/25 bg-amber-400/10 px-4 py-3 text-sm text-amber-100">
          {statusNote}
        </p>
      ) : null}

      <section className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatCard kicker="TARGETS" label="Tracked Persons" value={summary.total_unique_persons} accent="cyan" />
        <StatCard kicker="CRANIAL" label="Caps" value={summary.wearing_cap} accent="amber" />
        <StatCard kicker="RESPIRATORY" label="Masks" value={summary.wearing_mask} accent="orange" />
        <StatCard kicker="OPTICAL" label="Glasses" value={summary.wearing_glasses} accent="purple" />
        <StatCard kicker="ACOUSTIC" label="Headphones" value={summary.wearing_headphones} accent="cyan" />
        <StatCard kicker="PLAIN" label="No detected accessories" value={summary.plain} accent="slate" />
      </section>

      <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <VideoUploader
          backendOnline={backendOnline}
          processing={processing}
          progressText={progressText}
          onStart={onStart}
          completed={Boolean(result)}
        />
        <AnalysisResults
          result={result}
          streamUrl={processing ? streamUrl : ""}
          videoUrl={processing ? "" : videoUrl}
          videoDetail={videoDetail}
          liveTracks={processing ? liveTracks : null}
        />
      </section>

      <footer className="flex flex-col gap-1 border-t border-white/10 pt-4 text-[11px] tracking-[0.16em] text-slate-500 sm:flex-row sm:justify-between">
        <span>A.E.G.I.S // VISION INTELLIGENCE</span>
        <span>FINAL YEAR PROJECT</span>
      </footer>
    </main>
  );
}
