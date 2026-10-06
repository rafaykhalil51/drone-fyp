"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnalysisResults from "@/components/AnalysisResults";
import Header from "@/components/Header";
import StatCard from "@/components/StatCard";
import VideoUploader from "@/components/VideoUploader";
import { getHealth, getModelStatus } from "@/lib/api";

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

  function onComplete(analysis) {
    setResult(analysis);
    setSummary({
      total_unique_persons: analysis?.summary?.total_unique_persons ?? null,
      wearing_cap: analysis?.summary?.wearing_cap ?? null,
      wearing_mask: analysis?.summary?.wearing_mask ?? null,
      wearing_glasses: analysis?.summary?.wearing_glasses ?? null,
      wearing_headphones: analysis?.summary?.wearing_headphones ?? null,
      plain: analysis?.summary?.plain ?? null,
    });
  }

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
        <VideoUploader backendOnline={backendOnline} onComplete={onComplete} />
        <AnalysisResults result={result} />
      </section>

      <footer className="flex flex-col gap-1 border-t border-white/10 pt-4 text-[11px] tracking-[0.16em] text-slate-500 sm:flex-row sm:justify-between">
        <span>A.E.G.I.S // VISION INTELLIGENCE</span>
        <span>FINAL YEAR PROJECT</span>
      </footer>
    </main>
  );
}
