const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

export function apiBase() {
  return API_URL.replace(/\/$/, "");
}

function endpoint(path) {
  const base = apiBase();
  if (!base) {
    throw new Error("NEXT_PUBLIC_API_URL is not set.");
  }
  return `${base}${path}`;
}

async function readBody(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function errorMessage(status, body) {
  if (body && typeof body === "object" && typeof body.detail === "string") {
    return body.detail;
  }
  if (status === 415) return "Unsupported file. Use MP4, MOV, AVI, or MKV.";
  if (status === 503) return "A required model is unavailable.";
  if (status >= 500) return "Analysis failed. The backend could not finish this video.";
  if (status >= 400) return "The backend rejected this request.";
  return "The backend returned an unexpected response.";
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(endpoint(path), options);
  } catch (error) {
    console.error("Network error:", error);
    const offline = new Error("Backend offline. Check that the API is running.");
    offline.code = "offline";
    throw offline;
  }

  const body = await readBody(response);
  if (!response.ok) {
    const failure = new Error(errorMessage(response.status, body));
    failure.status = response.status;
    failure.code = response.status === 415 ? "unsupported" : response.status === 503 ? "model" : "request";
    console.error("API error:", response.status, body);
    throw failure;
  }
  return body;
}

export function getHealth() {
  return request("/api/health");
}

export function getModelStatus() {
  return request("/api/models/status");
}

export function analyzeVideo(file) {
  const body = new FormData();
  body.append("video", file);
  return request("/api/analyze/video", { method: "POST", body });
}

export function reportJsonUrl(analysisId) {
  return endpoint(`/api/reports/${encodeURIComponent(analysisId)}/json`);
}

export function reportCsvUrl(analysisId) {
  return endpoint(`/api/reports/${encodeURIComponent(analysisId)}/csv`);
}
