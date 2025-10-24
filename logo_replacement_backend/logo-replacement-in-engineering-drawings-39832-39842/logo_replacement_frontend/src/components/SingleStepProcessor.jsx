import React, { useEffect, useRef, useState } from "react";
import ProcessedFilesList from "./ProcessedFilesList";

/**
 * PUBLIC_INTERFACE
 * SingleStepProcessor
 * A simple UI that posts a single multipart /process request (drawings_zip or drawings_files[] plus logo_image),
 * polls /status/{job_id}, and shows progress and results.
 * Props:
 *  - apiBaseUrl: string backend base URL (e.g., http://localhost:3001)
 */
export default function SingleStepProcessor({ apiBaseUrl }) {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  // Form refs
  const drawingsZipRef = useRef(null);
  const drawingsFilesRef = useRef(null);
  const logoRef = useRef(null);

  async function startProcess(e) {
    e.preventDefault();
    setError(null);
    setJobId(null);
    setStatus(null);

    const fd = new FormData();
    const logo = logoRef.current?.files?.[0];
    if (!logo) {
      setError("Please select a logo image.");
      return;
    }
    fd.append("logo_image", logo);

    const zip = drawingsZipRef.current?.files?.[0];
    const files = drawingsFilesRef.current?.files;
    if (!zip && (!files || files.length === 0)) {
      setError("Please select a drawings ZIP or one or more drawing files.");
      return;
    }
    if (zip) {
      fd.append("drawings_zip", zip);
    }
    if (files && files.length > 0) {
      for (const f of files) {
        fd.append("drawings_files", f);
      }
    }

    try {
      const res = await fetch(`${apiBaseUrl}/process`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail || `Failed to start process (${res.status})`);
      }
      const data = await res.json();
      setJobId(data.job_id);
    } catch (err) {
      setError(err.message);
    }
  }

  // Poll status when jobId is set
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    async function poll() {
      try {
        const res = await fetch(`${apiBaseUrl}/status/${jobId}`);
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail?.detail || `Failed to fetch status (${res.status})`);
        }
        const data = await res.json();
        if (!cancelled) {
          setStatus(data);
          if (data?.status === "COMPLETED" || data?.status === "ERROR" || data?.status === "CANCELLED") {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    }

    poll();
    pollRef.current = setInterval(poll, 1500);
    return () => {
      cancelled = true;
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [apiBaseUrl, jobId]);

  const progress = typeof status?.progress === "number" ? status.progress : 0;
  const zipUrl = status?.result_url || (jobId ? `${apiBaseUrl}/download/${jobId}` : null);

  return (
    <div style={{ background: "#FFFFFF", border: "1px solid #E5E7EB", borderRadius: 8, padding: 16 }}>
      <h3 style={{ marginTop: 0, marginBottom: 12, color: "#111827" }}>Logo Replacement Processor</h3>
      <form onSubmit={startProcess} style={{ display: "grid", gap: 12 }}>
        <div>
          <label style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>Drawings ZIP (images and PDFs)</label>
          <input ref={drawingsZipRef} type="file" accept=".zip" />
          <div style={{ color: "#6B7280", fontSize: 12, marginTop: 4 }}>
            Upload a ZIP containing PNG, JPG/JPEG, TIFF, BMP, GIF, or PDF drawings.
          </div>
        </div>
        <div>
          <label style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>Or individual drawing files</label>
          <input ref={drawingsFilesRef} type="file" multiple accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.gif,.pdf" />
        </div>
        <div>
          <label style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>New logo image</label>
          <input ref={logoRef} type="file" accept="image/*" />
        </div>
        <div>
          <button
            type="submit"
            style={{
              backgroundColor: "#374151",
              color: "#FFFFFF",
              border: "none",
              padding: "8px 16px",
              borderRadius: 6,
              cursor: "pointer",
            }}
          >
            Start Processing
          </button>
        </div>
      </form>

      {error && <div style={{ marginTop: 12, color: "#DC2626" }}>Error: {error}</div>}

      {jobId && (
        <div style={{ marginTop: 16 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Job: {jobId}</div>
          <div
            style={{
              height: 10,
              background: "#E5E7EB",
              borderRadius: 6,
              overflow: "hidden",
              marginBottom: 8,
            }}
          >
            <div
              style={{
                width: `${progress}%`,
                height: "100%",
                background: "#059669",
                transition: "width 300ms ease",
              }}
            />
          </div>
          <div style={{ color: "#6B7280", fontSize: 14 }}>{status?.message || "Processing..."}</div>

          {status?.status === "COMPLETED" && (
            <div style={{ marginTop: 12 }}>
              <a href={zipUrl}>
                <button
                  style={{
                    backgroundColor: "#374151",
                    color: "#FFFFFF",
                    border: "none",
                    padding: "8px 16px",
                    borderRadius: 6,
                    cursor: "pointer",
                  }}
                >
                  Download ZIP
                </button>
              </a>
              <ProcessedFilesList apiBaseUrl={apiBaseUrl} jobId={jobId} className="mt-4" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
