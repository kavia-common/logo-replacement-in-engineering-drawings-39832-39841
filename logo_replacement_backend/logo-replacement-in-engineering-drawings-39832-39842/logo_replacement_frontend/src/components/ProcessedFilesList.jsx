import React, { useEffect, useState } from "react";

/**
 * PUBLIC_INTERFACE
 * ProcessedFilesList
 * This component shows per-file download buttons for a completed job.
 * Props:
 *  - apiBaseUrl: string backend base URL (e.g., http://localhost:3001)
 *  - jobId: string job identifier
 *  - className?: optional css classes
 */
export default function ProcessedFilesList({ apiBaseUrl, jobId, className = "" }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchFiles() {
      setLoading(true);
      setErr(null);
      try {
        const res = await fetch(`${apiBaseUrl}/jobs/${jobId}/files`);
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail?.detail || `Failed to fetch files (${res.status})`);
        }
        const data = await res.json();
        if (!cancelled) {
          setFiles(Array.isArray(data?.items) ? data.items : []);
        }
      } catch (e) {
        if (!cancelled) setErr(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    if (jobId) fetchFiles();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, jobId]);

  async function downloadFile(relPath) {
    const url = `${apiBaseUrl}/jobs/${jobId}/files/${encodeURIComponent(relPath)}`;
    const res = await fetch(url);
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail?.detail || `Failed to download file (${res.status})`);
    }
    // Extract filename from Content-Disposition
    const cd = res.headers.get("Content-Disposition");
    let filename = relPath.split("/").pop() || "file";
    if (cd) {
      const match = /filename\*=UTF-8''([^;]+)|filename="([^"]+)"|filename=([^;]+)/i.exec(cd);
      if (match) {
        filename = decodeURIComponent(match[1] || match[2] || match[3]).trim();
      }
    }
    const blob = await res.blob();
    const link = document.createElement("a");
    const blobUrl = URL.createObjectURL(blob);
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(blobUrl);
  }

  if (loading) return <div className={className}>Loading files…</div>;
  if (err) return <div className={className} style={{ color: "#DC2626" }}>Error: {err}</div>;
  if (!files.length) return <div className={className}>No individual files found.</div>;

  return (
    <div className={className}>
      <h4 style={{ marginBottom: 8 }}>Processed files</h4>
      <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {files.map((f) => (
          <li key={f.filename} style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ flex: "1 1 auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {f.filename}
            </span>
            <button
              onClick={() => downloadFile(f.filename)}
              style={{
                backgroundColor: "#374151",
                color: "#FFFFFF",
                border: "none",
                padding: "6px 12px",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              Download
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
