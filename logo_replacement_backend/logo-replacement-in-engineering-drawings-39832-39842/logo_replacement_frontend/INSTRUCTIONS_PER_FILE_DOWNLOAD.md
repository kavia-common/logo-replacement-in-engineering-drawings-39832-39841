# Step 3: Per-file downloads

- After the job status becomes COMPLETED, render a list of processed files alongside the ZIP download button.
- Use the new backend endpoints:
  - GET /jobs/{job_id}/files -> { items: [{ filename, size, content_type }] }
  - GET /jobs/{job_id}/files/{filename} streams the individual file with Content-Disposition header.

Example integration:

```jsx
import ProcessedFilesList from "./src/components/ProcessedFilesList";

function Step3({ apiBaseUrl, jobId, zipUrl }) {
  return (
    <div>
      <a href={zipUrl}>
        <button>Download ZIP</button>
      </a>
      <ProcessedFilesList apiBaseUrl={apiBaseUrl} jobId={jobId} className="mt-4" />
    </div>
  );
}
```

Notes:
- Ensure CORS exposes Content-Disposition header (backend already configured).
- The download handler extracts filename from Content-Disposition; if absent, it falls back to the path basename.
