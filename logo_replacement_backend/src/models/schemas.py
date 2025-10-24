from __future__ import annotations

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class JobState(str, Enum):
    """Enumeration of possible job states in the processing lifecycle."""
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


# PUBLIC_INTERFACE
class JobCreated(BaseModel):
    """Represents the response returned after a job is created.

    Attributes:
        job_id: Unique identifier for the job.
        status: Initial status for the created job. Usually PENDING or UPLOADING.
        message: Optional human-friendly description for the job creation outcome.
    """
    job_id: str = Field(..., description="Unique identifier for the job.")
    status: JobState = Field(..., description="Initial status of the created job.")
    message: Optional[str] = Field(None, description="Optional message describing job creation result.")


# PUBLIC_INTERFACE
class DetectionBox(BaseModel):
    """Normalized or absolute detection box returned in status logs."""
    x: float = Field(..., description="Top-left X in pixels relative to full resolution.")
    y: float = Field(..., description="Top-left Y in pixels relative to full resolution.")
    width: float = Field(..., description="Box width in pixels at full resolution.")
    height: float = Field(..., description="Box height in pixels at full resolution.")
    confidence: float = Field(..., description="Confidence score 0..1.")
    method: Optional[str] = Field(None, description="Method used: vision|template|heuristic|ocr")
    page: Optional[int] = Field(None, description="Optional page index if source was a PDF page image.")
    dtype: Optional[str] = Field(None, description="Detection type: logo | text")


# PUBLIC_INTERFACE
class PerFileDetectionSummary(BaseModel):
    """Per-file detection result summary for a processed input."""
    file: str = Field(..., description="Relative path of the input image/page processed.")
    found: bool = Field(..., description="Whether any region was detected.")
    method: Optional[str] = Field(None, description="Primary method used for detection: vision|template|heuristic|ocr")
    boxes: List[DetectionBox] = Field(default_factory=list, description="List of detected boxes (pixel coords).")
    reason: Optional[str] = Field(None, description="Optional reason or note when not found.")
    replaced_count: int = Field(0, description="Number of regions replaced on this page.")
    replaced_logo_count: int = Field(0, description="Number of logo regions replaced.")
    replaced_text_count: int = Field(0, description="Number of text regions replaced.")


# PUBLIC_INTERFACE
class JobStatus(BaseModel):
    """Represents current status of a job.

    Attributes:
        job_id: Unique identifier for the job.
        status: Current state in the job lifecycle.
        progress: Percentage indicating overall job progress [0,100].
        message: Optional human-friendly status message.
        error: Optional error message if status is ERROR.
        result_url: Optional URL for downloading the result when available.
        detections: Optional list of per-file detection summaries.
    """
    job_id: str = Field(..., description="Unique identifier for the job.")
    status: JobState = Field(..., description="Current state of the job.")
    progress: int = Field(0, ge=0, le=100, description="Progress percentage of the job (0-100).")
    message: Optional[str] = Field(None, description="Optional message describing current state.")
    error: Optional[str] = Field(None, description="Optional error message when job fails.")
    result_url: Optional[str] = Field(None, description="Optional URL pointing to the result artifact when available.")
    detections: Optional[List[PerFileDetectionSummary]] = Field(
        default=None,
        description="Per-file detection summaries recorded during processing."
    )


# PUBLIC_INTERFACE
class ProcessedFile(BaseModel):
    """Represents a single processed output file."""
    filename: str = Field(..., description="Relative path/name of the processed file.")
    size: int = Field(..., description="Size in bytes.")
    content_type: str = Field(..., description="MIME type guess.")


# PUBLIC_INTERFACE
class ProcessedFileList(BaseModel):
    """List of processed output files."""
    items: List[ProcessedFile] = Field(default_factory=list, description="Collection of processed files.")
