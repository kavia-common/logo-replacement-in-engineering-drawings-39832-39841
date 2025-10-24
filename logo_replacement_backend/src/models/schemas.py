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
class JobStatus(BaseModel):
    """Represents current status of a job.

    Attributes:
        job_id: Unique identifier for the job.
        status: Current state in the job lifecycle.
        progress: Percentage indicating overall job progress [0,100].
        message: Optional human-friendly status message.
        error: Optional error message if status is ERROR.
        result_url: Optional URL for downloading the result when available.
    """
    job_id: str = Field(..., description="Unique identifier for the job.")
    status: JobState = Field(..., description="Current state of the job.")
    progress: int = Field(0, ge=0, le=100, description="Progress percentage of the job (0-100).")
    message: Optional[str] = Field(None, description="Optional message describing current state.")
    error: Optional[str] = Field(None, description="Optional error message when job fails.")
    result_url: Optional[str] = Field(None, description="Optional URL pointing to the result artifact when available.")


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
