from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class ImageResult(BaseModel):
    image_name: str
    has_bright_spot: bool
    detections: List[DetectionBox]
    processed: bool = False
    inference_time_seconds: Optional[float] = None


class BatchInfo(BaseModel):
    batch_id: int
    status: BatchStatus
    images_processed: int
    total_images: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class RunStats(BaseModel):
    total_images: int = 0
    images_with_bright_spot: int = 0
    images_without_bright_spot: int = 0
    success_rate: float = 0.0  # Percentage of images WITHOUT bright spot (good modules)
    bright_spot_rate: float = 0.0  # Percentage of images WITH bright spot (defects)
    batches_completed: int = 0
    batches_total: int = 0
    total_detections: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    median_inference_time_seconds: Optional[float] = None
    images_processed_count: int = 0


class RunData(BaseModel):
    run_id: str
    run_name: str
    timestamp: datetime
    model_path: str
    input_dir: str
    output_dir: str
    confidence: float
    batch_size: int
    status: RunStatus
    stats: RunStats
    batches: List[BatchInfo] = []
    images_processed: List[str] = []


class StartRequest(BaseModel):
    model_path: str
    input_dir: str
    output_dir: str
    confidence: float
    run_name: str
    batch_size: int = 100


class StatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    progress: float  # 0.0 to 1.0
    current_batch: int
    total_batches: int
    stats: RunStats


class CombinedStatsRequest(BaseModel):
    run_ids: List[str]


class CombinedStatsResponse(BaseModel):
    total_runs: int
    total_images_processed: int
    total_bright_spots_found: int
    total_images_without_bright_spot: int
    average_success_rate: float
    average_bright_spot_rate: float
    runs: List[Dict[str, Any]]

