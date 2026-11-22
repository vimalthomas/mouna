from pydantic import BaseModel


class SignInfo(BaseModel):
    word: str
    category: str
    difficulty: str
    description: str
    manual_checks: list[str] = []
    nms_required: bool = False
    nms_type: str | None = None
    nms_notes: str | None = None


class SignListResponse(BaseModel):
    signs: list[SignInfo]


class VideoResponse(BaseModel):
    word: str
    video_url: str
    frame_start: int | None = None
    frame_end: int | None = None
    fps: int | None = None


class Prediction(BaseModel):
    sign: str
    confidence: float


class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    top3: list[Prediction]
    uncertain: bool
    inference_time_ms: float | None = None


class CoachingFeedback(BaseModel):
    prediction: str
    confidence: float
    correct: bool
    feedback: str
    tips: list[str]
    attempt_number: int
    session_accuracy: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
