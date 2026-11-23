"""Model architectures for sign language recognition."""

from mouna.models.baseline import KeypointBiLSTM, KeypointGRU
# The video model needs Hugging Face `transformers`, which the keypoint app
# (and its server image) doesn't install -- import it only when available.
try:
    from mouna.models.multimodal import MultimodalSignRecognition
except ImportError:  # pragma: no cover
    MultimodalSignRecognition = None  # type: ignore[assignment]
from mouna.models.fusion import LateFusion, CrossAttentionFusion

__all__ = [
    "KeypointBiLSTM",
    "KeypointGRU",
    "MultimodalSignRecognition",
    "LateFusion",
    "CrossAttentionFusion",
]
