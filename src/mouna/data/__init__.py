"""Data ingestion and preprocessing modules."""

__all__ = [
    "KeypointExtractor",
    "VideoPreprocessor",
    "SignLanguageDataset",
]


def __getattr__(name):
    if name in ("KeypointExtractor", "VideoPreprocessor"):
        from mouna.data.preprocessing import KeypointExtractor, VideoPreprocessor
        globals()["KeypointExtractor"] = KeypointExtractor
        globals()["VideoPreprocessor"] = VideoPreprocessor
        return globals()[name]
    if name == "SignLanguageDataset":
        from mouna.data.dataset import SignLanguageDataset
        return SignLanguageDataset
    raise AttributeError(f"module 'mouna.data' has no attribute {name!r}")
