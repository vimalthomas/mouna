from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Optional: when set, Claude writes the post-attempt coaching message
    # (services/coach_writer.py); without it, plain-language templates are used.
    # Demo / guest mode: requests without a login are served as one shared
    # guest account (no sign-in or sign-up needed). Off unless set in .env.
    allow_guest: bool = False
    # The /ws/debug route runs the models on any frames sent to it, without a
    # login. Fine locally; turned off in the public container image.
    enable_debug_routes: bool = True
    anthropic_api_key: str = ""
    coach_llm_model: str = "claude-opus-5-5"
    # Optional alternative when no Claude key is set: Google Gemini writes the
    # same messages from the same facts (services/coach_writer.py).
    gemini_api_key: str = ""
    # Tried in order; the next is used when one is busy (503) or retired (404).
    # "-latest" aliases follow Google's current models.
    gemini_model: str = "gemini-flash-latest,gemini-flash-lite-latest"
    model_checkpoint_path: str = "./checkpoints/pose_transformer_best.pt"
    signs_json_path: str = "./data/signs.json"
    wlasl_json_path: str = "../wlasl/WLASL_v0.3.json"
    videos_dir: str = "../wlasl/videos"
    mediapipe_model_dir: str = "/app/mediapipe_models"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:4200"
    # Optional pattern of extra allowed origins, e.g. https://.*\.vercel\.app for a
    # Vercel-hosted frontend (production and preview deployments). Empty = off.
    cors_origin_regex: str = ""
    confidence_threshold: float = 0.25
    max_seq_len: int = 150
    num_classes: int = 2000
    feature_dim: int = 225

    database_url: str = "postgresql+psycopg2://mouna:mouna@localhost:5432/mouna"
    jwt_secret_key: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14
    min_attempts_for_completion: int = 5
    completion_accuracy_threshold: float = 0.70

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
