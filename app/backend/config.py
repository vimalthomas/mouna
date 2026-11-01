from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    model_checkpoint_path: str = "./checkpoints/pose_transformer_best.pt"
    signs_json_path: str = "./data/signs.json"
    wlasl_json_path: str = "../wlasl/WLASL_v0.3.json"
    videos_dir: str = "../wlasl/videos"
    mediapipe_model_dir: str = "/app/mediapipe_models"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:4200"
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_coaching_enabled: bool = True
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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
