import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from db.session import SessionLocal
from db.signs_repo import sync_signs
from routers import auth, coach_frames, debug, health, signs, users
from services.keypoint_service import keypoint_service
from services.model_service import model_service
from services.signs_service import signs_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Mouna backend...")
    model_service.load_model()
    keypoint_service.initialize()
    signs_service.load()

    with SessionLocal() as db:
        sync_signs(db, list(model_service.idx_to_label.values()))

    yield
    logger.info("Shutting down Mouna backend...")


app = FastAPI(
    title="Mouna - Sign Language Recognition API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(signs.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(coach_frames.router)
app.include_router(debug.router)

videos_dir = Path(settings.videos_dir)
if videos_dir.exists():
    app.mount("/static/videos", StaticFiles(directory=str(videos_dir)), name="videos")
    logger.info("Mounted video directory: %s", videos_dir)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
