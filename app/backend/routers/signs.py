from fastapi import APIRouter, HTTPException

from models.schemas import SignListResponse, VideoResponse
from services.model_service import model_service
from services.signs_service import signs_service

router = APIRouter()


@router.get("/signs/{word}/reference-buckets")
async def get_reference_buckets(word: str) -> dict:
    buckets = model_service.get_reference_buckets(word.lower())
    if buckets is None:
        raise HTTPException(status_code=404, detail=f"No reference data for '{word}' yet.")
    return buckets


@router.get("/signs", response_model=SignListResponse)
async def get_signs() -> SignListResponse:
    return SignListResponse(signs=signs_service.get_all_signs())


@router.get("/signs-with-video")
async def get_signs_with_video() -> list[str]:
    return signs_service.get_signs_with_video()


@router.get("/video/{word}", response_model=VideoResponse)
async def get_video(word: str) -> VideoResponse:
    result = signs_service.get_video(word)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No video found for '{word}'")
    return result
