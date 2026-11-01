from fastapi import APIRouter

from models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from services.model_service import model_service

    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
    )
