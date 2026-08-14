from fastapi import APIRouter

from app.models.match import MatchRequest, MatchResult
from app.services.scorer import compute_match_score

router = APIRouter()


@router.post(
    "",
    response_model=MatchResult,
    summary="Calcule le score de correspondance entre un CV et une offre d'emploi",
)
async def match_cv_to_job(request: MatchRequest) -> MatchResult:
    return compute_match_score(request.cv_text, request.job_text)