from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    cv_text: str = Field(..., description="Texte du CV (idéalement déjà nettoyé)")
    job_text: str = Field(..., description="Texte de l'offre d'emploi")


class MatchResult(BaseModel):
    overall_score: float = Field(..., description="Score global de correspondance (0-100)")
    semantic_score: float = Field(
        ..., description="Score de similarité sémantique globale CV/offre (0-100)"
    )
    skill_coverage_score: float | None = Field(
        default=None,
        description=(
            "Pourcentage des compétences requises par l'offre trouvées dans le CV (0-100). "
            "None si aucune compétence technique n'a été détectée dans l'offre (auquel cas "
            "le score global repose entièrement sur la similarité sémantique)."
        ),
    )
    matched_skills: list[str] = Field(
        default_factory=list, description="Compétences requises par l'offre, trouvées dans le CV"
    )
    missing_skills: list[str] = Field(
        default_factory=list, description="Compétences requises par l'offre, absentes du CV"
    )