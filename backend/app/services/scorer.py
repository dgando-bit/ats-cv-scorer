"""
Service de scoring : combine la similarité sémantique globale et la
couverture des compétences requises en un score composite explicable,
plutôt qu'un score "boîte noire" difficile à justifier.
"""

from __future__ import annotations

from app.models.match import MatchResult
from app.services.embeddings import compute_semantic_similarity
from app.services.extractor import extract_skills

# Pondération du score composite : la couverture des compétences pèse
# plus lourd que la similarité sémantique globale, car c'est un signal
# plus direct et plus facile à justifier auprès d'un utilisateur final
# ("il vous manque Docker et Kubernetes") qu'un score sémantique brut.
_SEMANTIC_WEIGHT = 0.4
_SKILL_COVERAGE_WEIGHT = 0.6


def _compute_skill_coverage(
    cv_skills: list[str], required_skills: list[str]
) -> tuple[float, list[str], list[str]]:
    """
    Calcule la proportion des compétences requises par l'offre que le
    CV possède effectivement.

    Returns:
        Tuple (taux de couverture entre 0 et 1, compétences trouvées,
        compétences manquantes) — trié pour un affichage stable.
    """
    if not required_skills:
        # Aucune compétence détectée dans l'offre : on ne peut pas
        # pénaliser le CV pour quelque chose qui n'est pas demandé.
        return 1.0, [], []

    cv_skills_set = set(cv_skills)
    required_set = set(required_skills)

    matched = sorted(required_set & cv_skills_set)
    missing = sorted(required_set - cv_skills_set)

    coverage = len(matched) / len(required_set)
    return coverage, matched, missing


def compute_match_score(cv_text: str, job_text: str) -> MatchResult:
    """
    Calcule le score de correspondance entre un CV et une offre d'emploi.

    Réutilise deux services déjà construits et testés indépendamment
    (embeddings.py, extractor.py) plutôt que de dupliquer leur logique
    ici — ce module se contente de les combiner et de pondérer.
    """
    semantic_score = compute_semantic_similarity(cv_text, job_text)

    cv_skills = extract_skills(cv_text)
    required_skills = extract_skills(job_text)
    skill_coverage, matched_skills, missing_skills = _compute_skill_coverage(
        cv_skills, required_skills
    )

    overall_score = _SEMANTIC_WEIGHT * semantic_score + _SKILL_COVERAGE_WEIGHT * skill_coverage

    return MatchResult(
        overall_score=round(overall_score * 100, 1),
        semantic_score=round(semantic_score * 100, 1),
        skill_coverage_score=round(skill_coverage * 100, 1),
        matched_skills=matched_skills,
        missing_skills=missing_skills,
    )