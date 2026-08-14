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

    Précondition : required_skills ne doit pas être vide — c'est à
    l'appelant (compute_match_score) de gérer ce cas en amont, car il
    n'a pas de réponse "correcte" au niveau de cette fonction (voir
    compute_match_score pour le raisonnement).

    Returns:
        Tuple (taux de couverture entre 0 et 1, compétences trouvées,
        compétences manquantes) — trié pour un affichage stable.
    """
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

    Cas particulier important : si l'offre ne contient aucune
    compétence détectable par la taxonomie (ex: une offre orientée
    business/stratégie plutôt que technique), on ne peut PAS calculer
    de couverture de compétences significative. Dans ce cas, le score
    global repose entièrement sur la similarité sémantique — un
    fallback artificiel à 100% de couverture gonflerait le score de
    façon trompeuse (constaté concrètement : un CV Machine Learning
    Engineer face à une offre "Product Builder" non-technique donnait
    75.8% au lieu des ~39% reflétant le vrai désalignement du profil).
    """
    semantic_score = compute_semantic_similarity(cv_text, job_text)

    cv_skills = extract_skills(cv_text)
    required_skills = extract_skills(job_text)

    if not required_skills:
        return MatchResult(
            overall_score=round(semantic_score * 100, 1),
            semantic_score=round(semantic_score * 100, 1),
            skill_coverage_score=None,
            matched_skills=[],
            missing_skills=[],
        )

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