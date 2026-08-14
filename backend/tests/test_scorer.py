from unittest.mock import patch

import pytest

from app.services.scorer import _compute_skill_coverage, compute_match_score


# --- _compute_skill_coverage (fonction interne, logique pure) ---


def test_coverage_full_match():
    coverage, matched, missing = _compute_skill_coverage(
        cv_skills=["Python", "Docker", "FastAPI"],
        required_skills=["Python", "Docker"],
    )
    assert coverage == 1.0
    assert matched == ["Docker", "Python"]
    assert missing == []


def test_coverage_partial_match():
    coverage, matched, missing = _compute_skill_coverage(
        cv_skills=["Python"],
        required_skills=["Python", "Docker", "Kubernetes"],
    )
    assert coverage == pytest.approx(1 / 3)
    assert matched == ["Python"]
    assert missing == ["Docker", "Kubernetes"]


def test_coverage_no_match():
    coverage, matched, missing = _compute_skill_coverage(
        cv_skills=["Java"],
        required_skills=["Python"],
    )
    assert coverage == 0.0
    assert matched == []
    assert missing == ["Python"]


def test_coverage_empty_required_skills_returns_full_coverage():
    """Aucune compétence requise détectée dans l'offre : on ne peut pas
    pénaliser le CV pour quelque chose qui n'est pas demandé."""
    coverage, matched, missing = _compute_skill_coverage(cv_skills=["Python"], required_skills=[])
    assert coverage == 1.0
    assert matched == []
    assert missing == []


# --- compute_match_score (mocké : on teste la pondération, pas le NLP) ---


@patch("app.services.scorer.extract_skills")
@patch("app.services.scorer.compute_semantic_similarity")
def test_overall_score_combines_semantic_and_coverage(mock_similarity, mock_extract):
    mock_similarity.return_value = 0.8  # 80% similarité sémantique
    mock_extract.side_effect = [
        ["Python", "Docker"],  # compétences du CV (1er appel)
        ["Python", "Docker"],  # compétences requises par l'offre (2e appel) -> 100% couverture
    ]

    result = compute_match_score("texte cv", "texte offre")

    # overall = 0.4 * 80 + 0.6 * 100 = 92.0
    assert result.overall_score == 92.0
    assert result.semantic_score == 80.0
    assert result.skill_coverage_score == 100.0


@patch("app.services.scorer.extract_skills")
@patch("app.services.scorer.compute_semantic_similarity")
def test_matched_and_missing_skills_populated(mock_similarity, mock_extract):
    mock_similarity.return_value = 0.5
    mock_extract.side_effect = [
        ["Python"],  # CV
        ["Python", "Docker", "AWS"],  # offre
    ]

    result = compute_match_score("texte cv", "texte offre")

    assert result.matched_skills == ["Python"]
    assert result.missing_skills == ["AWS", "Docker"]


@patch("app.services.scorer.extract_skills")
@patch("app.services.scorer.compute_semantic_similarity")
def test_zero_semantic_and_zero_coverage_gives_zero_overall(mock_similarity, mock_extract):
    mock_similarity.return_value = 0.0
    mock_extract.side_effect = [[], ["Python"]]  # CV sans compétences, offre en demande une

    result = compute_match_score("texte cv", "texte offre")

    assert result.overall_score == 0.0


# --- Test d'intégration (PAS mocké) ---


def test_real_match_between_similar_cv_and_job():
    """Vérifie le câblage réel entre embeddings.py et extractor.py, sur
    un cas simple où le résultat attendu est sans ambiguïté."""
    cv_text = "Développeur Python avec 5 ans d'expérience en FastAPI et Docker."
    job_text = "Recherche développeur Python, maîtrise de FastAPI et Docker requise."

    result = compute_match_score(cv_text, job_text)

    assert result.overall_score > 70
    assert "Python" in result.matched_skills
    assert result.missing_skills == []