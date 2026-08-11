from app.services.extractor import extract_skills


def test_extracts_known_skill_exact_case():
    assert "Python" in extract_skills("Je maîtrise Python depuis 5 ans.")


def test_extracts_skill_case_insensitive_lowercase():
    assert "Docker" in extract_skills("Compétences : docker, kubernetes")


def test_extracts_skill_case_insensitive_uppercase():
    assert "SQL" in extract_skills("Bases de données : SQL, POSTGRESQL")


def test_normalizes_to_canonical_form():
    """Peu importe la casse du texte source, la sortie utilise toujours
    l'orthographe de référence définie dans la taxonomie."""
    result = extract_skills("javascript et TYPESCRIPT")
    assert "JavaScript" in result
    assert "TypeScript" in result
    assert "javascript" not in result
    assert "TYPESCRIPT" not in result


def test_extracts_multiple_distinct_skills():
    text = "Stack : Python, FastAPI, Docker, PostgreSQL, Git"
    result = extract_skills(text)
    assert result == ["Docker", "FastAPI", "Git", "PostgreSQL", "Python"]


def test_deduplicates_repeated_skill():
    text = "Python, python, PYTHON - expert Python"
    result = extract_skills(text)
    assert result.count("Python") == 1


def test_extracts_multiword_skill():
    text = "Formation en Machine Learning et GitLab CI"
    result = extract_skills(text)
    assert "Machine Learning" in result
    assert "GitLab CI" in result


def test_no_false_positive_on_unrelated_text():
    text = "Le chat mange une pomme dans le jardin."
    assert extract_skills(text) == []


def test_empty_string_returns_empty_list():
    assert extract_skills("") == []


def test_real_cv_skill_section():
    """Extrait réel de la section compétences d'un vrai CV (voir
    conversation projet) : vérifie qu'on retrouve les compétences clés
    dans un texte réaliste, pas juste des cas synthétiques isolés."""
    text = (
        "Python, Pandas, Numpy, Scikit-learn PyTorch (ou TensorFlow) "
        "FastAPI, Flask Docker, Git, GitLab CI SQL, PostgreSQL "
        "MLflow, DVC, Airflow GCP / AWS"
    )
    result = extract_skills(text)
    expected_subset = {
        "Python", "Pandas", "NumPy", "Scikit-learn", "PyTorch",
        "TensorFlow", "FastAPI", "Flask", "Docker", "Git", "GitLab CI",
        "SQL", "PostgreSQL", "MLflow", "DVC", "Airflow", "GCP", "AWS",
    }
    assert expected_subset.issubset(set(result))