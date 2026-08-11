from app.services.extractor import extract_email, extract_phone, extract_skills


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


# --- extract_email ---


def test_extracts_simple_email():
    assert extract_email("Contact : jean.dupont@gmail.com") == "jean.dupont@gmail.com"


def test_extracts_email_with_dots_and_plus():
    text = "Écrivez-moi à d.gbakary+cv@outlook.com pour toute question."
    assert extract_email(text) == "d.gbakary+cv@outlook.com"


def test_returns_none_when_no_email():
    assert extract_email("Aucune adresse ici, juste du texte.") is None


def test_extracts_first_email_when_multiple():
    text = "Pro: contact@entreprise.com Perso: jean@gmail.com"
    assert extract_email(text) == "contact@entreprise.com"


def test_does_not_match_malformed_email():
    """Pas de @ ou pas de domaine valide : ne doit rien matcher."""
    assert extract_email("Suivez-moi sur Twitter @jean_dupont") is None


# --- extract_phone ---


def test_extracts_phone_with_international_prefix_and_spaces():
    assert extract_phone("+33 6 70 50 41 98") == "+33 6 70 50 41 98"


def test_extracts_phone_with_leading_zero_and_spaces():
    assert extract_phone("06 70 50 41 98") == "06 70 50 41 98"


def test_extracts_phone_without_separators():
    assert extract_phone("0670504198") == "0670504198"


def test_extracts_phone_with_dots():
    assert extract_phone("06.70.50.41.98") == "06.70.50.41.98"


def test_extracts_phone_with_dashes():
    assert extract_phone("06-70-50-41-98") == "06-70-50-41-98"


def test_returns_none_when_no_phone():
    assert extract_phone("Pas de numéro dans ce texte.") is None


def test_extracts_phone_from_real_cv_contact_section():
    text = "Contact\n+33 6 70 50 41 98\nd.gbakary@outlook.com\nThiais, France"
    assert extract_phone(text) == "+33 6 70 50 41 98"
    assert extract_email(text) == "d.gbakary@outlook.com"