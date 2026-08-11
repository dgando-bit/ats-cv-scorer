from unittest.mock import patch

from app.services.extractor import extract_dates, extract_email, extract_phone, extract_skills


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


# --- extract_dates ---
#
# Contrairement aux tests précédents (100% déterministes), extract_dates()
# dépend d'un modèle BERT externe. On sépare deux familles de tests :
#   - tests "logique" (mockés, rapides, déterministes) : vérifient le
#     comportement de extract_dates() (filtrage, ordre, troncature)
#     indépendamment de ce que le modèle détecte réellement
#   - un test "intégration" (le vrai modèle, plus lent) : vérifie qu'un
#     cas concret déjà validé manuellement fonctionne toujours


def _fake_entity(word: str, score: float, entity_group: str = "DATE") -> dict:
    """Construit un faux résultat au format renvoyé par le pipeline
    transformers, pour simuler la sortie du modèle sans l'exécuter."""
    return {"word": word, "score": score, "entity_group": entity_group}


@patch("app.services.extractor._get_date_ner_pipeline")
def test_filters_out_low_confidence_results(mock_get_pipeline):
    mock_pipeline = mock_get_pipeline.return_value
    mock_pipeline.return_value = [
        _fake_entity("2020", score=0.9),
        _fake_entity("2019", score=0.3),  # sous le seuil (_DATE_MIN_CONFIDENCE = 0.5)
    ]
    assert extract_dates("texte quelconque") == ["2020"]


@patch("app.services.extractor._get_date_ner_pipeline")
def test_ignores_non_date_entity_groups(mock_get_pipeline):
    mock_pipeline = mock_get_pipeline.return_value
    mock_pipeline.return_value = [
        _fake_entity("Paris", score=0.9, entity_group="LOC"),
        _fake_entity("2021", score=0.9, entity_group="DATE"),
    ]
    assert extract_dates("texte quelconque") == ["2021"]


@patch("app.services.extractor._get_date_ner_pipeline")
def test_preserves_order_and_duplicates(mock_get_pipeline):
    """Contrairement à extract_skills(), pas de tri ni de déduplication :
    deux dates identiques peuvent être légitimement distinctes."""
    mock_pipeline = mock_get_pipeline.return_value
    mock_pipeline.return_value = [
        _fake_entity("2020", score=0.9),
        _fake_entity("2020", score=0.9),
        _fake_entity("2021", score=0.9),
    ]
    assert extract_dates("texte quelconque") == ["2020", "2020", "2021"]


@patch("app.services.extractor._get_date_ner_pipeline")
def test_empty_string_returns_empty_list(mock_get_pipeline):
    mock_pipeline = mock_get_pipeline.return_value
    mock_pipeline.return_value = []
    assert extract_dates("") == []


@patch("app.services.extractor._get_date_ner_pipeline")
def test_truncates_text_before_calling_model(mock_get_pipeline):
    """Vérifie la troncature de sécurité (_MAX_CHARS_FOR_DATE_MODEL) :
    on inspecte ce qui est réellement envoyé au pipeline, sans avoir
    besoin d'un texte de 2000+ caractères ni du vrai modèle."""
    mock_pipeline = mock_get_pipeline.return_value
    mock_pipeline.return_value = []

    long_text = "a" * 5000
    extract_dates(long_text)

    called_text = mock_pipeline.call_args[0][0]
    assert len(called_text) <= 2000


def test_real_model_extracts_duration_from_cv_text():
    """Test d'intégration (PAS mocké) : reproduit le cas déjà validé
    manuellement dans training/test_pretrained_ner.py, où CamemBERT
    détecte "10+ ans" avec 0.99 de confiance. Plus lent que les autres
    tests (charge le vrai modèle BERT en mémoire)."""
    text = "Fort de 10+ ans d'expérience en développement back-end"
    result = extract_dates(text)
    assert any("10" in date for date in result)