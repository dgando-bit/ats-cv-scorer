from app.services.preprocessor import clean_text


def test_merges_words_broken_across_lines():
    """Cas réel observé : justification PDF qui coupe un mot par ligne."""
    raw = (
        "Concevoir, \n"
        "implémenter \n"
        "et \n"
        "optimiser \n"
        "les \n"
        "bases \n"
        "de \n"
        "données\n"
        "(modélisations requêtes, intégrité)"
    )
    result = clean_text(raw)
    assert result == "Concevoir, implémenter et optimiser les bases de données (modélisations requêtes, intégrité)"


def test_preserves_section_headers():
    """Un titre de section (ligne courte, majuscule) ne doit pas être
    fusionné avec le contenu qui suit s'il commence par une majuscule."""
    raw = "Soft-skills\nEsprit Analytique\nRigueur"
    result = clean_text(raw)
    lines = result.split("\n")
    assert "Soft-skills" in lines
    assert "Esprit Analytique" in lines
    assert "Rigueur" in lines


def test_does_not_merge_when_line_ends_with_strong_punctuation():
    raw = "Domaine : imprimerie.\nGérer le parc informatique"
    result = clean_text(raw)
    lines = result.split("\n")
    assert lines[0] == "Domaine : imprimerie."
    assert lines[1] == "Gérer le parc informatique"


def test_merges_hyphenated_word_split_across_lines():
    """Mot coupé par un tiret de justification : pas d'espace à la fusion."""
    raw = "Développement d'applications multi-\nplateformes"
    result = clean_text(raw)
    assert result == "Développement d'applications multiplateformes"


def test_collapses_multiple_spaces():
    raw = "Python,   Pandas,    Numpy"
    result = clean_text(raw)
    assert result == "Python, Pandas, Numpy"


def test_removes_empty_lines():
    raw = "Formation\n\n\n2018 : BTS"
    result = clean_text(raw)
    assert "\n\n" not in result


def test_empty_string_returns_empty_string():
    assert clean_text("") == ""


def test_single_line_unchanged():
    assert clean_text("Développeur Python") == "Développeur Python"


def test_continuation_after_parenthesis_open():
    raw = "Mobilité urbaine & Billettique MaaS \n(2021 - 2025)"
    result = clean_text(raw)
    assert result == "Mobilité urbaine & Billettique MaaS (2021 - 2025)"