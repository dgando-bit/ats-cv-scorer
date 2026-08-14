from app.services.embeddings import compute_semantic_similarity, get_embedding


def test_get_embedding_returns_fixed_size_vector():
    embedding = get_embedding("Développeur Python senior")
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0


def test_identical_texts_have_similarity_close_to_one():
    text = "Développeur Python avec 5 ans d'expérience en FastAPI"
    similarity = compute_semantic_similarity(text, text)
    assert similarity > 0.99


def test_similar_meaning_different_words_have_high_similarity():
    """Le point clé du matching sémantique : deux formulations
    différentes du même métier doivent être jugées proches."""
    text_a = "Développeur back-end spécialisé en API Python"
    text_b = "Ingénieur logiciel côté serveur, expert Python"
    similarity = compute_semantic_similarity(text_a, text_b)
    assert similarity > 0.5


def test_unrelated_texts_have_low_similarity():
    text_a = "Développeur Python spécialisé en machine learning"
    text_b = "Chef cuisinier pâtissier, spécialiste des desserts en restaurant gastronomique"
    similarity = compute_semantic_similarity(text_a, text_b)
    assert similarity < 0.4


def test_similarity_is_symmetric():
    text_a = "Data scientist avec expérience en NLP"
    text_b = "Ingénieur en traitement du langage naturel"
    assert compute_semantic_similarity(text_a, text_b) == compute_semantic_similarity(text_b, text_a)


def test_similarity_score_is_bounded():
    text_a = "Python Docker Kubernetes"
    text_b = "Recette de tarte aux pommes traditionnelle"
    similarity = compute_semantic_similarity(text_a, text_b)
    assert -1.0 <= similarity <= 1.0