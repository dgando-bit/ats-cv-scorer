"""
Service de génération d'embeddings et de calcul de similarité
sémantique, utilisé pour comparer un CV à une offre d'emploi sur le
sens global du texte plutôt que sur la simple présence de mots-clés.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer, util

_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """
    Charge (une seule fois, en cache) le modèle d'embeddings multilingue.

    paraphrase-multilingual-MiniLM-L12-v2 : compromis vitesse/qualité
    retenu pour ce projet — supporte nativement le français, ~470 Mo,
    raisonnable sur CPU pour un usage interactif (contrairement à
    mpnet, plus précis mais nettement plus lourd).
    """
    return SentenceTransformer(_EMBEDDING_MODEL_NAME)


def get_embedding(text: str) -> np.ndarray:
    """
    Encode un texte en un vecteur numérique (embedding) de dimension fixe.

    Le vecteur capture le "sens" global du texte : deux textes au sens
    proche auront des vecteurs proches dans l'espace vectoriel, même
    s'ils n'utilisent pas les mêmes mots — contrairement à un matching
    de mots-clés classique, qui raterait par exemple le lien entre
    "développeur back-end" et "ingénieur serveur".
    """
    model = _get_embedding_model()
    return model.encode(text, convert_to_numpy=True)


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Calcule la similarité sémantique entre deux textes, sur une échelle
    de 0 à 1 (1 = sens quasi identique, 0 = aucun rapport).

    Utilise la similarité cosinus entre les deux embeddings : elle
    mesure l'angle entre les deux vecteurs plutôt que leur distance
    brute, ce qui la rend insensible à la longueur des textes comparés
    — un CV de 3000 caractères vs une offre de 200 caractères n'est
    pas pénalisé du simple fait de sa longueur.
    """
    embedding_a = get_embedding(text_a)
    embedding_b = get_embedding(text_b)

    similarity = util.cos_sim(embedding_a, embedding_b)
    # cos_sim retourne un tenseur 1x1 ; on en extrait le scalaire.
    return float(similarity.item())