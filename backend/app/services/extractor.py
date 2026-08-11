"""
Service d'extraction d'entités structurées à partir du texte nettoyé
d'un CV (compétences, email, téléphone, dates/durée d'expérience).

Architecture hybride, chaque type d'entité utilisant l'outil le plus
adapté plutôt qu'un unique modèle "universel" :
  - SKILL       : EntityRuler spaCy + taxonomie (services/extractor.py)
  - EMAIL/PHONE : regex (format très régulier, le NER n'apporte rien)
  - DATES/DURÉE : modèle CamemBERT pré-entraîné (Jean-Baptiste/camembert-ner-with-dates)
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.language import Language
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "skills_taxonomy.json"

_DATE_MODEL_NAME = "Jean-Baptiste/camembert-ner-with-dates"
_DATE_MIN_CONFIDENCE = 0.5
# Limite approximative (pas un comptage exact de tokens) pour rester
# sous la limite de séquence de CamemBERT (512 tokens). Un CV plus
# long verra sa fin ignorée par extract_dates() — limite acceptée pour
# un premier jet ; un découpage en chunks serait la vraie solution si
# ça s'avère gênant en pratique.
_MAX_CHARS_FOR_DATE_MODEL = 2000

# Regex email : pragmatique, pas 100% conforme RFC 5322 (qui est
# notoirement complexe), mais couvre la quasi-totalité des adresses
# réelles rencontrées sur des CV.
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Regex téléphone français : gère +33/0033 ou le 0 initial, suivi de
# 9 chiffres groupés par 2, avec séparateurs optionnels (espace, point,
# tiret) ou aucun séparateur. Ne couvre PAS les formats internationaux
# hors France — limite acceptée vu le contexte du projet (CV en français).
_PHONE_PATTERN = re.compile(r"(?:(?:\+33|0033)[\s.-]?|0)[1-9](?:[\s.-]?\d{2}){4}")


@lru_cache(maxsize=1)
def _load_skill_canonical_map() -> dict[str, str]:
    """
    Charge la taxonomie et construit une table de correspondance
    "forme en minuscules" → "forme canonique" (celle du JSON).

    Utile pour normaliser la sortie : si le CV contient "python" ou
    "PYTHON", on veut toujours renvoyer "Python" (l'orthographe de
    référence définie dans la taxonomie), pas la casse trouvée dans
    le texte source.

    @lru_cache : évite de relire et reparser le fichier JSON à chaque
    appel de extract_skills() — une seule lecture par processus.
    """
    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        taxonomy: dict[str, list[str]] = json.load(f)

    canonical_map: dict[str, str] = {}
    for skills in taxonomy.values():
        for skill in skills:
            canonical_map[skill.lower()] = skill

    return canonical_map


@lru_cache(maxsize=1)
def _get_skill_nlp() -> Language:
    """
    Construit (une seule fois, grâce à @lru_cache) un pipeline spaCy
    "blank" équipé d'un EntityRuler chargé avec la taxonomie de
    compétences.

    @lru_cache est important ici : construire l'EntityRuler a un coût
    (chargement JSON + indexation des patterns), et cette fonction sera
    appelée à chaque requête de l'API. Sans le cache, on referait ce
    travail à chaque appel — avec, il n'est fait qu'une fois par
    processus, à la première utilisation.
    """
    nlp = spacy.blank("fr")

    # phrase_matcher_attr="LOWER" : les patterns matchent en étant
    # insensibles à la casse (compare la forme en minuscules des tokens).
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})

    canonical_map = _load_skill_canonical_map()
    patterns = [{"label": "SKILL", "pattern": skill} for skill in canonical_map.values()]
    ruler.add_patterns(patterns)

    return nlp


def extract_skills(text: str) -> list[str]:
    """
    Détecte les compétences techniques présentes dans le texte, en se
    basant sur la taxonomie (correspondance exacte, insensible à la casse).

    Returns:
        Liste triée de compétences, dédupliquées et normalisées à leur
        orthographe canonique (ex: toujours "Python", jamais "python").
    """
    nlp = _get_skill_nlp()
    canonical_map = _load_skill_canonical_map()

    doc = nlp(text)
    found_skills = {
        canonical_map.get(ent.text.lower(), ent.text) for ent in doc.ents if ent.label_ == "SKILL"
    }

    return sorted(found_skills)


def extract_email(text: str) -> str | None:
    """
    Extrait la première adresse email trouvée dans le texte.

    Returns:
        L'adresse email, ou None si aucune n'est trouvée. Un CV n'a
        normalement qu'une seule adresse de contact, donc contrairement
        à extract_skills(), on ne retourne qu'un seul résultat, pas
        une liste.
    """
    match = _EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    """
    Extrait le premier numéro de téléphone français trouvé dans le texte.

    Returns:
        Le numéro tel que trouvé dans le texte (espaces/séparateurs
        d'origine préservés), ou None si aucun n'est trouvé.
    """
    match = _PHONE_PATTERN.search(text)
    return match.group(0).strip() if match else None


@lru_cache(maxsize=1)
def _get_date_ner_pipeline():
    """
    Charge (une seule fois, en cache) le pipeline CamemBERT pour la
    détection de dates/durées.

    use_fast=False : contourne un bug de conversion du tokenizer
    SentencePiece vers l'implémentation "fast" avec la version de
    transformers installée (AttributeError sur vocab_file — déjà
    rencontré lors du test manuel du modèle, voir
    training/test_pretrained_ner.py).

    Contrairement à _get_skill_nlp() (rapide à construire), charger un
    modèle BERT de ~440 Mo a un coût réel — le cache est encore plus
    important ici pour ne pas le refaire à chaque requête.
    """
    tokenizer = AutoTokenizer.from_pretrained(_DATE_MODEL_NAME, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(_DATE_MODEL_NAME)
    return hf_pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def extract_dates(text: str) -> list[str]:
    """
    Détecte les dates et durées (ex: "10+ ans", "2021 - 2025") dans le
    texte, via le modèle CamemBERT pré-entraîné.

    Contrairement à extract_skills() (taxonomie fermée et fiable), ce
    modèle généraliste n'est pas spécialisé CV — on filtre donc sur un
    seuil de confiance minimum (_DATE_MIN_CONFIDENCE) pour écarter les
    détections peu fiables plutôt que de tout retourner brut.

    Returns:
        Liste des dates/durées trouvées, dans l'ordre d'apparition dans
        le texte (pas de tri ni de déduplication — contrairement aux
        compétences, deux dates identiques peuvent être légitimement
        distinctes, ex: deux expériences démarrées la même année).
    """
    ner_pipeline = _get_date_ner_pipeline()

    # Troncature de sécurité : voir le commentaire sur
    # _MAX_CHARS_FOR_DATE_MODEL en haut du fichier.
    truncated_text = text[:_MAX_CHARS_FOR_DATE_MODEL]

    results = ner_pipeline(truncated_text)
    return [
        r["word"].strip()
        for r in results
        if r["entity_group"] == "DATE" and r["score"] >= _DATE_MIN_CONFIDENCE
    ]