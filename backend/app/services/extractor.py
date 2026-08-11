"""
Service d'extraction d'entités structurées à partir du texte nettoyé
d'un CV (compétences, email, téléphone, dates/durée d'expérience).

Architecture hybride, chaque type d'entité utilisant l'outil le plus
adapté plutôt qu'un unique modèle "universel" :
  - SKILL       : EntityRuler spaCy + taxonomie (services/extractor.py)
  - EMAIL/PHONE : regex (format très régulier, le NER n'apporte rien)
  - DATES/DURÉE : modèle CamemBERT pré-entraîné (à intégrer ensuite)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.language import Language

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "skills_taxonomy.json"


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