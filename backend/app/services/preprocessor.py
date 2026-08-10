"""
Service de pré-traitement du texte brut extrait d'un CV.

Ce module nettoie le texte issu de parser.py avant qu'il soit analysé
par extractor.py (extraction d'entités) ou embeddings.py (vectorisation).
Il ne fait AUCUNE interprétation du contenu — seulement du nettoyage
et de la normalisation de la mise en forme.
"""

from __future__ import annotations

import re
import unicodedata

# Une ligne est considérée comme "complète" (fin de phrase, item de liste,
# titre) si elle se termine par un de ces caractères : on ne cherche
# alors jamais à la fusionner avec la ligne suivante.
_SENTENCE_END_CHARS = (".", "!", "?", ":", ";", ")", "»")

# Caractères qui, en DÉBUT de ligne suivante, indiquent une continuation
# probable de la ligne précédente plutôt qu'une nouvelle phrase/section.
_CONTINUATION_START_CHARS = ("(", "«")


def _normalize_unicode(text: str) -> str:
    """
    Normalise les caractères unicode (formes composées vs décomposées).

    Utile car deux PDF contenant le même mot accentué ("é" par exemple)
    peuvent l'encoder différemment selon l'éditeur qui les a générés,
    ce qui casserait silencieusement des comparaisons de texte plus tard.
    """
    return unicodedata.normalize("NFKC", text)


def _collapse_whitespace(line: str) -> str:
    """Remplace les suites d'espaces/tabulations par un seul espace."""
    return re.sub(r"[ \t]+", " ", line).strip()


def _should_merge_with_next(current_line: str, next_line: str) -> bool:
    """
    Détermine si `current_line` doit être fusionnée avec `next_line`
    plutôt que traitée comme une ligne indépendante.

    Limite connue : cette heuristique peut se tromper, par exemple sur
    des lignes de contact empilées (numéro de téléphone suivi d'un
    email en minuscule) qui seront fusionnées à tort. Un nettoyage
    plus fin pourrait être ajouté plus tard si besoin.
    """
    if not current_line or not next_line:
        return False

    if current_line.endswith(_SENTENCE_END_CHARS):
        return False

    if current_line.endswith("-"):
        # Mot coupé en fin de ligne par la justification du PDF
        return True

    first_char = next_line[0]
    if first_char.islower():
        return True

    if next_line.startswith(_CONTINUATION_START_CHARS):
        return True

    return False


def _merge_wrapped_lines(text: str) -> str:
    """
    Fusionne les lignes artificiellement coupées par la justification
    du texte dans le PDF d'origine (parfois un seul mot par ligne),
    tout en préservant les sauts de ligne "légitimes" (fin de phrase,
    titres de section, items de liste).
    """
    raw_lines = [_collapse_whitespace(line) for line in text.split("\n")]
    raw_lines = [line for line in raw_lines if line]  # on retire les lignes vides

    if not raw_lines:
        return ""

    merged_lines: list[str] = []
    buffer = raw_lines[0]

    for line in raw_lines[1:]:
        if _should_merge_with_next(buffer, line):
            if buffer.endswith("-"):
                buffer = buffer[:-1] + line  # recolle le mot coupé, sans espace
            else:
                buffer = buffer + " " + line
        else:
            merged_lines.append(buffer)
            buffer = line

    merged_lines.append(buffer)
    return "\n".join(merged_lines)


def clean_text(raw_text: str) -> str:
    """
    Point d'entrée principal du preprocessing.

    Args:
        raw_text: texte brut tel que retourné par services.parser.extract_text

    Returns:
        Texte nettoyé : lignes coupées fusionnées, espaces normalisés,
        unicode normalisé. La structure ligne par ligne est conservée
        (une ligne = un item logique : titre de section, item de liste,
        phrase...) — ce sera la base de travail de extractor.py.
    """
    normalized = _normalize_unicode(raw_text)
    return _merge_wrapped_lines(normalized)