"""
Convertit le dataset Kaggle `yashpwrr/resume-ner-training-dataset`
(format : liste de {"text": ..., "annotations": [[start, end, label], ...]})
en fichiers .spacy exploitables par `spacy train`.

Le fichier JSON doit être téléchargé manuellement depuis Kaggle
(https://www.kaggle.com/datasets/yashpwrr/resume-ner-training-dataset)
et placé à : training/data/raw/train.json

Usage :
    python prepare_data.py --limit 2000 --test-size 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import spacy
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans

RAW_DATA_PATH = Path(__file__).parent / "data" / "raw" / "train.json"
OUTPUT_DIR = Path(__file__).parent / "data"


def _load_raw_records(limit: int | None) -> list[dict]:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {RAW_DATA_PATH}\n"
            "Téléchargez le dataset depuis Kaggle "
            "(yashpwrr/resume-ner-training-dataset) et placez le JSON à cet emplacement."
        )
    with open(RAW_DATA_PATH, encoding="utf-8") as f:
        records = json.load(f)

    if limit:
        records = records[:limit]

    return records


def _record_to_doc(record: dict, nlp: spacy.language.Language) -> Doc:
    text = record["text"]
    doc = nlp.make_doc(text)

    spans = []
    for start, end, label in record["annotations"]:
        # alignment_mode="contract" : si le span annoté ne tombe pas
        # exactement sur une frontière de token spaCy, on le rétrécit
        # plutôt que de le perdre.
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            spans.append(span)

    # Le dataset agrège 4 sources différentes : certaines annotations
    # se chevauchent (ex: un span SKILL englobant un autre span plus
    # précis). filter_spans() résout les conflits en gardant les spans
    # les plus longs — spaCy interdit que des entités se recouvrent.
    doc.ents = filter_spans(spans)
    return doc


def _build_docbin(records: list[dict], nlp: spacy.language.Language) -> DocBin:
    doc_bin = DocBin()
    skipped = 0

    for record in records:
        try:
            doc = _record_to_doc(record, nlp)
        except Exception:
            skipped += 1
            continue
        doc_bin.add(doc)

    if skipped:
        print(f"  {skipped} exemples ignorés (erreur de conversion)")

    return doc_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Nombre d'exemples à utiliser (défaut : tous)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.1, help="Proportion réservée à la validation"
    )
    args = parser.parse_args()

    print(f"Lecture de {RAW_DATA_PATH}...")
    records = _load_raw_records(args.limit)
    print(f"{len(records)} exemples chargés.")

    labels_found = {label for r in records for _, _, label in r["annotations"]}
    print(f"Labels détectés : {sorted(labels_found)}")

    random.seed(42)  # reproductibilité du split train/dev
    random.shuffle(records)
    split_idx = int(len(records) * (1 - args.test_size))
    train_records, dev_records = records[:split_idx], records[split_idx:]

    # nlp "blank" : on ne charge aucun modèle pré-entraîné, juste le
    # tokenizer anglais — suffisant pour construire les Doc à partir du texte.
    nlp = spacy.blank("en")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Conversion de {len(train_records)} exemples d'entraînement...")
    train_docbin = _build_docbin(train_records, nlp)
    train_docbin.to_disk(OUTPUT_DIR / "train.spacy")

    print(f"Conversion de {len(dev_records)} exemples de validation...")
    dev_docbin = _build_docbin(dev_records, nlp)
    dev_docbin.to_disk(OUTPUT_DIR / "dev.spacy")

    print(f"Terminé. Fichiers écrits dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()