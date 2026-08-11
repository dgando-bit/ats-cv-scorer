"""
Script de test ponctuel (pas encore intégré à l'app) pour évaluer
concrètement le modèle pré-entraîné yashpwr/resume-ner-bert-v2, avant
de décider s'il remplace notre modèle spaCy maison dans extractor.py.

Usage :
    python test_pretrained_ner.py
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Petit extrait de CV en anglais, dans le contexte d'entraînement normal
# du modèle (proche du dataset Kaggle qu'on a utilisé nous-mêmes).
ENGLISH_SAMPLE = """
Senior Software Engineer with 8 years of experience in Python, Docker,
and Kubernetes. Worked at Google as a Backend Developer from 2018 to 2022.
Holds a Master's degree in Computer Science from Stanford University.
Fluent in English and Spanish.
"""

# Extrait réel de votre CV en français (colonne principale, texte nettoyé).
FRENCH_SAMPLE = """
Destin GANDO
MACHINE LEARNING ENGINEER
Fort de 10+ ans d'expérience en développement back-end, je maîtrise Python,
SQL, Git et les bonnes pratiques logicielles que je mets désormais au service
de la préparation des données, de la création des pipelines et du déploiement
des modèles ML.
Compétences clés : Modélisation ML (Scikit-learn), Deep Learning, LLM, RAG,
CNN, RNN, Traitement d'images, Développement d'API (FastAPI)
"""


def run_and_print(label: str, text: str, ner_pipeline) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    results = ner_pipeline(text)
    if not results:
        print("  Aucune entité détectée.")
        return
    for entity in results:
        print(
            f"  [{entity['entity_group']:>15}] "
            f"\"{entity['word']}\" "
            f"(confiance: {entity['score']:.2f})"
        )


def main() -> None:
    print("Chargement du modèle yashpwr/resume-ner-bert-v2...")
    print("(premier lancement : téléchargement, peut prendre quelques minutes)")
    ner_pipeline = pipeline(
        "ner",
        model="yashpwr/resume-ner-bert-v2",
        aggregation_strategy="simple",  # regroupe les sous-tokens en entités complètes
    )

    run_and_print("CV EN ANGLAIS", ENGLISH_SAMPLE, ner_pipeline)
    run_and_print("CV EN FRANÇAIS (extrait réel)", FRENCH_SAMPLE, ner_pipeline)

    print("\n\nChargement du modèle Jean-Baptiste/camembert-ner-with-dates...")
    # use_fast=False : contourne un bug de conversion du tokenizer
    # SentencePiece vers l'implémentation "fast" avec cette version de
    # transformers (AttributeError sur vocab_file). Le tokenizer "slow"
    # (implémentation Python pure) fonctionne normalement.
    camembert_tokenizer = AutoTokenizer.from_pretrained(
        "Jean-Baptiste/camembert-ner-with-dates", use_fast=False
    )
    camembert_model = AutoModelForTokenClassification.from_pretrained(
        "Jean-Baptiste/camembert-ner-with-dates"
    )
    camembert_pipeline = pipeline(
        "ner",
        model=camembert_model,
        tokenizer=camembert_tokenizer,
        aggregation_strategy="simple",
    )
    run_and_print("CAMEMBERT - CV EN FRANÇAIS (extrait réel)", FRENCH_SAMPLE, camembert_pipeline)


if __name__ == "__main__":
    main()