# ATS CV Scorer

Application NLP d'analyse de CV et de matching sémantique avec des offres d'emploi.

## Stack

- **Backend** : FastAPI, spaCy, sentence-transformers, scikit-learn
- **Frontend** : React (Vite) + Tailwind CSS

## Démarrage rapide

```bash
docker compose up --build
```

- Backend : http://localhost:8000 (docs interactives : http://localhost:8000/docs)
- Frontend : http://localhost:5173

Le premier build sera un peu long (téléchargement du modèle spaCy `fr_core_news_lg`
pendant le build de l'image, puis téléchargement du modèle sentence-transformers au
premier appel — mis en cache ensuite dans le volume `model_cache`).

## Structure du projet

```
ats-cv-scorer/
├── backend/
│   └── app/
│       ├── main.py
│       ├── routers/       # endpoints API
│       ├── services/      # parsing, NLP, scoring
│       ├── models/        # schémas Pydantic
│       └── core/          # config
└── frontend/
    └── src/
        ├── components/
        ├── pages/
        └── api/
```

## Prochaines étapes

1. Service de parsing CV (PDF/DOCX → texte)
2. Extraction d'entités (compétences, expérience, formation)
3. Embeddings sémantiques (sentence-transformers)
4. Scoring composite CV ↔ offre
5. UI de restitution du score
