FROM python:3.11-slim

# Dépendances système nécessaires pour PyMuPDF / spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python d'abord (meilleur cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modèle spaCy pour le NER (français). Remplacer par en_core_web_lg si CV en anglais.
RUN python -m spacy download fr_core_news_lg

# Le modèle sentence-transformers sera téléchargé au premier lancement
# et mis en cache dans le volume model_cache (voir docker-compose.yml)

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
