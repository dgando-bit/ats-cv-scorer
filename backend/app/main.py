from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import upload

app = FastAPI(
    title="ATS CV Scorer API",
    description="API d'analyse de CV et de matching sémantique avec des offres d'emploi",
    version="0.1.0",
)

# CORS pour permettre au frontend React (localhost:5173) d'appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "ATS CV Scorer API — voir /docs pour la documentation interactive"}


app.include_router(upload.router, prefix="/api/cv", tags=["cv"])

# Les prochains routers (score, match) seront branchés de la même façon :
# from app.routers import score, match
# app.include_router(score.router, prefix="/api/score", tags=["score"])