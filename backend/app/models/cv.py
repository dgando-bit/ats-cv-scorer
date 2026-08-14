from pydantic import BaseModel, Field


class CVParseResponse(BaseModel):
    filename: str = Field(..., description="Nom du fichier original")
    file_type: str = Field(..., description="Type détecté : pdf ou docx")
    text: str = Field(..., description="Texte brut extrait du CV")
    char_count: int = Field(..., description="Nombre de caractères extraits")


class CVEntities(BaseModel):
    skills: list[str] = Field(default_factory=list, description="Compétences techniques détectées")
    email: str | None = Field(default=None, description="Adresse email détectée")
    phone: str | None = Field(default=None, description="Numéro de téléphone détecté")
    dates: list[str] = Field(
        default_factory=list, description="Dates/durées détectées (ex: durée d'expérience)"
    )


class CVAnalysisResponse(BaseModel):
    filename: str = Field(..., description="Nom du fichier original")
    file_type: str = Field(..., description="Type détecté : pdf ou docx")
    text: str = Field(..., description="Texte nettoyé du CV (après preprocessing)")
    char_count: int = Field(..., description="Nombre de caractères du texte nettoyé")
    entities: CVEntities = Field(..., description="Entités structurées extraites du CV")


class ErrorResponse(BaseModel):
    detail: str