from pydantic import BaseModel, Field


class CVParseResponse(BaseModel):
    filename: str = Field(..., description="Nom du fichier original")
    file_type: str = Field(..., description="Type détecté : pdf ou docx")
    text: str = Field(..., description="Texte brut extrait du CV")
    char_count: int = Field(..., description="Nombre de caractères extraits")


class ErrorResponse(BaseModel):
    detail: str
