from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.cv import CVAnalysisResponse, CVParseResponse
from app.services.extractor import extract_entities
from app.services.parser import (
    EmptyDocumentError,
    SupportedFileType,
    UnsupportedFileTypeError,
    _detect_file_type,
    extract_text,
)
from app.services.preprocessor import clean_text

router = APIRouter()

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


async def _read_upload_bytes(file: UploadFile) -> bytes:
    """Valide et lit le fichier uploadé, commun à toutes les routes CV."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant.")

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (max {MAX_FILE_SIZE_MB} Mo).",
        )

    return file_bytes


def _extract_clean_text(filename: str, file_bytes: bytes) -> tuple[SupportedFileType, str]:
    """Parsing + nettoyage, commun à /parse et /analyze."""
    try:
        file_type = _detect_file_type(filename)
        raw_text = extract_text(filename, file_bytes)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EmptyDocumentError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return file_type, clean_text(raw_text)


@router.post(
    "/parse",
    response_model=CVParseResponse,
    summary="Upload un CV (PDF ou DOCX) et retourne le texte nettoyé",
)
async def parse_cv(file: UploadFile = File(...)) -> CVParseResponse:
    file_bytes = await _read_upload_bytes(file)
    file_type, text = _extract_clean_text(file.filename, file_bytes)

    return CVParseResponse(
        filename=file.filename,
        file_type=file_type.value,
        text=text,
        char_count=len(text),
    )


@router.post(
    "/analyze",
    response_model=CVAnalysisResponse,
    summary="Upload un CV et retourne le texte nettoyé + les entités extraites",
)
async def analyze_cv(file: UploadFile = File(...)) -> CVAnalysisResponse:
    file_bytes = await _read_upload_bytes(file)
    file_type, text = _extract_clean_text(file.filename, file_bytes)
    entities = extract_entities(text)

    return CVAnalysisResponse(
        filename=file.filename,
        file_type=file_type.value,
        text=text,
        char_count=len(text),
        entities=entities,
    )