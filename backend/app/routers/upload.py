from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.cv import CVParseResponse
from app.services.parser import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    _detect_file_type,
    extract_text,
)

router = APIRouter()

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/parse",
    response_model=CVParseResponse,
    summary="Upload un CV (PDF ou DOCX) et retourne le texte extrait",
)
async def parse_cv(file: UploadFile = File(...)) -> CVParseResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant.")

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (max {MAX_FILE_SIZE_MB} Mo).",
        )

    try:
        file_type = _detect_file_type(file.filename)
        text = extract_text(file.filename, file_bytes)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EmptyDocumentError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return CVParseResponse(
        filename=file.filename,
        file_type=file_type.value,
        text=text,
        char_count=len(text),
    )
