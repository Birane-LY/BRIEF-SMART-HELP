from pathlib import Path
from fastapi import File, UploadFile, HTTPException

async def validate_file(file: UploadFile, allowed_extensions: set[str], max_size_mb:int | None = None) -> bytes:
  if not file.filename:
    raise HTTPException(
      status_code=422, detail={
        "code": "EMPTY_FILENAME",
        "message": "Le fichier envoyé est invalide",
      },
    )
  extension = Path(file.filename).suffix.lower()

  if extension not in allowed_extensions:
    raise HTTPException(
      status_code=415, detail={
        "code": "UNSUPPORTED_FILE_TYPE",
        "message": "Format de fichier non accepté",
      },
    )

  content = await file.read()

  if not content:
    raise HTTPException(
      status_code=422, detail={
        "code": "EMPTY_FILE",
        "message": "Le fichier envoyé est vide",
      },
    )

  if max_size_mb and len(content) > max_size_mb * 1024 * 1024:
    raise HTTPException(
        status_code=413, detail={
          "code": "FILE_TOO_LARGE",
          "message": f"La taille du fichier dépasse la limite autorisée ({max_size_mb} MB).",
        },
    )

  await file.seek(0)
  return content