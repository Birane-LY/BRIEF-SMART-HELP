from typing import Annotated
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.core.config import settings
from app.schemas.ticket_response import TicketResponse
from app.utils.file_validation import validate_file

router = APIRouter()

@router.post("/support-ticket", response_model=TicketResponse)
async def create_support_ticket(
  description: Annotated[str | None, Form()] = None,
  audio: Annotated[UploadFile | None, File()] = None,
  images: Annotated[list[UploadFile] | None, File()] = None,
):
  images = images or []
  description = description.strip() if description else None

  if len(images) > settings.max_images:
    raise HTTPException(
      status_code=422, detail={
        "code": "TOO_MANY_IMAGES",
        "message": f"Vous pouvez téléversez au maximum {settings.max_images} images.",
      },
    )

  if images and not description and audio is None:
    raise HTTPException(
      status_code=422, detail={
        "code": "MISSING_CLAIM_CONTEXT",
        "message": "Veuillez acompagner vos images d'une description écrite ou d'un message audio expliquant votre réclamation."
      },
    )

  audio_content = None
  if audio is not None:
   audio_content = await validate_file(file=audio, allowed_entensions=settings.audio_extensions_set, max_size_mb= settings.max_audio_size_mb)

  image_contents = []
  for image in images:
    content = await validate_file(file=image, allowed_entensions=settings.image_extensions_set, max_size_mb=settings.max_image_size_mb)
    image_contents.append(content)

  if not description and audio_content is None:
    raise HTTPException(
      status_code=422, detail={
        "code": "EMPTY_CLAIM",
        "message": "Une description écrite ou un message audio est obligatoire.",
      },
    )

  return{
    "message": "Réclamation reçue",
    "has_description": bool(description),
    "has_audio": audio_content is not None,
    "image_count": len(image_contents),
  }