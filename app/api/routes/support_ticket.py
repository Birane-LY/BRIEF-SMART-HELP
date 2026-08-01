from typing import Annotated
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.core.config import settings
from app.schemas.ticket_response import TicketResponse
from app.utils.file_validation import validate_file
from pydantic import WithJsonSchema 

router = APIRouter()

# Custom type to enforce binary file upload format in Swagger UI
SwaggerFile = Annotated[UploadFile, WithJsonSchema({"type": "string", "format": "binary"})]

@router.post("/support-ticket", response_model=TicketResponse)
async def create_support_ticket(
    description: Annotated[str | None, Form()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
    images: Annotated[list[SwaggerFile], File()] = [], 
):
    images = images or []
    description = description.strip() if description else None
    
    # Silently truncate to process only the first allowed images
    images_to_process = images[:settings.max_images]

    # Ensure images are accompanied by either a description or an audio message
    if images_to_process and not description and audio is None:
        raise HTTPException(
            status_code=422, 
            detail={
                "code": "MISSING_CLAIM_CONTEXT",
                "message": "Veuillez accompagner vos images d'une description écrite ou d'un message audio expliquant votre réclamation."
            },
        )

    # Validate and read audio file content
    audio_content = None
    if audio is not None:
        audio_content = await validate_file(
            file=audio, 
            allowed_entensions=settings.audio_extensions_set, 
            max_size_mb=settings.max_audio_size_mb
        )

    # Validate and read image contents (up to max_images)
    image_contents = []
    for image in images_to_process:
        content = await validate_file(
            file=image, 
            allowed_entensions=settings.image_extensions_set, 
            max_size_mb=settings.max_image_size_mb
        )
        image_contents.append(content)

    # Ensure the claim is not empty (a text description or audio recording is required)
    if not description and audio_content is None:
        raise HTTPException(
            status_code=422, 
            detail={
                "code": "EMPTY_CLAIM",
                "message": "Une description écrite ou un message audio est obligatoire.",
            },
        )

    return {
        "message": "Réclamation reçue",
        "has_description": bool(description),
        "has_audio": audio_content is not None,
        "image_count": len(image_contents),  # Returns at most the maximum allowed images
    }