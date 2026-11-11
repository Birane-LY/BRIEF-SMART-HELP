
import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import WithJsonSchema

from app.core.config import settings
from app.schemas.ticket_response import TicketResponse
from app.services.audio_service import asr_service
from app.services.vision_service import vision_service
from app.services.rag_service import rag_service
from app.services.diagnostic_service import diagnostic_service
from app.utils.file_validation import validate_file


router = APIRouter()

SwaggerFile = Annotated[
    UploadFile,
    WithJsonSchema({"type": "string", "format": "binary"}),
]

# Évite plusieurs traitements IA lourds simultanés
# dans le même processus Render.
AI_SEMAPHORE = asyncio.Semaphore(1)


@router.post("/support-ticket", response_model=TicketResponse)
async def create_support_ticket(
    description: Annotated[str | None, Form()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
    images: Annotated[list[SwaggerFile] | None, File()] = None,
):
    images = images or []
    description = description.strip() if description else None
    images_to_process = images[:settings.max_images]

    # 1. Vérification du contexte de réclamation
    if images_to_process and not description and audio is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "MISSING_CLAIM_CONTEXT",
                "message": (
                    "Veuillez accompagner vos images d'une description "
                    "écrite ou d'un message audio."
                ),
            },
        )

    # 2. Validation audio
    audio_content = None

    if audio is not None:
        audio_content = await validate_file(
            file=audio,
            allowed_extensions=settings.audio_extensions_set,
            max_size_mb=settings.max_audio_size_mb,
        )

    # 3. Validation des images
    image_contents = []

    for image in images_to_process:
        content = await validate_file(
            file=image,
            allowed_extensions=settings.image_extensions_set,
            max_size_mb=settings.max_image_size_mb,
        )
        image_contents.append(content)

    if not description and not audio_content:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "EMPTY_CLAIM",
                "message": (
                    "Une description écrite ou un message audio "
                    "est obligatoire."
                ),
            },
        )

    # 4. Pipeline IA : une requête lourde à la fois
    async with AI_SEMAPHORE:

        # Transcription audio
        transcription = None

        if audio_content:
            transcription = await asr_service.transcribe(audio_content)

        # Agrégation du contexte textuel
        full_text = " ".join(
            filter(None, [description, transcription])
        ).strip()

        # Analyse des images
        vision_analysis = []

        if image_contents:
            vision_analysis = await vision_service.analyze_images(
                image_contents
            )

        # Recherche RAG exécutée hors de la boucle événementielle
        rag_result = None

        if full_text:
            rag_result = await run_in_threadpool(
                rag_service.search,
                full_text,
            )

        # Diagnostic Groq exécuté hors de la boucle événementielle
        diag = await run_in_threadpool(
            diagnostic_service.evaluate,
            full_text,
            vision_analysis,
            rag_result,
        )

    return TicketResponse(
        message="Réclamation reçue et analysée.",
        has_description=bool(description),
        has_audio=audio_content is not None,
        image_count=len(image_contents),
        transcription=transcription,
        full_text_context=full_text or None,
        vision_analysis=vision_analysis,
        rag_result=rag_result,
        statut_propose=diag.get("statut_propose", "A_VERIFIER"),
        score_fiabilite=diag.get("score_fiabilite", 0.0),
        warnings=diag.get("warnings", []),
    )
