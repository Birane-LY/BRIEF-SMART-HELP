from pydantic import BaseModel


class VisionResult(BaseModel):
    label: str
    score: float
    is_relevant: bool

class RAGResult(BaseModel):
    id: str
    title: str
    status: str
    action: str
    source: str
    conditions: list[str] = []
    score: float

class TicketResponse(BaseModel):
    message: str
    has_description: bool
    has_audio: bool
    image_count: int
    transcription: str | None = None
    full_text_context: str | None = None
    vision_analysis: list[VisionResult] = []
    rag_result: RAGResult | None = None
    statut_propose: str = "A_VERIFIER"
    score_fiabilite: float = 0.0
    warnings: list[str] = []