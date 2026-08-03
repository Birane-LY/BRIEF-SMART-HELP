import io
import numpy as np
from app.models.asr_model import get_asr_model

class ASRService:
    def __init__(self):
        # Load the automatic speech recognition model into memory
        self.model = get_asr_model()

    async def transcribe(self, audio_bytes: bytes) -> str | None:
        """
        Transcribes raw audio byte streams directly in-memory without disk I/O overhead.
        """
        if not audio_bytes:
            return None

        try:
            # Wrap raw binary audio stream into an in-memory byte buffer
            audio_buffer = io.BytesIO(audio_bytes)
            segments, _ = self.model.transcribe(
                audio_buffer,
                beam_size=5,
                language=None,
            )
            
            transcription_text = " ".join(segment.text.strip() for segment in segments).strip()
            return transcription_text if transcription_text else None

        except Exception as e:
            print(f"[ERREUR AUDIO] Échec critique du traitement ou du décodage du flux audio : {str(e)}")
            return None

asr_service = ASRService()
