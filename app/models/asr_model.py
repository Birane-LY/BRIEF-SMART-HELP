from functools import lru_cache
from faster_whisper import WhisperModel
from app.core.config import settings

@lru_cache(maxsize=1)
def get_asr_model() -> WhisperModel:
  """ Load faster-whisper only once """
  return WhisperModel(
    settings.whisper_model_name, device=settings.whisper_device, compute_type=settings.whisper_compute_type
  )