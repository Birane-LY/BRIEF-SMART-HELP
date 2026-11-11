
from functools import lru_cache

from transformers import pipeline
from app.core.config import settings

@lru_cache(maxsize=1)
def get_vision_pipeline():
    return pipeline(
        task="zero-shot-image-classification",
        model=settings.vision_model_name,
        device=-1,
    )
