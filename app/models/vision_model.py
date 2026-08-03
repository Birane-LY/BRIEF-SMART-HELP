from functools import lru_cache
from transformers import pipeline
from app.core.config import settings

@lru_cache(maxsize=1)
def get_vision_pipeline():
  """ Load CLIP/ViT only once for the zero-shot classification """
  return pipeline(
    task="zero-shot-image-classification",
    model=settings.vision_model_name, device=-1,
  )