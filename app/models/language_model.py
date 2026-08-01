from functools import lru_cache
from transformers import pipeline
from app.core.config import settings

@lru_cache(maxsize=1)
def get_language_pipeline():
  """ Load AfroXLM-R to extract all textual characteristics referring to Wolof in the claim. """
  return pipeline(
    task="feature-extraction",
    model=settings.text_model_name,
    tokenizer=settings.text_model_name, device=-1,
  )