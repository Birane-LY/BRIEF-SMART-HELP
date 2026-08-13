import io
import logging
from PIL import Image
from app.core.config import settings
from app.models.vision_model import get_vision_pipeline

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(self):
        self.pipeline = get_vision_pipeline()

        self.candidate_labels = [
            "damaged product",
           
            "receipt or invoice",
            "wrong item",
            "screenshot of transaction",
            "packaging issue"
        ]

        # Translated labels containing strategic keywords
        # for optimal RAG matching
        self.label_translation = {
            "damaged product": "produit abime ou casse",
           
            "receipt or invoice": "facture ou recu de paiement",
            "wrong item": "mauvais article ou erreur de produit",
            "screenshot of transaction": "capture d ecran de transaction",
            "packaging issue": "emballage ou colis endommage"
        }

    async def analyze_images(self, images_bytes: list[bytes]) -> list[dict]:
        """
        Analyzes a list of image  and returns unified
        classification results in French.
        """
        if not images_bytes:
            return []

        results = []
        for idx, image_bytes in enumerate(images_bytes):
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                prediction = self.pipeline(image, candidate_labels=self.candidate_labels)

                if prediction and len(prediction) > 0:
                    top_prediction = prediction[0]
                    raw_label = top_prediction["label"]
                    score = float(top_prediction["score"])

                    french_label = self.label_translation.get(raw_label, raw_label)
                    is_relevant = score >= getattr(settings, "vision_threshold", 0.5)

                    results.append({
                        "raw_label": raw_label,
                        "label": french_label,
                        "score": round(score, 4),
                        "is_relevant": is_relevant
                    })
                else:
                    logger.warning(f"[VISION] Pipeline a renvoyé une prédiction vide pour l'image index {idx}")

            except Exception as e:
                logger.error(f"[VISION] Impossible d'analyser l'image à l'index {idx} : {str(e)}")

        return results


vision_service = VisionService()
