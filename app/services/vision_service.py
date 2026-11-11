
import io
import logging

from PIL import Image
from app.core.config import settings
from app.models.vision_model import get_vision_pipeline

logger = logging.getLogger(__name__)

class VisionService:
    def __init__(self):
        self._pipeline = None

        self.candidate_labels = [
            "damaged product",
            "receipt or invoice",
            "wrong item",
            "screenshot of transaction",
            "packaging issue",
        ]

        self.label_translation = {
            "damaged product": "produit abime ou casse",
            "receipt or invoice": "facture ou recu de paiement",
            "wrong item": "mauvais article ou erreur de produit",
            "screenshot of transaction": "capture d ecran de transaction",
            "packaging issue": "emballage ou colis endommage",
        }

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = get_vision_pipeline()

        return self._pipeline

    async def analyze_images(self, images_bytes: list[bytes]) -> list[dict]:
        if not images_bytes:
            return []

        results = []

        try:
            pipeline = self._get_pipeline()

            for idx, image_bytes in enumerate(images_bytes):
                try:
                    image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                    prediction = pipeline(
                        image,
                        candidate_labels=self.candidate_labels,
                    )

                    if prediction:
                        top = prediction[0]
                        raw_label = top["label"]
                        score = float(top["score"])

                        results.append({
                            "raw_label": raw_label,
                            "label": self.label_translation.get(
                                raw_label, raw_label
                            ),
                            "score": round(score, 4),
                            "is_relevant": score >= getattr(
                                settings, "vision_threshold", 0.5
                            ),
                        })

                except Exception as error:
                    logger.error(
                        "[VISION] Erreur image %s : %s",
                        idx,
                        error,
                    )

        except Exception as error:
            logger.error(
                "[VISION] Chargement du modèle impossible : %s",
                error,
            )

        return results

vision_service = VisionService()
