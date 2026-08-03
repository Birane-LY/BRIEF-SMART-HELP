import io
from PIL import Image
from app.core.config import settings
from app.models.vision_model import get_vision_pipeline


class VisionService:
    def __init__(self):
        # Load the zero-shot image classification model into memory
        self.pipeline = get_vision_pipeline()
        
        # Internal candidate labels required by the Hugging Face vision model
        self.candidate_labels = [
            "damaged product",
            "receipt or invoice",
            "wrong item",
            "screenshot of transaction",
            "packaging issue"
        ]
        
        # Translation mapping layer to align English predictions with French labels
        self.label_translation = {
            "damaged product": "produit cassé ou brisé",
            "receipt or invoice": "facture ou reçu",
            "wrong item": "mauvais article reçu",
            "screenshot of transaction": "capture d'écran de transaction",
            "packaging issue": "emballage endommagé"
        }

    async def analyze_images(self, images_bytes: list[bytes]) -> list[dict]:
        """
        Analyzes a list of image byte streams and returns unified French classification results.
        """
        if not images_bytes:
            return []

        results = []
        for idx, image_bytes in enumerate(images_bytes):
            try:
                # Convert raw incoming byte streams into standard RGB PIL Image objects
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Execute zero-shot image classification inference
                prediction = self.pipeline(image, candidate_labels=self.candidate_labels)
                
                if prediction and len(prediction) > 0:
                    top_prediction = prediction[0]
                    raw_label = top_prediction["label"]
                    score = float(top_prediction["score"])
                    
                    # Map the English model output label to its standardized French equivalent
                    french_label = self.label_translation.get(raw_label, raw_label)
                    
                    # Evaluate structural relevance using dynamic global threshold parameters
                    is_relevant = score >= settings.vision_threshold
                    
                    results.append({
                        "label": french_label, # Now perfectly translated to French
                        "score": round(score, 4),
                        "is_relevant": is_relevant
                    })
                else:
                    print(f"[ERREUR VISION] Le pipeline a renvoyé une prédiction vide pour l'image index {idx}")
                    
            except Exception as e:
                print(f"[ERREUR VISION] Impossible d'analyser le fichier binaire à l'index {idx} : {str(e)}")

        return results

vision_service = VisionService()
