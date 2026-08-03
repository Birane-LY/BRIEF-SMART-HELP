import json
import re
from groq import Groq
from app.core.config import settings


class DiagnosticService:

    def __init__(self):
        # Initialize the Groq cloud client
        api_key = getattr(settings, "groq_api_key", None)
        self.client = Groq(api_key=api_key)

    def evaluate(self, full_text: str | None, vision_results: list, rag_result: any) -> dict:
        # 1. Safely extract the first element from the vision results list
        vision_context = {"label": "Aucune image", "is_relevant": False, "score": 0.0}
        if vision_results and len(vision_results) > 0:
            top = vision_results[0]
            vision_context = {
                "label": top.get("label", ""),
                "is_relevant": top.get("is_relevant", False),
                "score": top.get("score", 0.0)
            }

        # 2. Format the RAG rule context
        rag_context = "Aucune règle spécifique trouvée dans la base de connaissances."
        if rag_result:
            rag_context = json.dumps(rag_result if isinstance(rag_result, dict) else rag_result.model_dump(), ensure_ascii=False)

        # 3. Ultimate System Prompt: Focus on the semantic relevance of the client description
        system_instruction = """Tu es l'expert en conformité et anti-fraude de SmartHelp. Ton rôle est d'analyser la réclamation d'un client et de juger sa pertinence par rapport au contexte e-commerce (produits physiques endommagés, non conformes ou non livrés).

        RÈGLES DE JUGEMENT STRICTES :
        1. ANALYSE EN PREMIER LA PERTINENCE DU TEXTE/AUDIO CLIENT :
           - Si la description du client est hors-sujet, poétique, vide, ou s'il s'agit d'un bruit/musique de fond (ex: "les cocos chantent...", "Sous-titres réalisés par..."), la réclamation est DIRECTEMENT INVALIDÉE.
           - TOUT HORS-SUJET DANS LE TEXTE OU L'AUDIO ENTRAÎNE UN STATUT "REFUSE" IMMÉDIAT, même si l'image fournie semble pertinente. Une description incohérente annule la validité du dossier.

        2. APPLICATION DE LA CHARTE DE DÉCISION (Si le texte est valide) :
           - "REMBOURSABLE" : Le texte est pertinent, la règle du RAG (R1.1/R1.2) correspond, ET l'image confirme le dommage (label: "damaged product", is_relevant: True).
           - "EN_ATTENTE_JUSTIFICATIFS" : Le texte décrit un vrai problème e-commerce légitime, mais l'image associée est hors-sujet ou invalide (is_relevant: False).
           - "A_VERIFIER" : Le texte décrit un problème, mais l'image montre un produit intact (label: "wrong item" sans dommage), créant une contradiction.

        CONSIGNE DE RÉDACTION : Ne fais JAMAIS référence aux exemples de cette consigne dans ta justification. Reste purement factuel sur ce que le client a envoyé.

        Tu dois répondre STRICTEMENT sous cette forme JSON, sans aucun texte autour :
        {
            "statut_propose": "REFUSE" ou "A_VERIFIER" ou "EN_ATTENTE_JUSTIFICATIFS" ou "REMBOURSABLE",
            "score_fiabilite": float entre 0.0 (fraude/hors-sujet) et 1.0 (conforme),
            "justification": "Une seule phrase claire et polie en français expliquant pourquoi la demande est acceptée, mise en attente ou rejetée."
        }"""

        # 4. User Prompt: Raw structured facts combined with the mandatory 'json' keyword constraint
        user_prompt = f"""
        Format your response strictly using a json object according to the system instructions.
        
        DONNÉES À ANALYSER :
        
        1. DESCRIPTION TEXTUELLE OU AUDIO DU CLIENT :
        "{full_text if full_text else 'Aucune description fournie.'}"

        2. RÈGLE TROUVÉE PAR LE RAG :
        {rag_context}

        3. ANALYSE DE L'IMAGE COMPLÉMENTAIRE :
        - Label de l'image : "{vision_context['label']}"
        - L'image montre-t-elle un produit e-commerce valide ? : {vision_context['is_relevant']}
        - Score de confiance de la vision : {vision_context['score']}
        """

        try:
            # Trigger Groq Cloud API inference
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            # Robust response parsing matching Groq structural outputs
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                content_text = choice.message.content if hasattr(choice, 'message') else choice['message']['content']
            else:
                content_text = response['choices'][0]['message']['content']

            result = json.loads(content_text)

            return {
                "statut_propose": result.get("statut_propose", "A_VERIFIER"),
                "score_fiabilite": round(float(result.get("score_fiabilite", 0.5)), 3),
                "warnings": [result.get("justification", "Analyse contextuelle effectuée.")]
            }

        except Exception as e:
            # Secure automated fallback to human review in case of upstream structural API failure
            return {
                "statut_propose": "A_VERIFIER",
                "score_fiabilite": 0.500,
                "warnings": [f"Bascule de sécurité. Erreur API Cloud : {str(e)}"]
            }


diagnostic_service = DiagnosticService()
