import json
import re
from typing import Any

from app.core.config import settings

try:
    from groq import Groq
except ImportError: 
    Groq = None  


class DiagnosticService:

    def __init__(self):
        api_key = (getattr(settings, "groq_api_key", "") or "").strip()
        self.client = Groq(api_key=api_key) if Groq is not None and api_key else None

    @staticmethod
    def _fallback(reason: str) -> dict:
        return {
            "statut_propose": "A_VERIFIER",
            "score_fiabilite": 0.500,
            "warnings": [reason],
        }

    def evaluate(self, full_text: str | None, vision_results: list, rag_result: Any) -> dict:
        if self.client is None:
            return self._fallback(
                "Diagnostic automatique indisponible : configurez GROQ_API_KEY pour activer l’analyse Groq."
            )

        image_fournie = bool(vision_results and len(vision_results) > 0)
        vision_context = {"label": "Aucune image", "is_relevant": False, "score": 0.0}
        if image_fournie:
            top = vision_results[0]
            vision_context = {
                "label": top.get("label", ""),
                "is_relevant": top.get("is_relevant", False),
                "score": top.get("score", 0.0)
            }

        rag_context = "Aucune règle spécifique trouvée dans la base de connaissances."
        if rag_result:
            rag_context = json.dumps(rag_result if isinstance(rag_result, dict) else rag_result.model_dump(), ensure_ascii=False)

        system_instruction = """Tu es l'expert en conformité et anti-fraude de SmartHelp. Tu dois suivre un ALGORITHME STRICT EN 2 ÉTAPES SÉQUENTIELLES, dans cet ordre exact, sans jamais les mélanger.

        ══════════════════════════════════════════
        ÉTAPE 1 — VALIDITÉ DU TEXTE (à faire EN PREMIER, TOUJOURS, INDÉPENDAMMENT de toute information sur l'image)
        ══════════════════════════════════════════
        Avant même de regarder si une image a été fournie ou non, réponds à cette seule question :
        "Ce texte décrit-il une réclamation e-commerce compréhensible (un problème avec un produit commandé, livré, endommagé, non conforme ou non reçu) ?"

        Signaux d'un texte INVALIDE (liste non exhaustive) :
        - Paroles de chanson, poésie, comptine, discours récité, formule de politesse répétée sans contenu.
        - Bruit ambiant capté par erreur, silence, transcription incompréhensible ou sans structure de phrase logique.
        - Texte qui ne mentionne AUCUN élément e-commerce (aucun produit, aucune commande, aucune livraison, aucun dommage).
        - Sujet manifestement hors du cadre du support client (politique, actualité, anecdote personnelle sans lien).

        → SI LE TEXTE EST INVALIDE, regarde uniquement si une image pertinente confirme un dommage :
           - image fournie, produit e-commerce pertinent et dommage visible (is_relevant: true) :
             statut = "EN_ATTENTE_JUSTIFICATIFS", score_fiabilite entre 0.30 et 0.60 ; la photo constitue
             un indice, mais le client doit fournir une description exploitable de la commande et du problème.
           - aucune image pertinente confirmant un dommage : statut = "REFUSE", score_fiabilite entre 0.0 et 0.15.
        Un texte quelconque ne doit donc jamais provoquer un refus lorsque la photo est pertinente et montre
        clairement un dommage.

        → SI LE TEXTE EST VALIDE (même bref, même sans détail technique, du moment qu'il décrit un vrai problème
        e-commerce compréhensible) : passe à l'étape 2 ci-dessous.

        ══════════════════════════════════════════
        ÉTAPE 2 — UNIQUEMENT SI LE TEXTE A ÉTÉ JUGÉ VALIDE À L'ÉTAPE 1
        (Pour un texte invalide avec image de dommage, appliquer directement le statut EN_ATTENTE_JUSTIFICATIFS
        défini à l’étape 1 et ne pas appliquer les règles de remboursement.)
        ══════════════════════════════════════════
        Regarde maintenant la variable "images_fournies" et le contexte vision :

        a) images_fournies = false (aucune image envoyée) :
           → statut = "EN_ATTENTE_JUSTIFICATIFS". Ce n'est pas suspect : le client n'a pas encore transmis de photo.

        b) images_fournies = true, mais l'image est hors-sujet (reçu, capture d'écran, mauvais article, emballage
           sans rapport avec le texte) :
           → statut = "EN_ATTENTE_JUSTIFICATIFS".

        c) images_fournies = true, et l'image confirme le dommage décrit dans le texte (label lié à un défaut,
           is_relevant: true) :
           - une règle RAG correspond → statut = "REMBOURSABLE"
           - aucune règle RAG ne correspond → statut = "A_VERIFIER"

        d) images_fournies = true, et l'image montre un produit "neuf ou en bon état" (aucun dommage visible),
           EN CONTRADICTION avec un texte qui affirme un dommage :
           → statut = "A_VERIFIER". La contradiction entre le texte et l'image doit être signalée explicitement
           dans la justification.

        CONSIGNE DE RÉDACTION : Ne fais jamais référence aux exemples de cette consigne dans ta justification.
        Reste factuel. Ta justification doit toujours être cohérente avec le statut choisi et avec l'étape qui
        l'a produit (mentionne si le rejet vient d'un problème de texte, ou si l'attente vient d'un problème
        d'image).

        Tu dois répondre STRICTEMENT sous cette forme JSON, sans aucun texte autour :
        {
            "statut_propose": "REFUSE" ou "A_VERIFIER" ou "EN_ATTENTE_JUSTIFICATIFS" ou "REMBOURSABLE",
            "score_fiabilite": float entre 0.0 (fraude/hors-sujet) et 1.0 (conforme),
            "justification": "Une seule phrase claire et polie en français expliquant pourquoi la demande est acceptée, mise en attente ou rejetée."
        }"""

        user_prompt = f"""
        Format your response strictly using a json object according to the system instructions.
        
        DONNÉES À ANALYSER :
        
        1. DESCRIPTION TEXTUELLE OU AUDIO DU CLIENT :
        IMPORTANT : un texte libre, incohérent ou sans rapport avec l’e-commerce est considéré comme invalide ;
        toutefois, si l’image fournie est pertinente et montre un dommage, le statut obligatoire est
        EN_ATTENTE_JUSTIFICATIFS, jamais REFUSE.
        "{full_text if full_text else 'Aucune description fournie.'}"

        2. RÈGLE TROUVÉE PAR LE RAG :
        {rag_context}

        3. ANALYSE DE L'IMAGE COMPLÉMENTAIRE :
        - images_fournies : {image_fournie}
        - Label de l'image : "{vision_context['label']}"
        - L'image montre-t-elle un produit e-commerce valide ? : {vision_context['is_relevant']}
        - Score de confiance de la vision : {vision_context['score']}
        """

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )

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
            return {
                "statut_propose": "A_VERIFIER",
                "score_fiabilite": 0.500,
                "warnings": [f"Bascule de sécurité. Erreur API Cloud : {str(e)}"]
            }


diagnostic_service = DiagnosticService()