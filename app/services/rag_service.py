import os
import re
import yaml
import logging
import unicodedata
from app.schemas.ticket_response import RAGResult

logger = logging.getLogger(__name__)

# Only greetings and pure grammar are excluded.
# ALL BUSINESS-SPECIFIC WORDS (bad, received, parcel, condition, payment, etc.) HAVE BEEN REMOVED.
STOPWORDS = {
    "bonjour", "cordialement", "merci", "svp", "plait", "vous", "nous",
    "avez", "avoir", "cette", "cela", "donc", "alors", "bien", "tout",
    "toute", "fois", "jour", "date", "veux", "voudrais", "besoin", "aide",
    "aimerais", "suite", "cause", "raison", "depuis", "quand", "comment",
    "pourquoi", "encore", "aussi", "meme", "chose", "faire", "peux", "peut",
    "pouvez", "etre", "dans", "pour", "avec", "sans", "sur", "sous"
}


class RAGService:

    def __init__(self, relative_yaml_path: str = "/home/birane/BRIEF-SMART-HELP/ data/knowledge_base/politique_support.yaml"):
        """
        Initializes the RAG service by resolving the YAML file path
        relative to the root of the Python project.
        """
        # Project root (goes up 3 levels above the current file)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.yaml_path = os.path.join(base_dir, relative_yaml_path)
        self.rules = []
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Loads the knowledge base from the YAML file."""
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    self.rules = data.get("rules", [])
                logger.info(f"[RAG INFO] Base de connaissances chargée : {len(self.rules)} règles trouvées.")
            except Exception as e:
                logger.error(f"[RAG ERREUR] Erreur de lecture du fichier YAML : {str(e)}")
        else:
            logger.warning(f"[RAG ATTENTION] Fichier YAML introuvable : {self.yaml_path}")

    def _normalize(self, text: str) -> str:
        """Converts text to lowercase and removes accents."""
        if not text:
            return ""
        return "".join(
            c for c in unicodedata.normalize("NFD", text.lower())
            if unicodedata.category(c) != "Mn"
        )

    def _extract_tokens(self, text: str, min_len: int = 3) -> set[str]:
        """
        Extracts unique full words by cleaning punctuation
        and removing stopwords.
        """
        normalized_text = self._normalize(text)
        # Strict extraction of alphanumeric words
        words = re.findall(r'\b\w+\b', normalized_text)
        return {w for w in words if len(w) >= min_len and w not in STOPWORDS}

    def search(self, text: str) -> RAGResult | None:
        """
        Scans the submitted text against the YAML rules.
        """
        if not text or not self.rules:
            return None

        claim_normalized = self._normalize(text)
        claim_tokens = self._extract_tokens(text, min_len=3)

        scored_rules = []

        for rule in self.rules:
            matches = 0

            # 1. Explicit standard phrases (strong signal)
            for query in rule.get("retrieval_queries", []):
                normalized_query = self._normalize(query)
                if normalized_query and normalized_query in claim_normalized:
                    matches += 3

            # 2. Significant words from the title (full words)
            title_tokens = self._extract_tokens(rule.get("title", ""), min_len=3)
            matching_title_words = title_tokens.intersection(claim_tokens)
            matches += len(matching_title_words) * 2  # Increased weight for the title

            # 3. Significant words from the conditions (full words)
            for condition in rule.get("conditions", []):
                condition_tokens = self._extract_tokens(condition, min_len=3)
                matching_condition_words = condition_tokens.intersection(claim_tokens)
                matches += len(matching_condition_words)

            if matches > 0:
                scored_rules.append((matches, rule))

        if not scored_rules:
            return None

        # Sort by descending score
        scored_rules.sort(key=lambda item: item[0], reverse=True)
        best_matches, best_rule = scored_rules[0]
        second_best = scored_rules[1][0] if len(scored_rules) > 1 else 0

        logger.info(
            f"[RAG INFO] Règle retenue : {best_rule.get('id')} ({best_matches} pts) | "
            f"Deuxième : {second_best} pts"
        )

        # Minimum threshold and confidence margin
        if best_matches < 2 or (best_matches - second_best) < 1:
            return None

        confidence_score = min(0.60 + (best_matches * 0.05), 0.98)
        outcome = best_rule.get("outcome", {})

        return RAGResult(
            id=best_rule.get("id"),
            title=best_rule.get("title"),
            status=outcome.get("status"),
            action=outcome.get("action"),
            source=outcome.get("source"),
            conditions=best_rule.get("conditions", []),
            score=round(confidence_score, 2)
        )


rag_service = RAGService()
