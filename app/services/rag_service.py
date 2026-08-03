import os
import yaml
from app.schemas.ticket_response import RAGResult

class RAGService:

    def __init__(self, yaml_path: str = "data/knowledge_base/politique_support.yaml"):
        # Resolve the configuration file path dynamically relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.yaml_path = os.path.join(base_dir, yaml_path)
        self.rules = []
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Parse configuration parameters from internal knowledge base YAML file."""
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self.rules = data.get("rules", [])
            except Exception as e:
                print(f"[ERREUR RAG] Impossible de charger ou d'analyser le fichier YAML : {str(e)}")
        else:
            print(f"[ATTENTION RAG] Le fichier de la base de connaissances est introuvable au chemin : {self.yaml_path}")

    def search(self, text: str) -> RAGResult | None:
        """Scan text using simple keyword mapping to forward the correct rule to the LLM context."""
        if not text or not self.rules:
            return None
            
        text_lower = text.lower()
        
        for rule in self.rules:
            if rule.get("id") == "R1.1" and ("fissur" in text_lower or "cass" in text_lower):
                outcome = rule.get("outcome", {})
                return RAGResult(
                    id=rule.get("id"),
                    title=rule.get("title"),
                    status=outcome.get("status"),
                    action=outcome.get("action"),
                    source=outcome.get("source"),
                    conditions=rule.get("conditions", []),
                    score=0.90
                )
        return None

rag_service = RAGService()
