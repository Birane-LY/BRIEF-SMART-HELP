"""Recherche sémantique dans la base de connaissances support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.schemas.ticket_response import RAGResult

logger = logging.getLogger(__name__)


class RAGService:
    """Indexe les règles YAML et les recherche par similarité cosinus."""

    def __init__(self, relative_yaml_path: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        configured_path = relative_yaml_path or settings.knowledge_base_path
        path = Path(configured_path)
        self.yaml_path = path if path.is_absolute() else project_root / path
        self.rules: list[dict[str, Any]] = []
        self._documents: list[str] = []
        self._model: Any = None
        self._index: Any = None
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        if not self.yaml_path.is_file():
            logger.warning("[RAG] Base de connaissances introuvable : %s", self.yaml_path)
            return
        try:
            with self.yaml_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            self.rules = data.get("rules", [])
            self._documents = [self._rule_to_document(rule) for rule in self.rules]
            logger.info("[RAG] Base chargée : %d règles depuis %s", len(self.rules), self.yaml_path)
        except (OSError, yaml.YAMLError) as exc:
            logger.error("[RAG] Impossible de lire %s : %s", self.yaml_path, exc)

    @staticmethod
    def _rule_to_document(rule: dict[str, Any]) -> str:
        queries = " ".join(rule.get("retrieval_queries", []))
        conditions = " ".join(rule.get("conditions", []))
        return f"{rule.get('title', '')}. {queries}. {conditions}".strip()

    def _ensure_index(self) -> bool:
        if self._index is not None:
            return True
        if not self._documents:
            return False
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model_name)
            embeddings = self._model.encode(
                self._documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._index = faiss.IndexFlatIP(embeddings.shape[1])
            self._index.add(embeddings)
            logger.info("[RAG] Index FAISS construit : %d documents", len(self._documents))
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.error("[RAG] Dépendances ou modèle d'embeddings indisponibles : %s", exc)
            return False

    def search(self, text: str) -> RAGResult | None:
        """Retourne la règle la plus proche si elle dépasse le seuil configuré."""
        if not text or not self._ensure_index():
            return None

        query_embedding = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores, indices = self._index.search(query_embedding, max(1, settings.rag_top_k))
        best_score = float(scores[0][0])
        best_index = int(indices[0][0])
        if best_index < 0 or best_score < settings.rag_similarity_threshold:
            logger.info("[RAG] Aucun résultat au-dessus du seuil : %.3f", best_score)
            return None

        rule = self.rules[best_index]
        outcome = rule.get("outcome", {})
        logger.info("[RAG] Règle retenue : %s (similarité %.3f)", rule.get("id"), best_score)
        return RAGResult(
            id=rule.get("id", ""),
            title=rule.get("title", ""),
            status=outcome.get("status", "A_VERIFIER"),
            action=outcome.get("action", ""),
            source=outcome.get("source", ""),
            conditions=rule.get("conditions", []),
            score=round(best_score, 3),
        )


rag_service = RAGService()
