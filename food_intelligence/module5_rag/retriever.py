from __future__ import annotations

from typing import Any, Dict, List, Optional
from food_intelligence.module4_knowledge_base import KnowledgeBaseRepository, KnowledgeDocument
from .vector_store import VectorStore


class RAGRetriever:
    """Retrieves relevant food knowledge for extracted products with fallback handling."""

    def __init__(self, kb_repo: KnowledgeBaseRepository | None = None, vector_store: VectorStore | None = None):
        self.kb_repo = kb_repo or KnowledgeBaseRepository()
        self.vector_store = vector_store or VectorStore()
        # Initialize vector store with KB documents
        docs = self.kb_repo.get_all_documents()
        self.vector_store.add_documents(docs)

    def retrieve_context_for_product(self, extracted_product: Dict[str, Any]) -> Dict[str, Any]:
        """Build RAG context from extracted product data.

        Returns retrieved knowledge documents and flags unknown items requiring LLM fallback.
        """
        ingredients = extracted_product.get("ingredients", [])
        additives = extracted_product.get("additives", [])
        allergens = extracted_product.get("allergens", {})

        retrieved_docs: List[Dict[str, Any]] = []
        unknown_ingredients: List[str] = []
        unknown_additives: List[str] = []

        # 1. Search for Additives
        for add in additives:
            raw_code = str(add.get("code") or add.get("raw_code") or "")
            canonical = str(add.get("canonical_name") or "")
            query = f"Additive INS {raw_code} {canonical}".strip()

            results = self.vector_store.similarity_search_with_score(query, k=1, doc_type="additive")
            matched = False
            if results:
                doc, score = results[0]
                if (raw_code and raw_code in doc.doc_id) or (canonical and canonical.lower() in doc.title.lower()) or score >= 0.5:
                    retrieved_docs.append({
                        "target": raw_code or canonical,
                        "type": "additive",
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "content": doc.content,
                        "similarity_score": round(score, 3),
                        "knowledge_source": "VectorDB_KnowledgeBase",
                    })
                    matched = True
            if not matched:
                unknown_additives.append(raw_code or canonical)

        # 2. Search for Ingredients
        for ing in ingredients:
            raw = str(ing.get("raw") or "")
            canonical = str(ing.get("canonical") or raw)
            query = f"Ingredient {canonical} {raw}".strip()

            results = self.vector_store.similarity_search_with_score(query, k=1, doc_type="ingredient")
            matched = False
            if results:
                doc, score = results[0]
                # Match if canonical name is explicitly in doc title/synonyms or score >= 0.85
                doc_title_lower = doc.title.lower()
                doc_synonyms = [s.lower() for s in doc.metadata.get("synonyms", [])]
                if (canonical and (canonical.lower() in doc_title_lower or canonical.lower() in doc_synonyms)) or score >= 0.85:
                    retrieved_docs.append({
                        "target": canonical,
                        "type": "ingredient",
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "content": doc.content,
                        "similarity_score": round(score, 3),
                        "knowledge_source": "VectorDB_KnowledgeBase",
                    })
                    matched = True
            if not matched:
                if canonical:
                    unknown_ingredients.append(canonical)

        # 3. Controlled Fallback Strategy
        fallback_required = bool(unknown_ingredients or unknown_additives)
        fallback_instructions = None
        if fallback_required:
            fallback_instructions = (
                "Some extracted ingredients/additives were not found in the verified Knowledge Base. "
                "The LLM (Ollama/Groq) should analyze them cautiously using general food chemistry knowledge, "
                "clearly marking them as 'LLM Extrapolated' rather than ground truth."
            )

        return {
            "document_id": extracted_product.get("document_id"),
            "grounded_context": retrieved_docs,
            "grounded_docs_count": len(retrieved_docs),
            "fallback_status": {
                "fallback_required": fallback_required,
                "unknown_ingredients": unknown_ingredients,
                "unknown_additives": unknown_additives,
                "instructions": fallback_instructions,
            },
        }
