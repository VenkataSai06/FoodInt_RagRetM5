from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from .schema import KnowledgeDocument


class KnowledgeBaseRepository:
    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_dir = data_dir
        self.documents: List[KnowledgeDocument] = []
        self._load_all()

    def _load_all(self) -> None:
        self.documents.clear()
        self._load_additives()
        self._load_ingredients()
        self._load_allergens()

    def _load_additives(self) -> None:
        path = os.path.join(self.data_dir, "additives.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                code = item.get("code")
                name = item.get("canonical_name")
                desc = item.get("description", "")
                category = item.get("category", "")
                effects = "; ".join(item.get("health_effects", []))
                safety = item.get("safety_rating", "Safe")

                text = f"Additive {name} (INS/E-{code}): Category {category}. Description: {desc}. Safety: {safety}. Health considerations: {effects}"
                doc = KnowledgeDocument(
                    doc_id=f"additive_{code}",
                    doc_type="additive",
                    title=f"Additive: {name} (INS {code})",
                    content=text,
                    metadata=item,
                )
                self.documents.append(doc)

    def _load_ingredients(self) -> None:
        path = os.path.join(self.data_dir, "ingredients.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                cid = item.get("canonical_id")
                name = item.get("name")
                synonyms = ", ".join(item.get("synonyms", []))
                desc = item.get("description", "")
                notes = "; ".join(item.get("health_notes", []))

                text = f"Ingredient {name} (Synonyms: {synonyms}). Category: {item.get('category')}. Description: {desc}. Health notes: {notes}"
                doc = KnowledgeDocument(
                    doc_id=f"ingredient_{cid}",
                    doc_type="ingredient",
                    title=f"Ingredient: {name}",
                    content=text,
                    metadata=item,
                )
                self.documents.append(doc)

    def _load_allergens(self) -> None:
        path = os.path.join(self.data_dir, "allergens.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                name = item.get("name")
                sources = ", ".join(item.get("common_sources", []))
                severity = item.get("severity_note", "")

                text = f"Allergen {name}. Severity: {severity}. Common sources: {sources}"
                doc = KnowledgeDocument(
                    doc_id=f"allergen_{name.lower()}",
                    doc_type="allergen",
                    title=f"Allergen: {name}",
                    content=text,
                    metadata=item,
                )
                self.documents.append(doc)

    def get_all_documents(self) -> List[KnowledgeDocument]:
        return self.documents

    def get_document_by_id(self, doc_id: str) -> KnowledgeDocument | None:
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None
