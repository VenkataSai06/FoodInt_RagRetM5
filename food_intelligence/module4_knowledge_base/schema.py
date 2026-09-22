from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AdditiveKnowledge(BaseModel):
    code: str  # e.g., "330" or "E330" or "INS 330"
    canonical_name: str  # e.g., "Citric Acid"
    category: str  # e.g., "Acidity Regulator / Antioxidant"
    ins_number: Optional[str] = None
    e_number: Optional[str] = None
    description: str
    health_effects: List[str] = Field(default_factory=list)
    safety_rating: str = "Safe"  # Safe, Moderate, Caution, Avoid
    max_daily_intake_note: Optional[str] = None


class IngredientKnowledge(BaseModel):
    canonical_id: str  # e.g., "carrageenan"
    name: str  # e.g., "Carrageenan"
    synonyms: List[str] = Field(default_factory=list)
    category: str  # e.g., "Gelling Agent / Thickener"
    description: str
    allergens: List[str] = Field(default_factory=list)
    health_notes: List[str] = Field(default_factory=list)
    dietary_attributes: Dict[str, bool] = Field(default_factory=dict)  # e.g., {"vegan": True, "gluten_free": True}


class AllergenKnowledge(BaseModel):
    name: str  # e.g., "Peanuts", "Wheat", "Soy"
    synonyms: List[str] = Field(default_factory=list)
    severity_note: str
    common_sources: List[str] = Field(default_factory=list)


class KnowledgeDocument(BaseModel):
    doc_id: str
    doc_type: str  # "additive", "ingredient", "allergen", "nutrition_guideline"
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
