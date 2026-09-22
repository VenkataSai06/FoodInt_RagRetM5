import pytest
from food_intelligence.module4_knowledge_base import KnowledgeBaseRepository
from food_intelligence.module5_rag import RAGRetriever, VectorStore, LightweightEmbedder
from food_intelligence.pipeline import run_food_pipeline


def test_knowledge_base_loading():
    repo = KnowledgeBaseRepository()
    docs = repo.get_all_documents()
    assert len(docs) > 0, "Knowledge Base should load documents."

    doc_ids = [d.doc_id for d in docs]
    assert "additive_330" in doc_ids
    assert "ingredient_carrageenan" in doc_ids
    assert "allergen_gluten" in doc_ids


def test_vector_store_retrieval():
    repo = KnowledgeBaseRepository()
    vs = VectorStore()
    vs.add_documents(repo.get_all_documents())

    results = vs.similarity_search_with_score("INS 330 Citric Acid", k=1, doc_type="additive")
    assert len(results) == 1
    doc, score = results[0]
    assert "Citric Acid" in doc.title
    assert score > 0.3


def test_rag_retriever_controlled_fallback():
    retriever = RAGRetriever()

    # Product with a known additive (330) and an unknown novel ingredient ("Extract of Cosmic Herb X")
    sample_product = {
        "document_id": "test_product",
        "ingredients": [
            {"raw": "Water", "canonical": "water"},
            {"raw": "Extract of Cosmic Herb X", "canonical": "extract of cosmic herb x"},
        ],
        "additives": [
            {"code": "330", "raw_code": "INS 330", "canonical_name": "Citric Acid"}
        ],
    }

    retrieved = retriever.retrieve_context_for_product(sample_product)
    assert retrieved["grounded_docs_count"] > 0
    fallback = retrieved["fallback_status"]
    assert fallback["fallback_required"] is True
    assert "extract of cosmic herb x" in fallback["unknown_ingredients"]
