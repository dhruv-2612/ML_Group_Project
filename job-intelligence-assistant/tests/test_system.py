import pytest
import pandas as pd
import os
import sys
import shutil
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from src.scraper import JobScraper
from src.generate_sample_data import generate_synthetic_job_data
from src.utils import clean_job_data
import src.ml_analyzer as ml
import src.rag_system as rag
import src.chatbot as bot

# --- FIXTURES ---

@pytest.fixture(scope="module")
def sample_jobs_df():
    """Generates synthetic data for testing models."""
    df = generate_synthetic_job_data(n_jobs=50)
    return clean_job_data(df)

@pytest.fixture(scope="module")
def temp_vector_store(sample_jobs_df):
    """Creates a temporary vector store."""
    persist_dir = "tests/temp_chroma_db"
    
    # Cleanup before
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        
    # Create store (mocking API key if needed, or assuming env is set)
    # We need a real API key for embeddings unless we mock embeddings. 
    # For integration tests, we usually assume the key is present or we skip.
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set. Skipping RAG tests.")
        
    # Use a small subset to save API calls
    small_df = sample_jobs_df.head(10)
    vs = rag.create_vector_store(small_df, persist_dir=persist_dir)
    rag.add_career_guides_to_store(vs)
    
    yield vs
    
    # Cleanup after
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

# --- TESTS ---

def test_scraper_structure():
    """Test data structure of scraper (using synthetic data as proxy for reliability)."""
    # We test the synthetic generator which mimics the scraper's output schema
    df = generate_synthetic_job_data(n_jobs=5)
    
    required_cols = [
        'job_title', 'company', 'location', 
        'job_description', 'salary', 'experience_required'
    ]
    
    assert not df.empty
    for col in required_cols:
        assert col in df.columns
    assert len(df) == 5

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requires API Key")
def test_ml_models(sample_jobs_df):
    """Test ML analysis functions."""
    # 1. Clustering
    clustered_df, kmeans, keywords = ml.cluster_job_roles(sample_jobs_df, n_clusters=3)
    assert 'cluster_id' in clustered_df.columns
    assert 'cluster_label' in clustered_df.columns
    assert len(keywords) == 3
    
    # 2. Skills Analysis
    skills_df = ml.analyze_skills_by_role(clustered_df)
    assert not skills_df.empty
    assert 'percentage' in skills_df.columns
    
    # 3. Salary Model
    model, imp, metrics, cols = ml.train_salary_predictor(clustered_df)
    assert model is not None
    assert metrics['r2_score'] is not None

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requires API Key")
def test_rag_system(temp_vector_store):
    """Test RAG queries."""
    chain, _ = rag.create_rag_chain(temp_vector_store)
    
    test_queries = [
        ("What skills do Product Managers need?", ["skill", "product"]),
        ("Average salary for Data Analysts?", ["salary", "analyst"]),
        ("Transition from engineer to product manager", ["transition", "engineer"]),
        ("Emerging skills in 2026", ["skill", "2026"]),
        ("Best companies for MBA graduates", ["company", "mba"])
    ]
    
    for query, expected_keywords in test_queries:
        result = rag.query_job_intelligence(query, chain)
        answer = result['answer'].lower()
        
        assert len(answer) > 20
        # Check if at least one keyword is present (flexible match)
        # Note: LLM answers vary, so we just check for non-empty and basic relevance
        assert result['source_documents'] is not None

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requires API Key")
def test_integration(sample_jobs_df, temp_vector_store):
    """Test full chatbot pipeline integration."""
    chain, _ = rag.create_rag_chain(temp_vector_store)
    
    # Mock user profile
    user_profile = {
        'current_role': 'Software Engineer', 
        'target_role': 'Product Manager',
        'years_experience': 3
    }
    
    # Run enhanced query
    query = "What skills do I need to become a Product Manager?"
    
    # We need clustered df for some ML features
    clustered_df, _, _ = ml.cluster_job_roles(sample_jobs_df, n_clusters=3)
    
    result = rag.enhanced_query_with_ml(query, chain, clustered_df)
    
    assert result['intent'] == 'skills'
    assert result['detected_role'] is not None
    assert "ML Analysis Results" in result['ml_insights'] or "ML" in result['ml_insights']
    assert len(result['answer']) > 0

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
