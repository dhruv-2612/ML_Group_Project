import os
import logging
import pandas as pd
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_chroma import Chroma

# Disable Chroma telemetry
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache() # Fix for some persistent client issues if any
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Local imports
try:
    import src.ml_analyzer as ml_analyzer
except ImportError:
    import ml_analyzer

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_vector_store(jobs_df: pd.DataFrame, persist_dir: str = "data/embeddings/chroma_db") -> Chroma:
    """
    Converts job postings into documents and stores them in a Chroma vector store.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found in environment.")
        raise ValueError("GEMINI_API_KEY is required.")

    if jobs_df.empty:
        logger.warning("Jobs DataFrame is empty. Cannot create vector store.")
        return None

    logger.info(f"Processing {len(jobs_df)} jobs into vector store...")
    
    documents = []
    for _, row in jobs_df.iterrows():
        # Combine fields into a single text block for embedding
        text_content = (
            f"Job Title: {row.get('job_title')}\n"
            f"Company: {row.get('company')}\n"
            f"Description: {row.get('job_description')}\n"
            f"Skills: {row.get('skills_text')}"
        )
        
        # Metadata for filtering and retrieval info
        metadata = {
            "source": "job_posting",
            "job_title": str(row.get('job_title', 'N/A')),
            "company": str(row.get('company', 'N/A')),
            "location": str(row.get('location', 'N/A')),
            "experience_level": str(row.get('experience_level', 'N/A')),
            "salary": str(row.get('salary_mid', 'N/A')),
            "cluster_id": int(row.get('cluster_id', -1)),
            "job_url": str(row.get('job_url', 'N/A'))
        }
        
        documents.append(Document(page_content=text_content, metadata=metadata))

    # Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)

    # Create and persist Vector Store in batches to avoid rate limits
    batch_size = 50
    vectorstore = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        logger.info(f"Indexing batch {i//batch_size + 1} ({len(batch)} docs)...")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name="job_postings_2026"
            )
        else:
            vectorstore.add_documents(batch)
            
        if i + batch_size < len(documents):
            logger.info("Waiting 30 seconds to avoid rate limits...")
            time.sleep(30)
    
    logger.info(f"Vector store created and persisted at {persist_dir}")
    return vectorstore

def add_career_guides_to_store(vectorstore: Chroma) -> Chroma:
    """
    Adds synthetic career guidance documents to the vector store.
    """
    logger.info("Adding synthetic career guides to vector store...")
    
    guides = [
        {
            "title": "Transitioning from Software Engineer to Product Manager",
            "category": "career_transition",
            "content": "Software Engineers (SDEs) are uniquely positioned to become excellent Product Managers because of their technical grounding. To transition successfully, you must shift your focus from 'how to build' to 'what to build' and 'why'. Key steps include: 1. Master product frameworks like CIRCLES or AARRR. 2. Develop user empathy by conducting user interviews. 3. Learn to communicate business value rather than technical implementation to stakeholders. 4. Bridge the gap by becoming a 'Technical PM' first. Focus on building a portfolio of side projects where you managed the entire lifecycle, not just the code."
        },
        {
            "title": "Essential Skills for Data Analysts in 2026",
            "category": "skills",
            "content": "The Data Analyst role in 2026 has evolved beyond just SQL and Excel. While these remain foundational, the market now demands: 1. Advanced Python for automation and predictive modeling. 2. Proficiency in Cloud Data Warehouses like Snowflake or BigQuery. 3. Strong storytelling skills to translate complex data into executive-ready insights. 4. Basic understanding of LLM integration for data cleaning and automated reporting. A top-tier analyst today acts as a strategic partner to the business, identifying growth opportunities before they appear in standard reports."
        },
        {
            "title": "Product Manager Interview Preparation Guide",
            "category": "interview_prep",
            "content": "PM interviews are notoriously multi-dimensional. Prepare for four main types of questions: 1. Product Sense: How would you improve Spotify? What's your favorite product and why? 2. Execution/Metrics: What would you do if Google Maps usage dropped by 10%? 3. Strategy: Should Amazon enter the healthcare space? 4. Behavioral: Tell me about a time you had a conflict with an engineer. Use the STAR method for behavioral and the CIRCLES method for design. Always identify the user persona and their pain points before proposing a solution."
        },
        {
            "title": "Salary Negotiation for MBA Graduates",
            "category": "career_advice",
            "content": "Negotiating your first post-MBA salary requires data and poise. Start by researching market rates specifically for MBA roles in hubs like Bangalore or Mumbai. Never be the first to give a number; instead, ask for the range allocated for the role. When you do negotiate, focus on the 'Total Package'—joining bonuses, ESOPs, and performance variables can often be increased even if the base salary is fixed. Highlight your specific value-add, such as previous technical experience or niche domain expertise from your MBA projects."
        },
        {
            "title": "Breaking into Management Consulting from Tech",
            "category": "career_transition",
            "content": "Management consulting firms like McKinsey, BCG, and Bain increasingly value tech backgrounds. To break in: 1. Polish your mental math and case study solving skills. 2. Translate your tech achievements into business impact (e.g., 'Reduced latency' becomes 'Increased conversion rate by 5%'). 3. Network with alumni who made the same jump. 4. Demonstrate 'structured thinking'—the ability to break complex problems into Mutually Exclusive, Collectively Exhaustive (MECE) components. Consulting is about solving the CEO's problems, not the CTO's."
        }
    ]
    
    extra_topics = [
        ("AI for Productivity in 2026", "skills"),
        ("Effective Stakeholder Management", "soft_skills"),
        ("Mastering A/B Testing", "analytics"),
        ("The Role of a Growth PM", "roles"),
        ("Financial Modeling for Non-Finance Managers", "domain"),
        ("User Research Methods", "skills"),
        ("Building a Personal Brand on LinkedIn", "career_advice"),
        ("Remote Work Success Strategies", "career_advice"),
        ("Agile vs. Kanban in Modern Tech", "product_tools"),
        ("The Future of FinTech in India", "domain")
    ]
    
    for title, cat in extra_topics:
        guides.append({
            "title": title,
            "category": cat,
            "content": f"This guide covers {title}. It provides essential insights into {cat} for professionals looking to excel in the 2026 job market. Practical steps include staying updated with latest trends, networking, and hands-on practice."
        })

    documents = []
    for g in guides:
        metadata = {
            "source": "career_guide",
            "doc_type": "career_guide",
            "title": g["title"],
            "category": g["category"],
            "relevance_score": 0.9
        }
        documents.append(Document(page_content=g["content"], metadata=metadata))
        
    vectorstore.add_documents(documents)
    logger.info(f"Added {len(documents)} career guides to vector store.")
    return vectorstore

def load_vector_store(persist_dir: str = "data/embeddings/chroma_db") -> Chroma:
    """
    Loads the existing Chroma vector store.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found in environment.")
        raise ValueError("GEMINI_API_KEY is required.")
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="job_postings_2026"
    )
    
    logger.info(f"Vector store loaded from {persist_dir}")
    return vectorstore

def create_rag_chain(vectorstore: Chroma) -> Tuple[RetrievalQA, Any]:
    """
    Creates a RetrievalQA chain using the vector store and Gemini LLM.
    """
    llm = GoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GEMINI_API_KEY, temperature=0.3)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    prompt_template = """
    You are an expert career advisor with access to job market data from 2026.
    
    Context from job database:
    {context}
    
    Question: {question}
    
    Provide a comprehensive answer with:
    1. Data-driven insights (cite specific statistics from context)
    2. Actionable recommendations
    3. Relevant job examples or patterns
    4. If salary-related, provide ranges with context
    
    Format using markdown with emojis for readability.
    Answer:
    """
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain, retriever

def query_job_intelligence(question: str, rag_chain: RetrievalQA, retriever: Any = None) -> Dict[str, Any]:
    """
    Queries the RAG system and returns the answer with source documents.
    """
    try:
        response = rag_chain.invoke(question)
        return {
            "answer": response['result'],
            "source_documents": response['source_documents']
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {"answer": "I encountered an error processing your request.", "source_documents": []}

def format_response_with_sources(response: str, source_docs: List[Document]) -> str:
    """
    Formats the LLM response with a list of sources.
    """
    formatted_response = response + "\n\n### 📚 Sources & Relevant Jobs:\n"
    
    seen_sources = set()
    for i, doc in enumerate(source_docs):
        # Create a unique key to dedup sources
        source_key = doc.metadata.get('job_url', '') or doc.metadata.get('title', '')
        if not source_key or source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        
        if doc.metadata.get('source') == 'job_posting':
            title = doc.metadata.get('job_title', 'Job')
            company = doc.metadata.get('company', 'Unknown')
            url = doc.metadata.get('job_url', '#')
            formatted_response += f"- **[{title} at {company}]({url})**\n"
        elif doc.metadata.get('source') == 'career_guide':
            title = doc.metadata.get('title', 'Guide')
            formatted_response += f"- 📖 Guide: **{title}**\n"
            
    return formatted_response

def get_personalized_recommendations(user_profile: Dict[str, Any], vectorstore: Chroma) -> Dict[str, Any]:
    """
    Generates a personalized career roadmap based on user profile.
    """
    current_role = user_profile.get('current_role', 'Professional')
    target_role = user_profile.get('target_role', 'target role')
    years_exp = user_profile.get('years_experience', 0)
    
    query = f"Career path from {current_role} to {target_role} for someone with {years_exp} years of experience. Key skills needed and transition strategy."
    
    chain, _ = create_rag_chain(vectorstore)
    result = query_job_intelligence(query, chain)
    
    formatted_answer = format_response_with_sources(result['answer'], result['source_documents'])
    
    return {
        "roadmap": formatted_answer,
        "raw_answer": result['answer'],
        "sources": result['source_documents']
    }

def _detect_role_and_intent(question: str, jobs_df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Helper to extract a job role from the query and detect intent.
    Intents: 'skills', 'salary', 'career_path', 'general'
    """
    question_lower = question.lower()
    
    # 1. Detect Intent
    intent = 'general'
    if any(k in question_lower for k in ['skill', 'learn', 'tech stack', 'tool', 'language']):
        intent = 'skills'
    elif any(k in question_lower for k in ['salary', 'pay', 'compensation', 'earning', 'ctc', 'lpa']):
        intent = 'salary'
    elif any(k in question_lower for k in ['career', 'path', 'grow', 'become', 'transition', 'future']):
        intent = 'career_path'
        
    # 2. Extract Role (Simple heuristic: matching unique titles in DB)
    # Get top 20 most frequent words/bigrams from titles to check against? 
    # Or simply check if known roles are substrings.
    found_role = None
    if not jobs_df.empty:
        # Create a set of common roles from data for matching
        # Normalize titles to simple forms for matching (e.g., "Product Manager" -> "product manager")
        unique_titles = jobs_df['job_title'].dropna().unique()
        
        # Sort by length desc to match "Senior Product Manager" before "Product Manager"
        unique_titles = sorted(unique_titles, key=len, reverse=True)
        
        for title in unique_titles:
            if title.lower() in question_lower:
                found_role = title
                break
                
        # Fallback: Check for generic common roles if exact match failed
        if not found_role:
            common_roles = ['product manager', 'data analyst', 'business analyst', 'data scientist', 'software engineer']
            for role in common_roles:
                if role in question_lower:
                    found_role = role
                    break
                    
    return found_role, intent

def enhanced_query_with_ml(question: str, rag_chain: RetrievalQA, jobs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Combines ML insights with RAG to answer user questions.
    """
    role, intent = _detect_role_and_intent(question, jobs_df)
    ml_context = "⚠️ Note: Current market data volume is too low to provide statistical ML insights for this specific query."
    
    try:
        if intent == 'skills' and role:
            logger.info(f"Detected intent 'skills' for role '{role}'. Running ML analysis...")
            stats = ml_analyzer.analyze_skills_by_role(jobs_df, role_filter=role)
            if not stats.empty:
                top_skills = stats.head(10)
                # Format as a markdown list for the LLM
                skill_list = "\n".join([f"- **{row['skills_list']}**: {row['percentage']:.1f}% frequency" for _, row in top_skills.iterrows()])
                ml_context = (
                    f"### 📊 ML Skills Analysis for '{role}'\n"
                    f"Based on {len(jobs_df)} analyzed job postings:\n"
                    f"{skill_list}\n"
                )
                
        elif intent == 'salary' and role:
            logger.info(f"Detected intent 'salary' for role '{role}'.")
            role_df = jobs_df[jobs_df['job_title'].str.contains(role, case=False, na=False)]
            if not role_df.empty and 'salary_mid' in role_df.columns:
                avg_sal = role_df['salary_mid'].mean()
                min_sal = role_df['salary_mid'].min()
                max_sal = role_df['salary_mid'].max()
                ml_context = (
                    f"### 💰 ML Salary Data for '{role}'\n"
                    f"- **Average Salary**: {avg_sal:.2f} LPA\n"
                    f"- **Range**: {min_sal:.2f} - {max_sal:.2f} LPA\n"
                    f"*(Data derived from {len(role_df)} postings)*\n"
                )
                
        elif intent == 'career_path' and role:
            logger.info(f"Detected intent 'career_path' for role '{role}'.")
            # Find the most common cluster for this role
            role_df = jobs_df[jobs_df['job_title'].str.contains(role, case=False, na=False)]
            if not role_df.empty and 'cluster_id' in role_df.columns:
                # Get mode cluster
                common_cluster = role_df['cluster_id'].mode()[0]
                cluster_stats = ml_analyzer.get_cluster_characteristics(jobs_df, common_cluster)
                
                if cluster_stats:
                    top_skills = ", ".join(cluster_stats.get('top_skills', [])[:5])
                    top_companies = ", ".join(cluster_stats.get('top_companies', [])[:3])
                    ml_context = (
                        f"### 🚀 Career Insights (Cluster {common_cluster})\n"
                        f"This role is part of a cluster characterized by:\n"
                        f"- **Key Skills**: {top_skills}\n"
                        f"- **Top Hiring Companies**: {top_companies}\n"
                        f"- **Avg Cluster Salary**: {cluster_stats.get('avg_salary_lpa', 'N/A')} LPA\n"
                    )
            else:
                ml_context = "Could not identify a specific career cluster for this role."

    except Exception as e:
        logger.error(f"Error generating ML context: {e}")
        ml_context = "Error retrieving ML insights."

    # Construct the final prompt for the LLM
    # We embed the ML context into the 'question' field of the RAG chain, 
    # as the chain puts retrieved docs into 'context'.
    augmented_query = (
        f"ML Analysis Results:\n{ml_context}\n\n"
        f"User Question: {question}\n\n"
        f"Instruction: Based on both the quantitative ML analysis above and the real job examples provided in the context, answer the user's question."
    )
    
    result = query_job_intelligence(augmented_query, rag_chain)
    result['intent'] = intent
    result['detected_role'] = role
    result['ml_insights'] = ml_context
    
    return result

if __name__ == "__main__":
    # Test run
    try:
        # Load clustered data for testing
        df_path = "data/processed/job_clusters.csv"
        if os.path.exists(df_path):
            df = pd.read_csv(df_path).head(50) 
            
            # 1. Create store
            vs = create_vector_store(df) # Ensure store exists
            
            # 2. Add guides
            add_career_guides_to_store(vs)
            
            # 3. Create Chain
            chain, retriever = create_rag_chain(vs)
            
            # 4. Test Query
            q = "What skills do I need for a Product Manager role in Bangalore?"
            logger.info(f"Querying: {q}")
            result = query_job_intelligence(q, chain)
            print("\n--- LLM Response ---")
            print(result['answer'])
            
            # 5. Test Personalized Recs
            profile = {
                "current_role": "Software Engineer",
                "target_role": "Product Manager",
                "years_experience": 3
            }
            logger.info("Generating personalized recommendations...")
            recs = get_personalized_recommendations(profile, vs)
            print("\n--- Personalized Roadmap ---")
            print(recs['roadmap'][:500] + "...")
            
        else:
            logger.warning("Clustered data not found. Run ml_analyzer.py first.")
            
    except Exception as e:
        logger.error(f"RAG test failed: {e}")
        import traceback
        traceback.print_exc()
