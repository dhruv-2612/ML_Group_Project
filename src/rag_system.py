import os
import logging
import pandas as pd
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Local imports
try:
    from src import ml_analyzer
except (ImportError, ModuleNotFoundError) as e:
    # If we are running from within src, or src is not a package
    try:
        import ml_analyzer
    except ImportError:
        # Re-raise the original error if it wasn't just a path issue
        raise e

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use a consistent model name from environment or fallback to gemini-2.5-flash
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

def create_vector_store(jobs_df: pd.DataFrame, persist_dir: str = "data/embeddings/chroma_db") -> Chroma:
    """
    Converts job postings into documents and stores them in a Chroma vector store.
    Uses incremental indexing to avoid re-processing existing jobs.
    """
    if not HF_TOKEN:
        logger.error("HUGGINGFACE_API_TOKEN not found in environment.")
        raise ValueError("HUGGINGFACE_API_TOKEN is required.")

    if jobs_df.empty:
        logger.warning("Jobs DataFrame is empty. Cannot create vector store.")
        return None

    # Initialize Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Initialize/Load Vector Store
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="job_postings_2026",
        client_settings=Settings(anonymized_telemetry=False)
    )

    # INCREMENTAL INDEXING: Get existing job URLs to avoid duplicates
    try:
        existing_data = vectorstore.get()
        existing_urls = set()
        if existing_data and 'metadatas' in existing_data:
            existing_urls = {m.get('job_url') for m in existing_data['metadatas'] if m.get('job_url')}
        logger.info(f"Found {len(existing_urls)} existing jobs in vector store.")
    except Exception as e:
        logger.warning(f"Could not retrieve existing IDs: {e}. Starting fresh...")
        existing_urls = set()

    new_documents = []
    for _, row in jobs_df.iterrows():
        url = str(row.get('job_url', ''))
        # Skip if already exists or no URL
        if not url or url in existing_urls:
            continue
            
        text_content = (
            f"Job Title: {row.get('job_title')}\n"
            f"Company: {row.get('company')}\n"
            f"Description: {row.get('job_description')}\n"
            f"Skills: {row.get('skills_text')}"
        )
        
        metadata = {
            "source": "job_posting",
            "job_title": str(row.get('job_title', 'N/A')),
            "company": str(row.get('company', 'N/A')),
            "location": str(row.get('location', 'N/A')),
            "experience_level": str(row.get('experience_level', 'N/A')),
            "salary": str(row.get('salary_mid', 'N/A')),
            "cluster_id": int(row.get('cluster_id', -1)),
            "job_url": url
        }
        
        new_documents.append(Document(page_content=text_content, metadata=metadata, id=url))

    if not new_documents:
        logger.info("No new unique jobs to add to the vector store.")
        return vectorstore

    # BATCHING & RATE LIMITING: Process in smaller batches with a safe delay
    # Limit: 100 RPM. We'll do 5 requests (batches) per minute to be extremely safe.
    batch_size = 10 
    logger.info(f"Processing {len(new_documents)} NEW jobs into vector store...")
    
    for i in range(0, len(new_documents), batch_size):
        batch = new_documents[i:i + batch_size]
        logger.info(f"Indexing batch {i//batch_size + 1} ({len(batch)} new docs)...")
        
        vectorstore.add_documents(batch)
            
        # 15-second delay between batches of 10 = ~40 documents per minute.
        # This is well under the 100 RPM limit and handles network overhead.
        if i + batch_size < len(new_documents):
            logger.info("Waiting 5 seconds to stay safely under 100 RPM limit...")
            time.sleep(5)
    
    logger.info(f"Vector store update complete at {persist_dir}")
    return vectorstore

def load_vector_store(persist_dir: str = "data/embeddings/chroma_db") -> Chroma:
    """
    Loads the existing Chroma vector store.
    """
    if not HF_TOKEN:
        logger.error("HUGGINGFACE_API_TOKEN not found in environment.")
        raise ValueError("HUGGINGFACE_API_TOKEN is required.")
        
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="job_postings_2026",
        client_settings=Settings(anonymized_telemetry=False)
    )
    
    logger.info(f"Vector store loaded from {persist_dir}")
    return vectorstore

def create_rag_chain(vectorstore: Chroma) -> Tuple[RetrievalQA, Any]:
    """
    Creates a RetrievalQA chain using the vector store and Gemini LLM.
    """
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0.3)
    
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

def extract_role_with_llm(question: str) -> Optional[str]:
    """Uses Gemini to extract the job role from a query when heuristics fail."""
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0)
        prompt = f"Extract the specific job role or professional title from this career-related question: '{question}'. Return ONLY the job title in lowercase (e.g., 'product manager'). If no clear job role is found, return 'None'."
        response = llm.invoke(prompt)
        content_response = response.content
        
        # Handle list content if returned by LLM
        if isinstance(content_response, list):
            content = " ".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content_response])
        else:
            content = content_response
        
        content = content.strip().lower()
        if "none" in content or len(content) > 50:
            return None
        return content
    except Exception as e:
        logger.error(f"LLM Role Extraction failed: {e}")
        return None

def generate_dynamic_career_guide(role: str, vectorstore: Chroma) -> str:
    """Generates a high-quality career guide using LLM and saves it to vector store."""
    logger.info(f"Generating dynamic career guide for: {role}")
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0.3)
        prompt = f"""
        Create a comprehensive career guide for the role of '{role.title()}'.
        Include:
        1. Essential Technical Skills (2026 market context).
        2. Soft Skills & Leadership requirements.
        3. Typical Career Path/Progression.
        4. Interview Tips specific to this role.
        5. Future Outlook for this role in the AI-driven economy.
        
        Format as a professional knowledge document.
        """
        guide_content_response = llm.invoke(prompt).content
        
        # Handle list content if returned by LLM
        if isinstance(guide_content_response, list):
            guide_content = " ".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in guide_content_response])
        else:
            guide_content = guide_content_response
        
        # Save to vector store for future RAG context
        metadata = {
            "source": "dynamic_career_guide",
            "doc_type": "career_guide",
            "title": f"Complete Guide to {role.title()}",
            "category": "generated_guide",
            "role": role,
            "relevance_score": 1.0,
            "generated_date": datetime.now().isoformat()
        }
        vectorstore.add_documents([Document(page_content=guide_content, metadata=metadata)])
        logger.info(f"Successfully generated and saved guide for {role}")
        return guide_content
    except Exception as e:
        logger.error(f"Dynamic guide generation failed: {e}")
        return ""

import json

def get_known_roles(jobs_df: pd.DataFrame) -> List[str]:
    """Retrieves a list of unique roles from the database and persistent storage."""
    roles_file = "data/processed/known_roles.json"
    roles = set()
    
    # 1. Start with high-frequency roles from current data
    if not jobs_df.empty:
        # Get unique titles, lowercase them
        unique_titles = [str(t).lower() for t in jobs_df['job_title'].dropna().unique()]
        roles.update(unique_titles)
        
    # 2. Add from persistent discovered roles
    if os.path.exists(roles_file):
        try:
            with open(roles_file, 'r') as f:
                saved_roles = json.load(f)
                roles.update(saved_roles)
        except: pass
        
    # 3. Add base hardcoded ones
    common_base = [
        'product manager', 'data analyst', 'business analyst', 'data scientist', 
        'software engineer', 'management consultant', 'marketing manager',
        'marketing analyst', 'ux designer', 'ui designer', 'blockchain developer',
        'full stack developer', 'backend developer', 'frontend developer',
        'project manager', 'scrum master', 'cloud architect', 'devops engineer',
        'investment banker', 'financial analyst', 'hr manager', 'sales manager'
    ]
    roles.update(common_base)
    
    # Return sorted by length desc to match "Senior Data Scientist" before "Data Scientist"
    return sorted(list(roles), key=len, reverse=True)

def save_discovered_role(role: str):
    """Saves a new role discovered by LLM to the dynamic dictionary."""
    roles_file = "data/processed/known_roles.json"
    os.makedirs(os.path.dirname(roles_file), exist_ok=True)
    
    roles = []
    if os.path.exists(roles_file):
        try:
            with open(roles_file, 'r') as f:
                roles = json.load(f)
        except: roles = []
        
    if role not in roles:
        roles.append(role)
        try:
            with open(roles_file, 'w') as f:
                json.dump(roles, f)
            logger.info(f"Discovered role '{role}' saved to dynamic dictionary.")
        except Exception as e:
            logger.warning(f"Failed to save discovered role: {e}")

def _detect_role_and_intent(question: str, jobs_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], str]:
    """
    Helper to extract job role and location from the query and detect intent.
    Uses a dynamic dictionary, regex patterns, and LLM fallback.
    """
    question_lower = question.lower()
    
    # 1. Detect Intent
    intent = 'general'
    if any(k in question_lower for k in ['skill', 'learn', 'tech stack', 'tool', 'language', 'requirement']):
        intent = 'skills'
    elif any(k in question_lower for k in ['salary', 'pay', 'compensation', 'earning', 'ctc', 'lpa', 'money']):
        intent = 'salary'
    elif any(k in question_lower for k in ['career', 'path', 'grow', 'become', 'transition', 'future', 'guide']):
        intent = 'career_path'
        
    # 2. Extract Role
    found_role = None
    
    # Step A: Dynamic Dictionary (Fastest)
    all_known_roles = get_known_roles(jobs_df)
    for role in all_known_roles:
        if role in question_lower:
            found_role = role
            break
            
    # Step B: Regex Pattern Matching (New Discovery without LLM)
    if not found_role:
        # Patterns like "for a marketing analyst", "become a data engineer", "salary of a product manager"
        role_patterns = [
            r'(?:for\s+a|as\s+a|become\s+a|of\s+a|to\s+be\s+a)\s+([a-z\s]+?)(?:\s+in|\s+at|\?|$|\s+role|\s+position)',
            r'([a-z\s]+?)\s+(?:jobs|role|position|career|analyst|manager|developer|engineer|consultant|specialist)'
        ]
        for pattern in role_patterns:
            match = re.search(pattern, question_lower)
            if match:
                potential_role = match.group(1).strip()
                # Simple validation: roles should be 2-4 words usually
                word_count = len(potential_role.split())
                if 1 <= word_count <= 4:
                    found_role = potential_role
                    logger.info(f"Regex discovered potential role: {found_role}")
                    break
            
    # Step C: LLM Fallback (Last Resort)
    if not found_role:
        logger.info("Heuristics and Regex failed. Calling LLM fallback...")
        found_role = extract_role_with_llm(question)
        if found_role:
            logger.info(f"LLM discovered new niche role: {found_role}")
            save_discovered_role(found_role)

    # 3. Extract Location
    found_location = None
    common_locations = ['bangalore', 'mumbai', 'pune', 'hyderabad', 'delhi', 'noida', 'gurgaon', 'chennai', 'remote', 'kolkata', 'ahmedabad']
    for loc in common_locations:
        if loc in question_lower:
            found_location = loc
            break
            
    return found_role, found_location, intent

# Data sufficiency thresholds
MIN_JOBS_PER_ROLE = 15  # Increased for better statistical significance
MIN_JOBS_PER_LOCATION = 15 # Ensure local insights are valid

def check_data_sufficiency(role: Optional[str], location: Optional[str], jobs_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Checks if we have enough data for a given role and/or location.
    Returns: (is_sufficient, reason)
    """
    if not role:
        return True, "No specific role detected."
        
    # Filter by role
    role_matches = jobs_df[jobs_df['job_title'].str.contains(role, case=False, na=False)]
    
    if len(role_matches) < MIN_JOBS_PER_ROLE:
        return False, f"Low data for '{role}' ({len(role_matches)} found, need {MIN_JOBS_PER_ROLE})"
        
    if location:
        # Correctly check for location within a list of locations
        loc_matches = role_matches[role_matches['location'].apply(
            lambda loc_list: isinstance(loc_list, list) and any(location.lower() in item.lower() for item in loc_list)
        )]
        if len(loc_matches) < MIN_JOBS_PER_LOCATION:
            return False, f"Low data for '{role}' in '{location}' ({len(loc_matches)} found, need {MIN_JOBS_PER_LOCATION})"
            
    return True, f"Sufficient data available.{len(role_matches)} matches for role, {len(loc_matches) if location else 'N/A'} matches for location."

def enhanced_query_with_ml(question: str, rag_chain: RetrievalQA, jobs_df: pd.DataFrame, forced_intent: str = None) -> Dict[str, Any]:
    """
    Combines ML insights with RAG to answer user questions.
    Generates dynamic career guides if missing.
    """
    # Force rebuild vector store to pick up new logic if called through pipeline
    # (Actually we just need the vectorstore object from the chain to add guides)
    vectorstore = rag_chain.retriever.vectorstore

    role, location, detected_intent = _detect_role_and_intent(question, jobs_df)
    
    # Use forced intent if provided (e.g. from UI buttons), otherwise use detected
    intent = forced_intent if forced_intent else detected_intent
    
    is_sufficient, reason = check_data_sufficiency(role, location, jobs_df)
    
    # --- AUTONOMOUS SCRAPING IF INSUFFICIENT ---
    if not is_sufficient and role:
        logger.info(f"Data sufficiency check failed: {reason}. Triggering autonomous scraping...")
        try:
            # Import here to avoid circular dependency at module level
            from src.pipeline import run_full_pipeline
            
            # Determine scraping parameters
            scrape_locs = [location] if location else ["Bangalore", "Mumbai"]
            
            # Run the pipeline (this updates CSVs and Vector Store)
            new_jobs = run_full_pipeline([role], scrape_locs, num_pages=2, use_selenium=True)
            
            if new_jobs > 0:
                logger.info(f"Successfully scraped {new_jobs} new jobs. Reloading data...")
                # Reload dataframe to reflect new data
                df_path = "data/processed/job_clusters.csv"
                if os.path.exists(df_path):
                    jobs_df = pd.read_csv(df_path)
                    # Re-check sufficiency with new data
                    is_sufficient, reason = check_data_sufficiency(role, location, jobs_df)
            else:
                logger.warning("Scraping completed but no new unique jobs were added.")
                
        except Exception as e:
            logger.error(f"Autonomous scraping failed: {e}")
    else:
        logger.info(f"Data sufficiency check passed: {reason}")
    # --- DYNAMIC GUIDE GENERATION ---
    dynamic_guide = ""
    if intent == 'career_path' and role:
        # Check if we already have a generated guide for this role
        existing_guides = vectorstore.similarity_search(
            f"Complete Guide to {role.title()}", 
            k=1, 
            filter={"$and": [{"source": "dynamic_career_guide"}, {"role": role}]}
        )
        
        if not existing_guides:
            dynamic_guide = generate_dynamic_career_guide(role, vectorstore)
        else:
            dynamic_guide = existing_guides[0].page_content
            logger.info(f"Using existing dynamic guide for {role}")

    ml_context = "⚠️ Note: Current market data volume is too low to provide statistical ML insights for this specific query."
    
    try:
        # Filter data by role AND location for strict location-based insights
        role_df = jobs_df.copy()
        if role:
            role_df = role_df[role_df['job_title'].str.contains(role, case=False, na=False)]
        
        loc_context_str = ""
        if location:
            # Correctly check for location within a list of locations
            loc_matches = role_df[role_df['location'].apply(
                lambda loc_list: isinstance(loc_list, list) and any(location.lower() in item.lower() for item in loc_list)
            )]
            if len(loc_matches) >= MIN_JOBS_PER_LOCATION:
                role_df = loc_matches
                loc_context_str = f" in '{location.title()}'"
            else:
                loc_context_str = f" (Global/India context used due to limited data in '{location.title()}')"

        role_count = len(role_df)

        if intent == 'skills' and role:
            logger.info(f"Detected intent 'skills' for role '{role}'{loc_context_str}. Running ML analysis...")
            
            stats = ml_analyzer.analyze_skills_by_role(role_df) # Use filtered role_df
            if not stats.empty:
                top_skills = stats.head(10)
                skill_list = "\n".join([f"- **{row['skill_name']}**: {row['percentage']:.1f}% frequency" for _, row in top_skills.iterrows()])
                ml_context = (
                    f"### 📊 ML Skills Analysis for '{role.title()}'{loc_context_str}\n"
                    f"Analysis based on {role_count} fresh (last 14 days) job postings matching this role and location:\n"
                    f"{skill_list}\n"
                )
                
        elif intent == 'salary' and role:
            logger.info(f"Detected intent 'salary' for role '{role}'{loc_context_str}.")
            
            # Use the robust statistical benchmark function
            sal_stats = ml_analyzer.get_salary_benchmark(jobs_df, role_filter=role, location_filter=location)
            
            if sal_stats['count'] > 0:
                ml_context = (
                    f"### 💰 Salary Market Statistics for '{role.title()}'{loc_context_str}\n"
                    f"Based on **{sal_stats['count']}** fresh job postings (last 14 days):\n"
                    f"- **Median Salary**: ₹{sal_stats['median']} LPA\n"
                    f"- **Market Range**: ₹{sal_stats['min']} - ₹{sal_stats['max']} LPA\n"
                    f"- **Mean Average**: ₹{sal_stats['mean']} LPA\n"
                )
            else:
                ml_context = f"Insufficient salary data to provide specific statistics for '{role}'{loc_context_str}."
                
        elif intent == 'career_path' and role:
            logger.info(f"Detected intent 'career_path' for role '{role}'{loc_context_str}.")
            
            if not role_df.empty and 'cluster_id' in role_df.columns:
                common_cluster = role_df['cluster_id'].mode()[0]
                cluster_stats = ml_analyzer.get_cluster_characteristics(jobs_df, common_cluster)
                
                if cluster_stats:
                    top_skills = ", ".join(cluster_stats.get('top_skills', [])[:5])
                    top_companies = ", ".join(cluster_stats.get('top_companies', [])[:3])
                    ml_context = (
                        f"### 🚀 Career Insights for '{role.title()}'{loc_context_str}\n"
                        f"Analysis of {role_count} fresh postings within this cluster context:\n"
                        f"- **Key Skills**: {top_skills}\n"
                        f"- **Top Hiring Companies**: {top_companies}\n"
                        f"- **Avg Cluster Salary**: ₹{cluster_stats.get('avg_salary_lpa', 'N/A')} LPA\n"
                    )

    except Exception as e:
        logger.error(f"Error generating ML context: {e}")
        ml_context = "Error retrieving ML insights."

    guide_hint = "\n(A new expert career guide has been dynamically generated and added to the database for this role.)" if dynamic_guide and "generated" in dynamic_guide else ""

    augmented_query = (
        f"ML Analysis Results:\n{ml_context}\n\n"
        f"Dynamic Career Guide Knowledge:\n{dynamic_guide[:2000]}\n\n"
        f"User Question: {question}\n\n"
        f"Instruction: Based on the quantitative ML analysis, the dynamically generated career guide, and real job examples, answer the user's question.{guide_hint}"
    )
    
    result = query_job_intelligence(augmented_query, rag_chain)
    result['intent'] = intent
    result['detected_role'] = role
    result['detected_location'] = location
    result['ml_insights'] = ml_context
    result['needs_scrape'] = not is_sufficient
    result['scrape_reason'] = reason
    
    return result
