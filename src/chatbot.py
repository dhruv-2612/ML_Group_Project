import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# LangChain imports
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Local imports
try:
    import src.rag_system as rag
    import src.ml_analyzer as ml
except ImportError:
    import rag_system as rag
    import ml_analyzer as ml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_conversation_memory() -> ConversationBufferMemory:
    """
    Initializes conversation memory to store the last 5 exchanges.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer', # Important for ConversationalRetrievalChain
        k=5
    )
    return memory

def create_conversational_chain(vectorstore: Any, memory: ConversationBufferMemory) -> ConversationalRetrievalChain:
    """
    Creates a ConversationalRetrievalChain using the vector store and memory.
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found in environment.")

    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=os.getenv("GEMINI_API_KEY"), 
        temperature=0.3
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Custom condense prompt to ensure context is preserved
    condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Include key entities like job roles or locations if they are implied by the conversation history.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=True
    )
    
    return chain

def _generate_follow_ups(intent: str, role: Optional[str]) -> List[str]:
    """Helper to generate context-aware follow-up questions."""
    role_str = role if role else "this role"
    
    if intent == 'skills':
        return [
            f"What is the salary range for {role_str}?",
            f"How do I learn these skills quickly?",
            f"Show me entry-level jobs for {role_str}"
        ]
    elif intent == 'salary':
        return [
            f"What skills pay the highest for {role_str}?",
            f"Does a Master's degree increase salary for {role_str}?",
            f"Compare {role_str} salaries in Bangalore vs Mumbai"
        ]
    elif intent == 'career_path':
        return [
            f"What are the top certifications for {role_str}?",
            f"What is the typical career progression after {role_str}?",
            f"Mock interview questions for {role_str}"
        ]
    elif intent == 'interview':
        return [
            f"Give me a sample answer for a behavioral question",
            f"What technical concepts should I review?",
            f"How to negotiate salary for {role_str}"
        ]
    else:
        return [
            "Tell me about high-demand jobs in 2026",
            "How can I switch my career?",
            "What skills are trending right now?"
        ]

def process_user_query(
    query: str, 
    chain: ConversationalRetrievalChain, 
    user_profile: Dict[str, Any], 
    jobs_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Main handler for user queries in the chat interface.
    Detects intent, enriches with ML, and retrieves response.
    """
    # Reuse detection logic from rag_system if available, otherwise implement simple version
    # Since we are in chatbot.py, let's implement the specific logic requested
    
    query_lower = query.lower()
    detected_intent = "general"
    
    # Intent Detection
    if any(k in query_lower for k in ["skill", "learn", "required", "stack", "tool"]):
        detected_intent = "skills"
    elif any(k in query_lower for k in ["salary", "pay", "compensation", "ctc", "earning"]):
        detected_intent = "salary"
    elif any(k in query_lower for k in ["become", "transition", "career path", "roadmap", "grow"]):
        detected_intent = "career_path"
    elif any(k in query_lower for k in ["interview", "questions", "prepare", "test"]):
        detected_intent = "interview"
        
    # Extract Role (simple match)
    detected_role = None
    if not jobs_df.empty:
        # Sort by length to match longest first
        titles = sorted(jobs_df['job_title'].dropna().unique(), key=len, reverse=True)
        for title in titles:
            if title.lower() in query_lower:
                detected_role = title
                break
    
    # ML Enrichment
    ml_context = ""
    try:
        if detected_intent == 'skills' and detected_role:
            stats = ml.analyze_skills_by_role(jobs_df, role_filter=detected_role)
            if not stats.empty:
                top = stats.head(8)
                ml_context = f"ML INSIGHTS: Top skills for {detected_role}: {', '.join(top['skills_list'].tolist())}. "
        
        elif detected_intent == 'salary' and detected_role:
            role_df = jobs_df[jobs_df['job_title'].str.contains(detected_role, case=False, na=False)]
            if not role_df.empty and 'salary_mid' in role_df.columns:
                avg = role_df['salary_mid'].mean()
                ml_context = f"ML INSIGHTS: Average salary for {detected_role} is {avg:.2f} LPA. "
                
    except Exception as e:
        logger.error(f"ML enrichment failed: {e}")
        
    # Augment Query
    augmented_query = f"{ml_context}
Question: {query}" if ml_context else query
    
    # Execute Chain
    try:
        response = chain.invoke({"question": augmented_query})
        answer = response.get('answer', "I couldn't generate an answer.")
        source_docs = response.get('source_documents', [])
        
        # Format sources simply
        formatted_sources = rag.format_response_with_sources("", source_docs).replace("### 📚 Sources & Relevant Jobs:
", "")
        
        return {
            "answer": answer,
            "intent": detected_intent,
            "ml_insights": ml_context,
            "source_jobs": formatted_sources, # Just text representation for now
            "follow_up_suggestions": _generate_follow_ups(detected_intent, detected_role)
        }
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        return {
            "answer": "I encountered an error processing your request. Please try again.",
            "intent": "error",
            "ml_insights": "",
            "source_jobs": [],
            "follow_up_suggestions": []
        }

def generate_personalized_career_plan(
    user_profile: Dict[str, Any], 
    jobs_df: pd.DataFrame, 
    rag_chain: Any # Using RetrievalQA chain here ideally
) -> str:
    """
    Generates a comprehensive markdown career plan.
    """
    current_role = user_profile.get('current_role', 'Professional')
    target_role = user_profile.get('target_role', 'Target Role')
    current_skills = user_profile.get('skills', [])
    years_exp = user_profile.get('years_experience', 0)
    
    report = f"# 🚀 Personalized Career Roadmap: {current_role} ➝ {target_role}

"
    
    # 1. Skill Gap Analysis
    try:
        target_stats = ml.analyze_skills_by_role(jobs_df, role_filter=target_role)
        if not target_stats.empty:
            top_required = target_stats.head(15)['skills_list'].tolist()
            user_skills_lower = [s.lower() for s in current_skills]
            
            missing = [s for s in top_required if s.lower() not in user_skills_lower]
            existing = [s for s in top_required if s.lower() in user_skills_lower]
            
            report += "## 📊 Skill Gap Analysis
"
            report += f"**✅ You Have:** {', '.join(existing) if existing else 'None of the top skills yet'}

"
            report += f"**⚠️ Critical Gaps:** {', '.join(missing[:5])}

"
            
            # Timeline estimation
            est_months = max(2, len(missing) * 1.5) # Heuristic: 1.5 months per major skill
            report += f"**⏱️ Estimated Transition Timeline:** {est_months:.0f} - {est_months+3:.0f} months

"
            
    except Exception as e:
        report += f"Could not perform deep skill analysis: {e}

"
    
    # 2. LLM Synthesis for Strategy
    try:
        prompt = (
            f"Create a step-by-step learning path for a {current_role} with {years_exp} years experience "
            f"wanting to become a {target_role}. "
            f"They already know {', '.join(current_skills)}. "
            f"Focus on practical projects and high-value certifications."
        )
        
        # We assume rag_chain here is a RetrievalQA or similar that has .invoke
        # Use a simple direct invocation if possible, or query_job_intelligence
        llm_response = rag.query_job_intelligence(prompt, rag_chain)
        strategy = llm_response['answer']
        
        report += "## 🗺️ Strategic Learning Path
"
        report += strategy + "

"
        
    except Exception as e:
        report += f"Could not generate strategic path: {e}

"
        
    return report

if __name__ == "__main__":
    # Simple test
    pass
