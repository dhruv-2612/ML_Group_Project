import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from typing import Dict, Any

# Fix for asyncio event loop in certain environments
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Local imports
import sys
project_root = os.path.join(os.getcwd(), 'job-intelligence-assistant')
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src import rag_system as rag
    from src import ml_analyzer as ml
    from src.utils import clean_job_data
except ImportError as e:
    st.error(f"Import Error: {e}. Project root: {project_root}")
    # Fallback for different deployment layouts
    try:
        import src.rag_system as rag
        import src.ml_analyzer as ml
        from src.utils import clean_job_data
    except ImportError:
        st.error("Critical: Could not find 'src' module in sys.path")

# --- Page Configuration ---
st.set_page_config(
    page_title="Job Intelligence Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .job-card {
        background-color: #ffffff;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        background-color: #E3F2FD;
        color: #1565C0;
        border-radius: 15px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'analyzed_profile' not in st.session_state:
    st.session_state.analyzed_profile = False

# --- Resource Loading ---
@st.cache_resource
def load_data_and_models():
    """Load cleaned data and ML models."""
    try:
        # Load main dataset
        df_path = "data/processed/job_clusters.csv"
        if not os.path.exists(df_path):
            df_path = "data/processed/cleaned_jobs.csv"
            
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            if df.empty:
                st.error("The processed data file is empty. Please run the scraping pipeline with valid keywords.")
                return None, None
            # Ensure safe literal eval for lists if they are strings
            if 'skills_list' in df.columns:
                 df['skills_list'] = df['skills_list'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        else:
            st.error("Data file not found. Please run the pipeline first.")
            return None, None

        # Load Vector Store
        # Check if persist dir exists
        if os.path.exists("data/embeddings/chroma_db"):
            vectorstore = rag.load_vector_store()
        else:
            st.warning("Vector store not found. Creating one now (this may take a minute)...")
            vectorstore = rag.create_vector_store(df)
            rag.add_career_guides_to_store(vectorstore)
            
        # Create Chain
        chain, retriever = rag.create_rag_chain(vectorstore)
        
        return df, chain
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

# Load resources
jobs_df, rag_chain = load_data_and_models()

# --- Sidebar ---
with st.sidebar:
    st.title("🎯 Career Intelligence")
    st.caption("AI-Powered Job Market Insights")
    
    st.markdown("### 👤 Your Profile")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Resume (Optional)", type=['pdf', 'txt'])
    if uploaded_file:
        st.success("Resume uploaded! (Parsing pending)")
    
    # Inputs
    current_role = st.text_input("Current Role", value="MBA Student")
    
    target_role = st.selectbox(
        "Target Role",
        ["Product Manager", "Data Analyst", "Business Analyst", "Data Scientist", "Management Consultant", "Marketing Manager"]
    )
    
    experience = st.slider("Years of Experience", 0, 15, 2)
    
    # Common skills for convenience
    common_skills = [
        "Python", "SQL", "Excel", "Tableau", "Power BI", "Machine Learning", 
        "Product Management", "Agile", "Communication", "Marketing", "Strategy",
        "JIRA", "AWS", "Figma"
    ]
    
    skills = st.multiselect("Your Skills", common_skills, default=["Excel", "Communication"])
    
    locations = st.multiselect(
        "Preferred Locations", 
        ["Bangalore", "Mumbai", "Pune", "Hyderabad", "Delhi/NCR", "Remote"],
        default=["Bangalore", "Mumbai"]
    )
    
    if st.button("Analyze Profile", type="primary"):
        st.session_state.user_profile = {
            "current_role": current_role,
            "target_role": target_role,
            "years_experience": experience,
            "skills": skills,
            "locations": locations
        }
        st.session_state.analyzed_profile = True
        st.success("Profile Updated!")

# --- Main Content ---
st.title("Job Market Intelligence Assistant")

if not jobs_df is None:
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "📊 Market Insights", "🎯 Skills Gap", "💼 Job Matches"])

    # --- TAB 1: Chatbot ---
    with tab1:
        st.markdown("### Ask about skills, salaries, or career paths")
        
        # Display history
        for msg in st.session_state.conversation_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources & Details"):
                        st.markdown(msg["sources"])

        # Chat Input
        if prompt := st.chat_input("Ex: What skills do Product Managers need in Bangalore?"):
            # User message
            st.session_state.conversation_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing job market data..."):
                    try:
                        # 1. Get Response
                        response_data = rag.enhanced_query_with_ml(prompt, rag_chain, jobs_df)
                        answer = response_data['answer']
                        
                        # 2. Format Sources
                        sources_text = rag.format_response_with_sources("", response_data['source_documents'])
                        # Remove the prepended newline/header from the formatter if present to just get list
                        if "📚 Sources" in sources_text:
                            sources_text = sources_text.split("### 📚 Sources & Relevant Jobs:\n")[1]

                        # Display
                        st.markdown(answer)
                        
                        # Append to history
                        st.session_state.conversation_history.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources_text
                        })
                        
                        # Show debug info about ML usage
                        if response_data.get('ml_insights'):
                            with st.expander("🔍 ML Insights Used"):
                                st.info(response_data['ml_insights'])

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    # --- TAB 2: Market Insights ---
    with tab2:
        st.header(f"Market Insights for {st.session_state.user_profile.get('target_role', 'Selected Role')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Required Skills")
            target = st.session_state.user_profile.get('target_role', 'Product Manager')
            skill_stats = ml.analyze_skills_by_role(jobs_df, role_filter=target)
            
            if not skill_stats.empty:
                fig_skills = px.bar(
                    skill_stats.head(10),
                    x='percentage',
                    y='skills_list',
                    orientation='h',
                    title=f"Top Skills for {target}",
                    labels={'skills_list': 'Skill', 'percentage': 'Frequency (%)'},
                    color='percentage',
                    color_continuous_scale='Blues'
                )
                fig_skills.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_skills, use_container_width=True)
            else:
                st.info("No data available for this role.")

        with col2:
            st.subheader("Salary Distribution")
            role_df = jobs_df[jobs_df['job_title'].str.contains(target, case=False, na=False)]
            if not role_df.empty and 'salary_mid' in role_df.columns:
                fig_sal = px.box(
                    role_df, 
                    y='salary_mid', 
                    points="all",
                    title=f"Salary Range for {target}",
                    labels={'salary_mid': 'Annual Salary (LPA)'}
                )
                st.plotly_chart(fig_sal, use_container_width=True)
                
                avg = role_df['salary_mid'].mean()
                st.metric("Average Salary", f"₹{avg:.1f} LPA")
            else:
                st.info("Insufficient salary data.")

        st.subheader("Job Clusters & Career Map")
        if 'cluster_id' in jobs_df.columns:
            # We need PCA coordinates. If not in DF, we'd need to compute them.
            # Assuming they might be saved or we compute roughly here (expensive).
            # For speed, let's just show cluster distribution for the target role
            cluster_counts = role_df['cluster_label'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            fig_cluster = px.pie(
                cluster_counts, 
                values='Count', 
                names='Cluster',
                title=f"Role Variations for {target}",
                hole=0.4
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("Clustering data not available.")

    # --- TAB 3: Skills Gap ---
    with tab3:
        st.header("Skills Gap Analysis")
        
        if not st.session_state.analyzed_profile:
            st.warning("Please update your profile in the sidebar first.")
        else:
            user_skills = [s.lower() for s in st.session_state.user_profile.get('skills', [])]
            target = st.session_state.user_profile.get('target_role', 'Product Manager')
            
            # Get market top skills
            market_stats = ml.analyze_skills_by_role(jobs_df, role_filter=target)
            
            if not market_stats.empty:
                top_20 = market_stats.head(20)
                
                matched = []
                missing = []
                
                for _, row in top_20.iterrows():
                    skill_name = row['skills_list']
                    freq = row['percentage']
                    if skill_name.lower() in user_skills:
                        matched.append((skill_name, freq))
                    else:
                        missing.append((skill_name, freq))
                
                # Visuals
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"✅ You have {len(matched)} top skills!")
                    for s, f in matched:
                        st.markdown(f"- **{s}** (Found in {f:.0f}% of jobs)")
                        st.progress(f/100)
                        
                with c2:
                    st.error(f"⚠️ Missing {len(missing)} critical skills")
                    for s, f in missing:
                        st.markdown(f"- **{s}** (Found in {f:.0f}% of jobs)")
                        st.progress(f/100)
                        
                # Match Score
                score = len(matched) / (len(matched) + len(missing)) * 100
                st.metric("Profile Match Score", f"{score:.0f}%")
                
            else:
                st.info("No skill data for analysis.")

    # --- TAB 4: Job Matches ---
    with tab4:
        st.header(f"Top Job Matches for {st.session_state.user_profile.get('target_role', 'You')}")
        
        if not st.session_state.analyzed_profile:
            st.warning("Update profile to see personalized matches.")
        else:
            # Simple matching logic
            target = st.session_state.user_profile.get('target_role', '')
            user_skills = set([s.lower() for s in st.session_state.user_profile.get('skills', [])])
            pref_locs = [l.lower() for l in st.session_state.user_profile.get('locations', [])]
            
            # Filter by Role
            matches = jobs_df[jobs_df['job_title'].str.contains(target, case=False, na=False)].copy()
            
            # Filter by Location (if specific ones selected)
            if pref_locs and "Remote" not in st.session_state.user_profile.get('locations', []):
                 matches = matches[matches['location'].str.lower().apply(lambda x: any(l in x for l in pref_locs))]

            if matches.empty:
                st.info("No direct matches found. Try broadening your location or role.")
            else:
                # Calculate simple score
                def calc_score(row_skills):
                    if not isinstance(row_skills, list): return 0
                    row_skills_set = set([str(s).lower() for s in row_skills])
                    overlap = len(user_skills.intersection(row_skills_set))
                    return overlap

                matches['match_score'] = matches['skills_list'].apply(calc_score)
                matches = matches.sort_values('match_score', ascending=False).head(10)
                
                for _, job in matches.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="job-card">
                            <h4>{job['job_title']}</h4>
                            <p><strong>{job['company']}</strong> | 📍 {job['location']} | 💰 {job.get('salary_mid', 'N/A')} LPA</p>
                            <p>{job['job_description'][:200]}...</p>
                            <a href="{job['job_url']}" target="_blank">View Job</a>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show match skills
                        job_skills = job['skills_list'] if isinstance(job['skills_list'], list) else []
                        overlap = [s for s in job_skills if str(s).lower() in user_skills]
                        st.caption(f"Matched Skills: {', '.join(overlap)}")

else:
    st.error("Application failed to initialize. Please check data files.")
