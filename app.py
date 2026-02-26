import os
# Disable telemetry before any other imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import asyncio
import streamlit as st

# --- Streamlit Cloud Secrets Injection ---
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
if "HUGGINGFACE_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACE_API_TOKEN"] = st.secrets["HUGGINGFACE_API_TOKEN"]
if "MODEL_NAME" in st.secrets:
    os.environ["MODEL_NAME"] = st.secrets["MODEL_NAME"]

import ast
import sys
import os

# Add current directory to path to ensure 'src' is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Page Configuration ---
st.set_page_config(
    page_title="Job Intelligence Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
try:
    import src.rag_system as rag
    import src.ml_analyzer as ml
    from src.utils import clean_job_data, ALL_SKILLS
    from src.pipeline import run_full_pipeline
    from src.resume_parser import parse_resume
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure you are running the app from the root directory.")
    # Print the full error to console for easier debugging
    import traceback
    traceback.print_exc()
    st.stop()

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
        color: #212529; /* Dark text color for visibility */
        border: 1px solid #e0e0e0;
        border-left: 5px solid #1E88E5;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    .job-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .job-card h4 {
        color: #1E88E5;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .job-card p {
        color: #495057;
        margin-bottom: 8px;
    }
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        background-color: #E3F2FD;
        color: #1565C0;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .match-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        background-color: #E8F5E9;
        color: #2E7D32;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: bold;
        border: 1px solid #C8E6C9;
    }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def get_all_locations(df):
    """Extract all unique locations from the dataset."""
    if df is None or df.empty:
        return ["Bangalore", "Mumbai", "Pune", "Hyderabad", "Delhi", "Remote"]
        
    flat_db_locs = []
    if 'location' not in df.columns:
        return ["Bangalore", "Mumbai", "Pune", "Remote"]
    
    for loc_entry in df['location'].dropna():
        if isinstance(loc_entry, list):
            flat_db_locs.extend(loc_entry)
        else:
            # Handle string representation if not parsed
            if isinstance(loc_entry, str) and loc_entry.startswith('['):
                try:
                    import ast
                    loc_entry = ast.literal_eval(loc_entry)
                    if isinstance(loc_entry, list):
                        flat_db_locs.extend(loc_entry)
                    else: flat_db_locs.append(str(loc_entry))
                except: flat_db_locs.append(loc_entry)
            else:
                flat_db_locs.append(loc_entry)
                
    common_locs = ["Bangalore", "Mumbai", "Pune", "Hyderabad", "Delhi", "Remote", "Gurgaon", "Noida", "Chennai"]
    # Clean up and sort
    all_locs = sorted(list(set([str(l).strip() for l in flat_db_locs if l] + common_locs)))
    return all_locs

# --- Session State Management ---
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'analyzed_profile' not in st.session_state:
    st.session_state.analyzed_profile = False
if 'last_queried_role' not in st.session_state:
    st.session_state.last_queried_role = None
if 'last_queried_location' not in st.session_state:
    st.session_state.last_queried_location = None

# --- Resource Loading ---
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
                st.error("The processed data file is empty. Please run the scraping pipeline.")
                return None, None
            # Ensure safe literal eval for lists if they are strings
            if 'skills_list' in df.columns:
                 df['skills_list'] = df['skills_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
            if 'skills_extracted' in df.columns:
                 df['skills_extracted'] = df['skills_extracted'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
            if 'location' in df.columns:
                df['location'] = df['location'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        else:
            # Instead of auto-running hardcoded pipeline, return None to prompt user initialization
            return None, None

        # Load Vector Store
        persist_dir = "data/embeddings/chroma_db"
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            try:
                vectorstore = rag.load_vector_store()
            except Exception as e:
                st.warning(f"Vector store error: {str(e)}. Rebuilding index...")
                vectorstore = rag.create_vector_store(df)
        else:
            with st.status("Initializing Intelligence Database...", expanded=True) as status:
                st.write("Creating vector store from available data...")
                vectorstore = rag.create_vector_store(df)
                status.update(label="Database Initialized!", state="complete", expanded=False)
            
        # Create Chain
        chain, retriever = rag.create_rag_chain(vectorstore)
        
        return df, chain
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

# Load resources
if 'jobs_df' not in st.session_state or 'rag_chain' not in st.session_state:
    jobs_df, rag_chain = load_data_and_models()
    st.session_state.jobs_df = jobs_df
    st.session_state.rag_chain = rag_chain
    
    # Initialize engines
    if jobs_df is not None:
        if 'transition_model' not in st.session_state:
            st.session_state.transition_model = ml.CareerTransitionModel()
        if 'salary_engine' not in st.session_state:
            st.session_state.salary_engine = ml.SalaryPredictionEngine()
            
        # Trigger automatic training if not trained
        if not getattr(st.session_state.transition_model, 'is_trained', False):
            try:
                with st.spinner("Training career transition engine..."):
                    st.session_state.transition_model.train(jobs_df)
            except Exception as e:
                st.error(f"Transition Model Training Failed: {e}")
                
        if not getattr(st.session_state.salary_engine, 'is_trained', False):
            try:
                with st.spinner("Training salary predictor..."):
                    st.session_state.salary_engine.train(jobs_df)
            except Exception as e:
                st.error(f"Salary Engine Training Failed: {e}")
else:
    jobs_df = st.session_state.jobs_df
    rag_chain = st.session_state.rag_chain
    
    # Ensure models exist in state even if training was skipped
    if 'transition_model' not in st.session_state:
        st.session_state.transition_model = ml.CareerTransitionModel()
    if 'salary_engine' not in st.session_state:
        st.session_state.salary_engine = ml.SalaryPredictionEngine()

# --- Main Content ---
st.title("Job Market Intelligence Assistant")

if jobs_df is None:
    st.markdown("### 👋 Welcome! Let's get started.")
    st.info("No job market data found. Please initialize the database by telling us what you're looking for.")
    
    with st.form("init_form"):
        init_role = st.text_input("Target Role (e.g., Product Manager)", value="Product Manager")
        init_loc = st.text_input("Target Location (e.g., Mumbai)", value="Mumbai")
        submitted = st.form_submit_button("🚀 Start Analysis")
        
        if submitted:
            with st.status("Initializing Intelligence Database...", expanded=True) as status:
                st.write(f"Scraping live market data for **{init_role}** in **{init_loc}**...")
                # Run pipeline with user inputs
                run_full_pipeline([init_role], [init_loc], num_pages=2, use_selenium=True)
                
                st.write("Processing data and generating insights...")
                # Reload resources
                new_df, new_chain = load_data_and_models()
                
                if new_df is not None:
                    st.session_state.jobs_df = new_df
                    st.session_state.rag_chain = new_chain
                    st.session_state.last_queried_role = init_role
                    st.session_state.last_queried_location = init_loc
                    status.update(label="Initialization Complete!", state="complete", expanded=False)
                    st.rerun()
                else:
                    st.error("Failed to initialize data. Please try again.")

elif jobs_df is not None:
    
    # Updated Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Job Inquiry", "📊 ML Insights", "👤 Profile Analysis", "💼 Job Matches", "🔄 Career Transition", "🤝 Skill Associations"])

    # --- TAB 1: Job Inquiry ---
    with tab1:
        st.markdown("### 🔍 Search Job Market")
        
        # 1. Query Inputs
        c1, c2 = st.columns([2, 1])
        with c1:
            def update_global_role():
                st.session_state.last_queried_role = st.session_state.role_input_tab1

            query_role = st.text_input(
                "Role (e.g. Product Manager)", 
                value=st.session_state.last_queried_role or "",
                key="role_input_tab1",
                on_change=update_global_role
            )
        with c2:
            all_locs = get_all_locations(jobs_df)
            
            # Ensure session state has a valid location
            if not st.session_state.last_queried_location:
                st.session_state.last_queried_location = "Mumbai" 
    
            default_ix = all_locs.index(st.session_state.last_queried_location)
            
            def update_global_loc():
                st.session_state.last_queried_location = st.session_state.loc_selector_tab1

            query_loc = st.selectbox(
                "Location", 
                all_locs, 
                index=default_ix, 
                key="loc_selector_tab1",
                on_change=update_global_loc
            )

        # 2. Intent Buttons
        st.caption("Quick Analysis:")
        b1, b2, b3, b4 = st.columns(4)
        
        intent_prompt = None
        selected_intent = None
        
        if b1.button("💰 Salary Insights", use_container_width=True):
            intent_prompt = f"""As a career advisor, provide a detailed salary analysis for a {query_role} in {query_loc}. I want to understand the full compensation spectrum. Please include:
1. The typical salary range (25th to 75th percentile).
2. The median salary.
3. Factors that influence salary for this role (e.g., specific skills, years of experience, company size).
4. Any recent trends in compensation for this role in the specified location."""
            selected_intent = "salary"
        if b2.button("🛠️ Top Skills", use_container_width=True):
            intent_prompt = f"""As a career analyst, I need a comprehensive breakdown of the essential skills for a {query_role} in {query_loc}. Please categorize the skills and provide specific examples. The breakdown should include:
1.  **Technical Skills:** (e.g., programming languages, software, tools).
2.  **Soft Skills:** (e.g., communication, leadership, teamwork).
3.  **Domain-Specific Skills:** (e.g., financial modeling for a finance role).
4.  **Emerging/Trending Skills:** What new skills are becoming important for this role?"""
            selected_intent = "skills"
        if b3.button("🚀 Career Path", use_container_width=True):
            intent_prompt = f"""As a mentor, outline a typical career path for a {query_role}, starting from an entry-level position. Please provide a roadmap that includes:
1.  Common entry-level job titles for this career track.
2.  Mid-level and senior-level roles one can progress to.
3.  Typical timelines for career progression (e.g., 2-4 years to senior).
4.  Key responsibilities and expectations at each stage.
5.  Suggestions for professional development (e.g., certifications, further education) to accelerate growth."""
            selected_intent = "career_path"
        if b4.button("🏢 Top Companies", use_container_width=True):
            intent_prompt = f"""As a market intelligence analyst, identify the top companies hiring for the {query_role} role in {query_loc}. For each company, please provide:
1.  A brief overview of the company and its culture.
2.  The types of projects or teams they are hiring for.
3.  What makes them an attractive employer for this role.
4.  Any specific or unique requirements they have for this role mentioned in job postings."""
            selected_intent = "general" # Companies fall under general/market info

        st.divider()

        # 3. Chat Interface
        # Display history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.conversation_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg and msg["sources"]:
                        with st.expander("📚 Sources & Details"):
                            st.markdown(msg["sources"])
                    if "ml_insights" in msg and msg["ml_insights"]:
                        with st.expander("🔍 ML Insights Used"):
                            st.info(msg["ml_insights"])

        # Chat Input (Custom or Button-triggered)
        user_input = st.chat_input("Ask a specific question...")
        
        final_prompt = intent_prompt if intent_prompt else user_input
        
        if final_prompt:
            # Update Session State
            st.session_state.last_queried_role = query_role
            st.session_state.last_queried_location = query_loc
            
            # User message
            st.session_state.conversation_history.append({"role": "user", "content": final_prompt})
            
            # Assistant response
            with st.spinner(f"Analyzing {query_role} in {query_loc}..."):
                try:
                    # 1. Get Response (Pass explicit role/loc to help RAG)
                    if not intent_prompt and user_input: # Enhance general user queries
                        rag_query = f"""As an expert career advisor, please provide a comprehensive answer to the following question:

"{user_input}"

When answering, please consider all relevant aspects of the question and use the context of Role={query_role} and Location={query_loc} to tailor your answer."""
                    else:
                        rag_query = f"{final_prompt} (Context: Role={query_role}, Location={query_loc})"
                    
                    response_data = rag.enhanced_query_with_ml(rag_query, rag_chain, jobs_df, forced_intent=selected_intent)
                    
                    # 2. Check if dynamic scrape is needed
                    if response_data.get('needs_scrape'):
                        max_scrape_attempts = 2 # Reduced for responsiveness
                        current_attempt = 1
                        
                        while response_data.get('needs_scrape') and current_attempt <= max_scrape_attempts:
                            with st.status(f"🔍 Low data. Scraping live jobs for '{query_role}'...", expanded=True) as status:
                                # Run pipeline for the specific role/loc
                                new_jobs_count = run_full_pipeline([query_role], [query_loc], num_pages=2, use_selenium=True)
                                
                                if new_jobs_count > 0:
                                    status.write("✅ New data found! Updating intelligence...")
                                    # Reload resources
                                    new_df, new_chain = load_data_and_models()
                                    if new_df is not None:
                                        st.session_state.jobs_df = new_df
                                        st.session_state.rag_chain = new_chain
                                        jobs_df = new_df
                                        rag_chain = new_chain
                                    status.update(label="Database Updated!", state="complete", expanded=False)
                                else:
                                    status.update(label="No new unique jobs found.", state="complete", expanded=False)
                                    
                                # Re-check sufficiency
                                response_data = rag.enhanced_query_with_ml(rag_query, rag_chain, jobs_df, forced_intent=selected_intent)
                                
                                if not response_data.get('needs_scrape'):
                                    break
                                current_attempt += 1

                    answer = response_data['answer']
                    
                    # 3. Format Sources
                    sources_text = rag.format_response_with_sources("", response_data['source_documents'])
                    if "📚 Sources" in sources_text:
                        sources_text = sources_text.split("### 📚 Sources & Relevant Jobs:\n")[1]

                    # Append to history
                    st.session_state.conversation_history.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources_text,
                        "ml_insights": response_data.get('ml_insights')
                    })
                    
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # --- TAB 2: Market Insights ---
    with tab2:
        # Strict logic: Insights are ONLY for the role currently being discussed in chat
        target = st.session_state.last_queried_role
        default_loc = st.session_state.last_queried_location or "All Locations"
        
        if not target:
            st.info("👋 **Start by asking a question in the 'Chat' tab!**")
            st.markdown("""
            For example:
            - *"What skills do I need for a Product Manager role?"*
            - *"How much does a Data Analyst earn in Bangalore?"*
            - *"Show me the career path for a Business Analyst."*
            
            Once we detect the role you're interested in, this dashboard will light up with real-time market data! 🚀
            """)
        else:
            target = target.title()
            
            # --- Location Filter ---
            # Extract unique locations for this role to populate dropdown
            role_base_df = jobs_df[jobs_df['job_title'].str.contains(target, case=False, na=False)]
            # Manually flatten and find unique locations
            flat_locs = set()
            for loc_list in role_base_df['location'].dropna():
                if isinstance(loc_list, list):
                    for loc in loc_list:
                        flat_locs.add(loc)
                else: # Handle single string entries
                    flat_locs.add(loc_list)
            
            available_locs = ["All Locations"] + sorted(list(flat_locs))
            
            # Use Tab 1 selection if available
            current_loc = st.session_state.last_queried_location or "All Locations"
            
            loc_index = 0
            if current_loc in available_locs:
                loc_index = available_locs.index(current_loc)
            
            def sync_loc_to_tab1():
                st.session_state.last_queried_location = st.session_state.loc_selector_tab2

            selected_loc = st.selectbox(
                "Filter by Location:", 
                available_locs, 
                index=loc_index,
                key="loc_selector_tab2",
                on_change=sync_loc_to_tab1
            )

            st.header(f"Market Insights: {target} in {selected_loc}")
            
            # Strictly filter data for this role AND location
            if selected_loc == "All Locations":
                role_df = role_base_df
            else:
                role_df = role_base_df[role_base_df['location'].apply(
                    lambda loc_list: isinstance(loc_list, list) and any(selected_loc.lower() in item.lower() for item in loc_list)
                )]
            
            role_count = len(role_df)
            
            if role_count > 0:
                st.info(f"📊 Analysis based on **{role_count}** specific job postings for '{target}' found in the last 14 days.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Required Skills")
                    skill_stats = ml.analyze_skills_by_role(role_df) # Pass already filtered DF
                    
                    if not skill_stats.empty:
                        fig_skills = px.bar(
                            skill_stats.head(10),
                            x='percentage',
                            y='skill_name',
                            orientation='h',
                            title=f"Skill Frequency (n={role_count})",
                            labels={'skill_name': 'Skill', 'percentage': 'Frequency (%)'},
                            color='percentage',
                            color_continuous_scale='Blues'
                        )
                        fig_skills.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_skills, use_container_width=True)
                    else:
                        st.info("No skill data available.")

                with col2:
                    st.subheader("Salary Distribution")
                    if not role_df.empty and 'salary_mid' in role_df.columns and not role_df['salary_mid'].isna().all():
                        fig_sal = px.box(
                            role_df, 
                            y='salary_mid', 
                            points="all",
                            title=f"Salary Range (n={role_df['salary_mid'].count()})",
                            labels={'salary_mid': 'Annual Salary (LPA)'},
                            color_discrete_sequence=['#43A047']
                        )
                        st.plotly_chart(fig_sal, use_container_width=True)
                        
                        avg = role_df['salary_mid'].mean()
                        st.metric(f"Average {target} Salary", f"₹{avg:.1f} LPA")
                    else:
                        st.info(f"Insufficient salary data for '{target}'.")

                # Row 2
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Top Hiring Companies")
                    top_cos = role_df['company'].value_counts().head(10).reset_index()
                    top_cos.columns = ['Company', 'Job Count']
                    fig_cos = px.bar(
                        top_cos,
                        x='Job Count',
                        y='Company',
                        orientation='h',
                        title=f"Top Employers (n={role_count})",
                        color='Job Count',
                        color_continuous_scale='Greens'
                    )
                    fig_cos.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_cos, use_container_width=True)

                with c2:
                    st.subheader("Role Variations (Clusters)")
                    if 'cluster_id' in role_df.columns:
                        # Filter out garbage clusters (e.g., error pages)
                        valid_clusters = role_df[~role_df['cluster_label'].str.contains("error|000", case=False, na=False)]
                        
                        if not valid_clusters.empty:
                            cluster_counts = valid_clusters['cluster_label'].value_counts().reset_index()
                            cluster_counts.columns = ['Cluster', 'Count']
                            
                            fig_cluster = px.pie(
                                cluster_counts, 
                                values='Count', 
                                names='Cluster',
                                title=f"Sub-specializations (n={len(valid_clusters)})",
                                hole=0.4
                            )
                            st.plotly_chart(fig_cluster, use_container_width=True)
                        else:
                            st.info("No valid cluster data available.")
                    else:
                        st.warning("Clustering data not available.")

                st.divider()
                st.subheader("🚀 Advanced Market Intelligence")
                
                c_adv1, c_adv2 = st.columns(2)
                
                with c_adv1:
                    st.markdown("#### Emerging Skills (Growth Trends)")
                    # Run the emerging skills model for the specific role
                    emerging_df = ml.identify_emerging_skills(role_df)
                    if not emerging_df.empty:
                        fig_emerge = px.line(
                            emerging_df.head(10),
                            x='skill_name',
                            y='growth_rate',
                            title=f"Fastest Growing Skills for {target}",
                            labels={'skill_name': 'Skill', 'growth_rate': 'Growth Rate (%)'},
                            markers=True
                        )
                        st.plotly_chart(fig_emerge, use_container_width=True)
                        st.caption("Growth rate calculated against simulated baseline for current market volume.")
                    else:
                        st.info("Insufficient historical data for emerging skill analysis.")

                with c_adv2:
                    st.markdown("#### Skill Salary Premiums (Financial Impact)")
                    # Run the salary impact model using Top 20 skills from analysis
                    top_20_skills = skill_stats.head(20)['skill_name'].tolist() if not skill_stats.empty else None
                    impact_df = ml.calculate_skill_salary_impact(role_df, skill_list=top_20_skills)
                    if not impact_df.empty:
                        # Filter for significant results (p < 0.1) if possible, or just top premiums
                        top_impact = impact_df.sort_values('premium', ascending=False).head(10)
                        fig_impact = px.bar(
                            top_impact,
                            x='premium',
                            y='skill',
                            orientation='h',
                            title=f"Highest Paying Skills for {target}",
                            labels={'skill': 'Skill', 'premium': 'Avg. Salary Boost (LPA)'},
                            color='premium',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_impact, use_container_width=True)
                        st.caption("Premium: Average salary difference between jobs with vs. without this skill.")
                    else:
                        st.info("Insufficient salary data to calculate skill premiums.")
                
                # --- Salary Benchmark Table ---
                st.markdown("#### 📊 Statistical Salary Benchmarks")
                
                # Check for Salary Prediction Engine output
                engine = st.session_state.get('salary_engine')
                profile = st.session_state.get('user_profile')
                
                if engine and engine.is_trained and profile and 'current_role' in profile:
                    # Determine cluster for prediction
                    if not role_df.empty:
                        target_cluster = role_df['cluster_label'].iloc[0]
                        predicted_sal = engine.predict(profile, target_cluster)
                        
                        st.success(f"🤖 **Personalized Salary Prediction:** Based on your experience ({profile['years_experience']} yrs) and skills ({len(profile['skills'])}), our **Random Forest model** predicts a potential salary of **₹{predicted_sal} LPA** for a {target} role.")
                
                benchmarks = ml.analyze_salary_trends(role_df)
                if benchmarks and 'by_experience' in benchmarks and benchmarks['by_experience']:
                    exp_data = []
                    for level, stats in benchmarks['by_experience'].items():
                        exp_data.append({
                            'Experience Level': level,
                            '25th Percentile': f"₹{stats['25%']} LPA",
                            'Median (50th)': f"₹{stats['50%']} LPA",
                            '75th Percentile': f"₹{stats['75%']} LPA"
                        })
                    st.table(pd.DataFrame(exp_data))
                else:
                    st.info("Detailed statistical benchmarks not available for this sample size.")

                # Action to get more data
                if role_count < 15:
                    st.warning(f"⚠️ Sample size ({role_count}) is low. Insights may not be statistically significant.")
                    if st.button(f"Fetch more '{target}' jobs from live market"):
                        with st.status(f"Scraping more live jobs for '{target}'...", expanded=True):
                            new_count = run_full_pipeline([target], ["Bangalore", "Mumbai", "Pune", "Hyderabad"], num_pages=3, use_selenium=True)
                            if new_count > 0:
                                st.success(f"Added {new_count} new jobs! Reloading...")
                                st.rerun()
                            else:
                                st.info("No more new unique jobs found at this time.")
            else:
                st.error(f"No data found for '{target}'.")
                if st.button(f"Scrape live jobs for '{target}' now"):
                    with st.status(f"Searching for '{target}' jobs..."):
                        new_count = run_full_pipeline([target], ["Bangalore", "Mumbai"], num_pages=2, use_selenium=True)
                        if new_count > 0:
                            st.rerun()

    # --- TAB 3: Profile Analysis ---
    with tab3:
        st.header("👤 Your Profile & Gap Analysis")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("### Profile Details")
            uploaded_file = st.file_uploader("Upload Resume (Optional)", type=['pdf', 'txt'])
            
            # --- Resume Processing ---
            if uploaded_file:
                if 'resume_processed' not in st.session_state or st.session_state.resume_processed != uploaded_file.name:
                    with st.spinner("Analyzing Resume..."):
                        resume_data = parse_resume(uploaded_file, uploaded_file.type)
                        if resume_data:
                            # Skills extraction
                            if resume_data.get('skills'):
                                extracted = resume_data['skills']
                                # Merge with existing skills (avoid duplicates)
                                current = set(st.session_state.user_profile.get('skills', []))
                                current.update(extracted)
                                st.session_state.user_profile['skills'] = list(current)
                            
                            # Education extraction
                            if resume_data.get('education'):
                                st.session_state.user_profile['education'] = resume_data['education']
                            
                            # Update session state
                            st.session_state.resume_processed = uploaded_file.name
                            st.success(f"✅ Extracted profile data from resume!")
                        else:
                            st.warning("Could not extract data from this resume.")
            
            # Inputs (Pre-filled from session state if available)
            p_role = st.text_input("Current Role", value=st.session_state.user_profile.get('current_role', "MBA Student"))
            p_exp = st.slider("Years of Experience", 0, 15, st.session_state.user_profile.get('years_experience', 2))
            
            # Education Input
            edu_options = ["High School", "Bachelor's", "B.Tech/B.E", "Master's", "MBA", "M.Tech/M.E", "PhD"]
            current_edu = st.session_state.user_profile.get('education', ["Master's"])
            if isinstance(current_edu, list) and current_edu:
                default_edu = current_edu[0] if current_edu[0] in edu_options else "Master's"
            elif isinstance(current_edu, str) and current_edu in edu_options:
                default_edu = current_edu
            else:
                default_edu = "Master's"
                
            p_edu = st.selectbox("Highest Education Level", edu_options, index=edu_options.index(default_edu))
            
            # Use centralized skill dictionary
            common_skills = ALL_SKILLS
            default_skills = st.session_state.user_profile.get('skills', ["Excel", "Communication"])
            # Filter default skills to only include those present in common_skills
            default_skills = [s for s in default_skills if s in common_skills]
                
            p_skills = st.multiselect("Your Skills", common_skills, default=default_skills)
            
            p_locs = st.multiselect(
                "Preferred Locations", 
                ["Bangalore", "Mumbai", "Pune", "Hyderabad", "Delhi/NCR", "Remote"],
                default=st.session_state.user_profile.get('locations', ["Bangalore", "Mumbai"])
            )
            
            if st.button("Analyze Profile", type="primary", use_container_width=True):
                st.session_state.user_profile = {
                    "current_role": p_role,
                    "years_experience": p_exp,
                    "education": [p_edu],
                    "skills": p_skills,
                    "locations": p_locs
                }
                st.session_state.analyzed_profile = True
                st.rerun()

        with c2:
            target = st.session_state.last_queried_role
            if not target:
                st.info("👈 **Update your profile on the left**, then ask about a target role in 'Job Inquiry' to see the gap analysis.")
            elif not st.session_state.analyzed_profile:
                st.warning("Please click 'Analyze Profile' to see your skills gap.")
            else:
                st.subheader(f"Gap Analysis for {target.title()}")
                
                # Get strictly filtered market stats
                role_df = jobs_df[jobs_df['job_title'].str.contains(target, case=False, na=False)]
                
                if role_df.empty:
                    st.warning(f"No job data found for '{target}' to analyze gaps.")
                else:
                    market_stats = ml.analyze_skills_by_role(role_df)
                    
                    if not market_stats.empty:
                        top_20 = market_stats.head(20)
                        
                        matched = []
                        missing = []
                        current_user_skills = st.session_state.user_profile.get('skills', [])
                        user_skills_lower = [s.lower() for s in current_user_skills]
                        
                        for _, row in top_20.iterrows():
                            skill_name = row['skill_name']
                            freq = row['percentage']
                            if skill_name.lower() in user_skills_lower:
                                matched.append((skill_name, freq))
                            else:
                                missing.append((skill_name, freq))
                        
                        # Score
                        total_relevant = len(matched) + len(missing)
                        score = (len(matched) / total_relevant * 100) if total_relevant > 0 else 0
                        
                        st.metric("Profile Match Score", f"{score:.0f}%")
                        st.progress(score/100)
                        
                        c_a, c_b = st.columns(2)
                        with c_a:
                            st.success(f"✅ You have {len(matched)} top skills")
                            for s, f in matched:
                                st.caption(f"{s} ({f:.0f}%)")
                        with c_b:
                            st.error(f"⚠️ Missing {len(missing)} critical skills")
                            for s, f in missing:
                                st.caption(f"{s} ({f:.0f}%)")
                                
                        # --- Experience Alignment Analysis ---
                        st.divider()
                        st.subheader("📊 Experience Alignment Analysis")
                        
                        exp_analysis = ml.analyze_experience_alignment(role_df, p_exp)
                        
                        if exp_analysis:
                            col_ea1, col_ea2 = st.columns([1, 1])
                            
                            with col_ea1:
                                st.markdown(f"""
                                <div style="padding: 20px; border-radius: 10px; background-color: {exp_analysis['color']}22; border-left: 5px solid {exp_analysis['color']};">
                                    <h3 style="margin-top: 0; color: {exp_analysis['color']};">{exp_analysis['status']}</h3>
                                    <p style="font-size: 1.1em;">{exp_analysis['message']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.write("")
                                st.metric("Accessible Market Volume", f"{exp_analysis['accessible_volume_pct']}%", 
                                          help="Percentage of jobs in this role that are realistically within reach for your experience level (up to user experience + 1 year).")
                            
                            with col_ea2:
                                # Distribution chart
                                dist_data = pd.DataFrame([
                                    {'Experience': k, 'Job Count': v} 
                                    for k, v in exp_analysis['exp_distribution'].items()
                                ])
                                
                                fig_exp_dist = px.bar(
                                    dist_data,
                                    x='Experience',
                                    y='Job Count',
                                    title=f"Market Experience Distribution: {target.title()}",
                                    labels={'Experience': 'Required Experience (Min)', 'Job Count': 'Number of Postings'},
                                    color_discrete_sequence=[exp_analysis['color']]
                                )
                                
                                # Add vertical line for user experience
                                fig_exp_dist.add_vline(x=p_exp, line_dash="dash", line_color="black", 
                                                      annotation_text=f"You", 
                                                      annotation_position="top right")
                                
                                st.plotly_chart(fig_exp_dist, use_container_width=True)
                        else:
                            st.info("Insufficient experience data for market volume analysis.")
                    else:
                        st.info("No skill data available for analysis.")

    # --- TAB 4: Job Matches ---
    with tab4:
        target = st.session_state.last_queried_role
        
        if not target:
            st.info("💬 **Step 1:** Search for a role in 'Job Inquiry'.")
        elif not st.session_state.analyzed_profile:
            st.info("👤 **Step 2:** Go to 'Profile Analysis' and click 'Analyze Profile' to see personalized matches.")
        else:
            st.markdown(f"### 💼 Personalized Matches: {target.title()}")
            
            # Matching logic using Tab 3 profile data
            user_skills = set([s.lower() for s in st.session_state.user_profile.get('skills', [])])
            pref_locs = [l.lower() for l in st.session_state.user_profile.get('locations', [])]
            
            # Filter by Role
            matches = jobs_df[jobs_df['job_title'].str.contains(target, case=False, na=False)].copy()
            
            # Filter by Location (if specific ones selected)
            if pref_locs:
                 matches = matches[matches['location'].apply(
                     lambda loc_list: isinstance(loc_list, list) and any(p_loc.lower() in map(str.lower, loc_list) for p_loc in pref_locs)
                 )]

            # Filter by Experience (Realistic Range: User Exp + 3 years max)
            user_exp = st.session_state.user_profile.get('years_experience', 0)
            matches = matches[matches['experience_min'] <= (user_exp + 3)]

            if matches.empty:
                st.warning(f"No direct matches found for **{target}** with your experience level ({user_exp} yrs).")
                st.caption("Try broadening your role search or adding more locations.")
            else:
                # --- Improved Matching Logic ---
                user_edu = st.session_state.user_profile.get('education', [])
                
                def calc_score(row):
                    # 1. Skill Score (Weight: 30%) - Decreased from 60%
                    row_skills = row['skills_list']
                    if not isinstance(row_skills, list) or not row_skills: 
                        skill_score = 0
                    else:
                        row_skills_set = set([str(s).lower() for s in row_skills])
                        overlap_count = len(user_skills.intersection(row_skills_set))
                        skill_score = (overlap_count / len(row_skills_set)) * 30
                    
                    # 2. Experience Score (Weight: 60%) - Increased from 30%
                    job_exp_min = row.get('experience_min', 0)
                    if pd.isna(job_exp_min): job_exp_min = 0
                    
                    if user_exp >= job_exp_min:
                        diff = user_exp - job_exp_min
                        # Higher importance: penalty for over-qualification is steeper to keep matches relevant
                        exp_score = 60 - min(diff * 4, 20) 
                    else:
                        diff = job_exp_min - user_exp
                        # Heavy penalty for under-qualification
                        exp_score = max(0, 60 - (diff * 15))
                        
                    # 3. Education Score (Weight: 10%)
                    job_edu = row.get('education_extracted', [])
                    if isinstance(job_edu, str):
                        try:
                            import ast
                            job_edu = ast.literal_eval(job_edu)
                        except:
                            job_edu = [job_edu]
                    
                    # Use transition model's matching logic if available, else simple check
                    model = st.session_state.get('transition_model')
                    if model and hasattr(model, '_match_education'):
                        edu_match = model._match_education(user_edu, job_edu)
                    else:
                        # Simple fallback
                        user_edu_set = set(e.lower() for e in user_edu)
                        edu_match = 1.0 if any(e.lower() in user_edu_set for e in job_edu) or not job_edu else 0.0
                    
                    edu_score = edu_match * 10
                        
                    return skill_score + exp_score + edu_score

                matches['match_score'] = matches.apply(calc_score, axis=1)

                matches = matches.sort_values('match_score', ascending=False).head(15)
                
                # Summary Metric
                st.markdown(f"Found **{len(matches)}** potential matches based on your skills and **{user_exp} years** of experience.")
                
                for _, job in matches.iterrows():
                    # Format location
                    loc_display = ", ".join(job['location']) if isinstance(job['location'], list) else str(job['location'])
                    
                    # Salary display
                    salary = job.get('salary_mid', 'N/A')
                    salary_display = f"₹{salary} LPA" if salary != 'N/A' and salary != 0 else "Not Disclosed"

                    # Match Skills logic
                    job_skills = job['skills_list'] if isinstance(job['skills_list'], list) else []
                    overlap = [s for s in job_skills if str(s).lower() in user_skills]
                    missing = [s for s in job_skills if str(s).lower() not in user_skills]

                    with st.container():
                        # Card Header and Info
                        score_color = "#2E7D32" if job['match_score'] > 70 else "#FFA726" if job['match_score'] > 40 else "#D32F2F"
                        
                        st.markdown(f"""
                        <div class="job-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <h4>{job['job_title']}</h4>
                                <span style="background-color: {score_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold;">
                                    {job['match_score']:.0f}% Match
                                </span>
                            </div>
                            <p style="font-size: 1.1em;"><strong>{job['company']}</strong></p>
                            <p>📍 {loc_display} | 💰 {salary_display} | ⏳ {job.get('experience_min', 0)}-{job.get('experience_max', 5)} yrs</p>
                            <p style="font-style: italic; color: #6c757d;">{job['job_description'][:300]}...</p>
                            <div style="margin-top: 15px;">
                                <span style="font-weight: bold; color: #2E7D32;">Matched Skills:</span><br>
                                {' '.join([f'<span class="match-tag">{s}</span>' for s in overlap]) if overlap else '<span style="color: #6c757d; font-size: 0.9em;">No direct skill matches</span>'}
                            </div>
                            <div style="margin-top: 10px; margin-bottom: 20px;">
                                <span style="font-weight: bold; color: #D32F2F;">Skills to add for better matching:</span><br>
                                {' '.join([f'<span class="skill-tag" style="background-color: #FFEBEE; color: #C62828; border: 1px solid #FFCDD2;">{s}</span>' for s in missing[:8]]) if missing else '<span style="color: #2E7D32; font-size: 0.9em;">You have all specified skills!</span>'}
                            </div>
                            <a href="{job['job_url']}" target="_blank" style="text-decoration: none;">
                                <button style="background-color: #1E88E5; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                                    Apply on Portal ↗
                                </button>
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write("") # Spacer

    # --- TAB 5: Career Transition ---
    with tab5:
        st.header("🔄 Career Transition Intelligence (Logistic Regression)")
        
        target = st.session_state.last_queried_role
        model = st.session_state.get('transition_model')
        
        if not target:
            st.info("💬 **Step 1:** Search for a target role in 'Job Inquiry' to analyze your transition.")
        elif not st.session_state.analyzed_profile:
            st.info("👤 **Step 2:** Complete your 'Profile Analysis' to enable transition probability calculation.")
        elif model is None or not getattr(model, 'is_trained', False):
            st.warning("⚠️ Transition model is initializing or requires training with the current dataset.")
            if st.button("🚀 Initialize & Train Model", type="primary"):
                 with st.spinner("Training reliable Logistic Regression engine..."):
                     try:
                         # Ensure we have a fresh instance if needed
                         if model is None:
                             st.session_state.transition_model = ml.CareerTransitionModel()
                             model = st.session_state.transition_model
                         
                         model.train(jobs_df)
                         st.success("✅ Model trained successfully!")
                         st.rerun()
                     except Exception as e:
                         st.error(f"Training failed: {e}")
        else:
            # 1. Calculate for current target
            # Create a 'Market Representative' job by aggregating all roles of this type
            role_jobs = jobs_df[jobs_df['job_title'].str.contains(target, case=False, na=False)]
            
            if role_jobs.empty:
                st.error(f"No specific job data found for '{target}' to run transition analysis.")
            else:
                with st.spinner(f"Aggregating market standards for {target}..."):
                    # Get Top 20 skills for this role (Market Average)
                    market_skills_df = ml.analyze_skills_by_role(role_jobs)
                    if not market_skills_df.empty:
                        market_skills = market_skills_df.head(20)['skill_name'].tolist()
                    else:
                        market_skills = []
                        
                    # Create a Virtual Representative Job
                    # Logic: Average of bottom 10 min_exp values as the 'entry point' requirement
                    if 'experience_min' in role_jobs.columns:
                        exp_req = role_jobs['experience_min'].nsmallest(10).mean()
                    else:
                        exp_req = 0

                    rep_job = {
                        'job_title': target,
                        'skills_list': market_skills,
                        'experience_min': exp_req,
                        'location': role_jobs['location'].iloc[0] if not role_jobs.empty else "Multiple"
                    }
                    
                    # Use Top 20 weighted match
                    result = model.predict(st.session_state.user_profile, rep_job, market_skills=market_skills_df)
                
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    st.subheader(f"Transition: {st.session_state.user_profile.get('current_role')} ➔ {target.title()}")
                    prob = result['probability']
                    
                    # Probability Gauge
                    fig_prob = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Success Probability (%)", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#1E88E5"},
                            'steps': [
                                {'range': [0, 40], 'color': "#FFEBEE"},
                                {'range': [40, 70], 'color': "#FFF3E0"},
                                {'range': [70, 100], 'color': "#E8F5E9"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': prob
                            }
                        }
                    ))
                    fig_prob.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_prob, use_container_width=True, key="transition_gauge")
                    
                    st.markdown(f"**Confidence Level:** `{result['confidence']}`")
                    st.info(result['recommendation'])
                    
                    with st.expander("📝 Detailed Factors & Logic"):
                        st.write("### Transition Analysis Factors")
                        st.caption("Analyzed via Logistic Regression trained on 3,000 market simulations.")
                        for k, v in result['factors'].items():
                            st.write(f"**{k}:** {v}")
                        
                        st.divider()
                        st.write("### 🧠 Model Logic (Session 8: Logistic Regression)")
                        st.write("""
                        This probability is calculated using a **Logistic Regression model** that has been trained to recognize successful vs. failed transitions based on historical market standards.
                        
                        **How the intelligence works:**
                        - **Weighted Skill Match:** Your profile is compared against the **Top 20 skills** found in all real-world postings for this role. Each skill is weighted by its **market frequency %**—matching a high-demand skill (e.g., Python at 85%) impacts your score more than a niche skill.
                        - **Experience Clipping:** Having more experience than required helps, but after a 3-year surplus, the model prioritizes skills and other factors to prevent seniority from masking a total skill gap.
                        - **Sigmoid Activation:** The final factors are passed through a Sigmoid function to produce a probability between 0% and 100%.
                        """)
                
                with c2:
                    st.subheader("💡 Learned Impact Factors")
                    # Feature Importance (Weights from the model)
                    importance = model.get_feature_importance()
                    if importance:
                        # Map internal names to readable ones
                        readable_names = {
                            'Skill Match': 'Skill Match Strength',
                            'Experience Fit': 'Experience Fit',
                            'Location Alignment': 'Location Alignment',
                            'Education Compatibility': 'Education Compatibility',
                            'Role Complexity': 'Role Complexity'
                        }
                        imp_df = pd.DataFrame([
                            {'Factor': readable_names.get(k, k), 'Impact': v} 
                            for k, v in importance.items()
                        ]).sort_values('Impact', ascending=True)
                        
                        fig_imp = px.bar(
                            imp_df, 
                            x='Impact', 
                            y='Factor', 
                            orientation='h',
                            title="Learned Model Coefficients",
                            color='Impact',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_imp, use_container_width=True, key="factor_impact_chart")
                        st.caption("Positive (green) factors increase success chance. Negative (red) factors decrease it.")
                    
                    if st.button("🔄 Retrain on New Market Data"):
                         with st.spinner("Retraining..."):
                             model = ml.CareerTransitionModel()
                             model.train(jobs_df)
                             st.session_state.transition_model = model
                             st.rerun()
                    
                st.divider()
                
                # 2. Transition Matrix / Heatmap (Current vs Market Roles)
                st.subheader("🌐 Market-wide Transition Heatmap")
                st.write("""
                This visualization shows your likelihood of transitioning to other common roles in the industry. 
                Each cell is calculated by passing your profile through the Logistic Regression engine against market-average standards for that role.
                """)
                
                # Sample a few roles for the heatmap
                market_roles = ["Product Manager", "Data Scientist", "Software Engineer", "Business Analyst", "Marketing Manager", "HR Specialist", "Sales Manager"]
                heatmap_data = []
                
                for role in market_roles:
                    # Find role jobs
                    r_jobs = jobs_df[jobs_df['job_title'].str.contains(role, case=False, na=False)]
                    if not r_jobs.empty:
                        # Aggregate skills for this market role
                        r_skills_df = ml.analyze_skills_by_role(r_jobs)
                        r_skills = r_skills_df.head(20)['skill_name'].tolist() if not r_skills_df.empty else []
                        
                        # Create virtual rep
                        # Use average of 10 lowest min_exp as the 'entry point' requirement
                        if 'experience_min' in r_jobs.columns:
                            r_exp_req = r_jobs['experience_min'].nsmallest(10).mean()
                        else:
                            r_exp_req = 0

                        r_rep = {
                            'job_title': role,
                            'skills_list': r_skills,
                            'experience_min': r_exp_req,
                            'location': r_jobs['location'].iloc[0]
                        }
                        
                        # Use weighted Top 20 match
                        res = model.predict(st.session_state.user_profile, r_rep, market_skills=r_skills_df)
                        heatmap_data.append({'Target Role': role, 'Probability': res['probability']})
                
                if heatmap_data:
                    h_df = pd.DataFrame(heatmap_data).sort_values('Probability', ascending=False)
                    
                    # Create a 1D Heatmap using go.Heatmap
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=[h_df['Probability'].tolist()],
                        x=h_df['Target Role'].tolist(),
                        y=['Success Prob %'],
                        colorscale='Viridis',
                        text=[[f"{p}%" for p in h_df['Probability']]],
                        texttemplate="%{text}",
                        showscale=True
                    ))
                    
                    fig_heat.update_layout(
                        title=f"Transition Potential: {st.session_state.user_profile.get('current_role')} to Industry Roles",
                        height=250,
                        xaxis_nticks=len(market_roles)
                    )
                    
                    st.plotly_chart(fig_heat, use_container_width=True, key="market_transition_heatmap_final")
                    
                    st.info("""
                    **Understanding the Heatmap:**
                    - **Yellow/Brighter colors** indicate roles that are highly compatible with your current profile.
                    - **Darker/Purple colors** indicate roles that are a major stretch or require significant upskilling.
                    - The values are probabilities (%) derived from our Reliable Logistic Regression model.
                    """)

    # --- TAB 6: Skill Associations ---
    with tab6:
        st.header("🤝 Skill Co-occurrence Network Analysis")
        st.write("This analysis uses **Association Rule Mining** (Market Basket Analysis) to discover which skills frequently appear together in job postings.")
        
        target = st.session_state.last_queried_role
        
        # Filter data based on current target role if available, otherwise use all data
        if target:
            assoc_df = jobs_df[jobs_df['job_title'].str.contains(target, case=False, na=False)]
            st.info(f"Analyzing skill associations for roles matching: **{target}** (n={len(assoc_df)})")
            
            # Use all Top 20 skills directly
            all_top_stats = ml.analyze_skills_by_role(assoc_df)
            if not all_top_stats.empty:
                top_20_list = all_top_stats.head(20)['skill_name'].tolist()
            else:
                top_20_list = None
        else:
            assoc_df = jobs_df
            top_20_list = None
            st.info(f"Analyzing skill associations across all {len(assoc_df)} job postings.")
            
        if len(assoc_df) < 20:
            st.warning("⚠️ Insufficient data for reliable association analysis. Try searching for a role with more job postings.")
        else:
            with st.spinner("Mining skill associations..."):
                # 1. Association Rules - using lower support to use all available data patterns
                rules = ml.analyze_skill_associations(assoc_df, min_support=0.01, allowed_skills=top_20_list)
                
                if not top_20_list:
                    st.info("No skill data available for network analysis.")
                else:
                    # 2. Network Graph - Using raw pairwise frequency
                    G, centrality, community_map = ml.create_cooccurrence_network(assoc_df, top_20_list)
                    fig_network = ml.plot_skill_network(G, centrality, community_map)
                    
                    if fig_network:
                        st.plotly_chart(fig_network, use_container_width=True)
                        st.caption("Nodes represent the Top 20 skills. Edge thickness represents how often two skills appear together in the same job posting.")
                    
                    # 3. Insights Table
                    st.subheader("💡 Strategic Skill Insights")
                    st.caption("These insights reveal skill pairings that are statistically much stronger than expected by chance.")
                    
                    # Filter for actionable insights:
                    # - Consequents (You likely also need...) limited to exactly 1 skill for precision
                    # - Skip trivial suggestions that just repeat the role name
                    if not rules.empty:
                        # Define trivial keywords based on target role
                        exclude_set = {target.lower()} | set(s.lower() for s in target.split()) if target else set()
                        exclude_set.update(['management', 'manager', 'lead', 'senior', 'specialist', 'associate', 'professional'])

                        display_rules = rules[
                            (rules['antecedents'].apply(len) <= 2) & 
                            (rules['consequents'].apply(len) == 1) &
                            (rules['consequents'].apply(lambda x: list(x)[0].lower() not in exclude_set)) &
                            (rules['confidence'] >= 0.25)
                        ].sort_values('lift', ascending=False).head(10)
                    else:
                        display_rules = pd.DataFrame()
                    
                    # Format for readability
                    insight_data = []
                    
                    if not display_rules.empty:
                        for _, row in display_rules.iterrows():
                            ant = ", ".join(row['antecedents_list'])
                            cons = ", ".join(row['consequents_list'])
                            conf = row['confidence'] * 100
                            lift = row['lift']
                            
                            insight_data.append({
                                "If you have...": ant,
                                "You likely also need...": cons,
                                "Probability (Confidence)": f"{conf:.1f}%",
                                "Market Strength (Lift)": f"{lift:.2f}x"
                            })
                    
                    if insight_data:
                        st.table(pd.DataFrame(insight_data))
                    else:
                        st.info("No strong strategic skill bundles found for this sample. The skills for this role are largely independent.")
                    
                    # 4. Cluster Breakdown
                    st.subheader("🧩 Skill Clusters (Communities)")
                    # Group nodes by community ID
                    clusters = {}
                    for node, comm_id in community_map.items():
                        if comm_id not in clusters:
                            clusters[comm_id] = []
                        clusters[comm_id].append(node)
                    
                    # Display top clusters
                    cols = st.columns(min(len(clusters), 3))
                    for i, (comm_id, nodes) in enumerate(list(clusters.items())[:3]):
                        with cols[i]:
                            st.markdown(f"**Cluster {i+1}**")
                            # Highlight top 5 central skills in cluster
                            cluster_centrality = {n: centrality[n] for n in nodes}
                            sorted_nodes = sorted(cluster_centrality.items(), key=lambda x: x[1], reverse=True)
                            for node, _ in sorted_nodes[:8]:
                                st.markdown(f"- {node}")

else:
    st.error("Application failed to initialize. Please check data files.")
