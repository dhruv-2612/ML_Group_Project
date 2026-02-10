# GEMINI.md - Global Context for Job Market Intelligence Assistant

## 🎯 PROJECT OVERVIEW

**Project Name:** Job Market Intelligence Assistant  
**Timeline:** 1 week sprint (Feb 11-17, 2026)  
**Course:** Leveraging Machine Learning in Business Applications (MBA, 6th Trimester)  
**Objective:** Build an AI-powered career guidance chatbot that integrates ML clustering, RAG (Retrieval Augmented Generation), and LLMs to provide personalized job market insights for MBA students.

---

## 📋 PROJECT REQUIREMENTS & CONSTRAINTS

### Must-Have Features:
1. **Web scraping** job postings from Naukri.com/Instahyre (1000+ jobs)
2. **ML Analysis:**
   - K-Means clustering for job role pattern discovery
   - Skills extraction and frequency analysis
   - Salary prediction model (Random Forest)
3. **RAG System:**
   - LangChain + Google Gemini integration
   - Vector database (ChromaDB) with job postings + career guides
   - Context-aware query responses
4. **Interactive Chatbot:**
   - Streamlit web interface
   - Conversational memory
   - Personalized career recommendations
5. **ML + LLM Integration:**
   - ML models provide quantitative insights
   - LLM synthesizes insights into natural language
   - Combined responses (Session 10 course alignment)

### Technical Constraints:
- **No paid services** - Use free tiers only
- **Free data sources** - Public web scraping or synthetic data
- **Vibe coding** - All code generated via Gemini CLI
- **Simple deployment** - Streamlit Community Cloud (free)
- **Fast development** - Prioritize working demo over perfection

### Target Users:
- MBA students (primary: 2nd year looking for jobs)
- Engineering graduates transitioning to business roles
- Career switchers needing market intelligence

---

## 🏗️ SYSTEM ARCHITECTURE

### Data Flow:
```
Web Scraping → Data Cleaning → ML Analysis → Vector Store Creation
                                      ↓
User Query → Intent Detection → ML Insights + RAG Retrieval → LLM Synthesis → Response
```

### Component Stack:
- **Scraping:** BeautifulSoup, Requests, Selenium (fallback)
- **ML:** Scikit-learn, Pandas, NumPy
- **NLP/Embeddings:** LangChain, Google Generative AI Embeddings
- **Vector DB:** ChromaDB (persistent storage)
- **LLM:** Google Gemini 1.5 Flash (free tier: 1500 requests/day)
- **Frontend:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Deployment:** Streamlit Community Cloud

---

## 📁 PROJECT STRUCTURE

```
job-intelligence-assistant/
├── data/
│   ├── raw/                      # Scraped job CSVs
│   │   ├── naukri_jobs_*.csv
│   │   ├── instahyre_jobs_*.csv
│   │   └── synthetic_jobs.csv    # Backup if scraping fails
│   ├── processed/                # Cleaned data & models
│   │   ├── cleaned_jobs.csv
│   │   ├── job_clusters.csv
│   │   ├── skills_analysis.csv
│   │   ├── salary_model.pkl
│   │   └── *.html (visualizations)
│   └── embeddings/               # ChromaDB vector store
│       └── chroma_db/
├── src/
│   ├── scraper.py                # Job scraping functions
│   ├── utils.py                  # Data cleaning utilities
│   ├── ml_analyzer.py            # ML models & analysis
│   ├── rag_system.py             # LangChain RAG implementation
│   ├── chatbot.py                # Conversational logic
│   ├── pipeline.py               # End-to-end orchestration
│   └── generate_sample_data.py   # Synthetic data fallback
├── tests/
│   └── test_system.py            # Basic validation tests
├── notebooks/
│   └── eda.ipynb                 # Quick data exploration
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .env                          # API keys (gitignored)
├── .env.example                  # API key template
├── .gitignore
├── README.md                     # Project documentation
├── DEMO.md                       # Demo script for presentation
├── ARCHITECTURE.md               # Technical deep-dive
└── DEPLOYMENT.md                 # Deployment instructions
```

---

## 📊 DATA SCHEMAS

### Job Posting Schema (CSV/DataFrame):
```python
{
    'job_id': str,                    # Unique identifier
    'job_title': str,                 # e.g., "Product Manager"
    'company': str,                   # e.g., "Flipkart"
    'location': str,                  # "Bangalore", "Mumbai", etc.
    'experience_min': float,          # Minimum years (e.g., 2.0)
    'experience_max': float,          # Maximum years (e.g., 5.0)
    'experience_level': str,          # "Entry", "Mid", "Senior", "Lead"
    'skills_list': list,              # ['Python', 'SQL', 'A/B Testing']
    'skills_text': str,               # Comma-separated for storage
    'job_description': str,           # Full cleaned description
    'salary_min': float,              # In LPA (lakhs per annum)
    'salary_max': float,              # In LPA
    'salary_currency': str,           # "INR"
    'posted_date': datetime,          # When job was posted
    'job_url': str,                   # Link to original posting
    'cluster_id': int,                # Assigned by K-Means
    'cluster_label': str,             # Human-readable cluster name
    'scraped_date': datetime          # When we scraped it
}
```

### User Profile Schema:
```python
{
    'current_role': str,              # "MBA Student", "Software Engineer"
    'target_role': str,               # "Product Manager", "Data Analyst"
    'years_experience': float,        # Total work experience
    'current_skills': list,           # Skills user already has
    'education': str,                 # "MBA", "B.Tech", etc.
    'preferred_locations': list,      # ["Bangalore", "Mumbai"]
    'salary_expectation': tuple       # (min, max) in LPA
}
```

### RAG Document Metadata:
```python
{
    'doc_id': str,
    'doc_type': str,                  # "job_posting" or "career_guide"
    'source': str,                    # "naukri", "instahyre", "synthetic"
    'job_title': str,                 # For job docs
    'company': str,                   # For job docs
    'location': str,
    'experience_level': str,
    'cluster_id': int,
    'salary_range': str,              # "15-25 LPA"
    'top_skills': list,               # Top 5 skills for this doc
    'category': str,                  # For guide docs: "career_transition", "interview_prep"
    'relevance_score': float,         # 0-1 importance weight
    'created_date': datetime
}
```

---

## 🔧 KEY FUNCTION SIGNATURES

### src/scraper.py
```python
def scrape_naukri_jobs(
    keywords: list[str],              # e.g., ["Product Manager", "Data Analyst"]
    locations: list[str],             # e.g., ["Bangalore", "Mumbai"]
    num_pages: int = 10,
    delay: float = 2.0                # Rate limiting
) -> pd.DataFrame:
    """Scrape job postings from Naukri.com"""

def scrape_instahyre_jobs(
    keywords: list[str],
    locations: list[str],
    num_pages: int = 5
) -> pd.DataFrame:
    """Backup scraper for Instahyre"""

def parse_job_card(soup_element: BeautifulSoup) -> dict:
    """Extract structured data from single job listing"""
```

### src/utils.py
```python
def clean_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main cleaning pipeline: dedup, standardize, parse fields"""

def extract_skills(text: str, skill_dict: dict) -> list[str]:
    """Extract skills from job description using regex"""

def categorize_experience_level(years: float) -> str:
    """Map years to experience category"""

def parse_salary_text(salary_str: str) -> tuple[float, float]:
    """Convert salary text to (min, max) in LPA"""
```

### src/ml_analyzer.py
```python
def cluster_job_roles(
    df: pd.DataFrame,
    n_clusters: int = 6,
    max_features: int = 500
) -> tuple[pd.DataFrame, np.ndarray, list]:
    """K-Means clustering on TF-IDF job descriptions
    Returns: (df_with_clusters, cluster_centers, top_keywords_per_cluster)"""

def analyze_skills_by_role(
    df: pd.DataFrame,
    role_filter: str = None
) -> pd.DataFrame:
    """Skill frequency analysis
    Returns: columns [skill, count, percentage, avg_salary_with_skill]"""

def train_salary_predictor(
    df: pd.DataFrame
) -> tuple[RandomForestRegressor, pd.DataFrame, dict]:
    """Train salary prediction model
    Returns: (model, feature_importance_df, evaluation_metrics)"""

def predict_salary(
    model: RandomForestRegressor,
    user_profile: dict
) -> tuple[float, tuple[float, float]]:
    """Predict salary with confidence interval
    Returns: (predicted_salary, (lower_bound, upper_bound))"""

def get_cluster_characteristics(
    df: pd.DataFrame,
    cluster_id: int
) -> dict:
    """Extract cluster insights: top keywords, avg salary, common locations"""

def calculate_skill_salary_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Statistical analysis of skill premiums"""
```

### src/rag_system.py
```python
def create_vector_store(
    jobs_df: pd.DataFrame,
    persist_dir: str = "data/embeddings/chroma_db"
) -> Chroma:
    """Create ChromaDB vector store from job postings"""

def add_career_guides_to_store(
    vectorstore: Chroma,
    guides: list[dict]
) -> Chroma:
    """Add synthetic career guidance documents"""

def load_vector_store(
    persist_dir: str = "data/embeddings/chroma_db"
) -> Chroma:
    """Load existing vector store"""

def create_rag_chain(
    vectorstore: Chroma,
    temperature: float = 0.3,
    k: int = 8
) -> RetrievalQA:
    """Create LangChain RAG query chain"""

def query_job_intelligence(
    question: str,
    rag_chain: RetrievalQA,
    retriever: VectorStoreRetriever
) -> dict:
    """Query RAG system
    Returns: {answer: str, source_documents: list}"""

def enhanced_query_with_ml(
    question: str,
    rag_chain: RetrievalQA,
    ml_analyzer: object,
    jobs_df: pd.DataFrame
) -> dict:
    """Combine ML insights with RAG responses
    Returns: {answer: str, ml_insights: dict, sources: list, intent: str}"""
```

### src/chatbot.py
```python
def initialize_conversation_memory() -> ConversationBufferMemory:
    """Setup LangChain conversation memory"""

def create_conversational_chain(
    vectorstore: Chroma,
    memory: ConversationBufferMemory
) -> ConversationalRetrievalChain:
    """Create conversational RAG chain with history"""

def detect_query_intent(query: str) -> str:
    """Classify query type: skills/salary/career_path/interview/general"""

def process_user_query(
    query: str,
    chain: ConversationalRetrievalChain,
    user_profile: dict,
    ml_analyzer: object,
    jobs_df: pd.DataFrame
) -> dict:
    """Main chatbot logic with intent-based routing
    Returns: {answer, intent, ml_insights, source_jobs, follow_ups}"""

def generate_personalized_career_plan(
    user_profile: dict,
    jobs_df: pd.DataFrame,
    ml_analyzer: object,
    rag_chain: RetrievalQA
) -> str:
    """Multi-step career planning analysis
    Returns: Comprehensive markdown report"""
```

### src/pipeline.py
```python
def run_full_pipeline(
    keywords_list: list[str],
    locations_list: list[str],
    num_pages: int = 10,
    use_synthetic: bool = False
) -> dict:
    """Execute complete workflow: scrape → clean → ML → RAG
    Returns: {status: str, stats: dict, files_created: list}"""
```

---

## 🎨 CODING STYLE GUIDELINES

### General Principles:
1. **Clarity over cleverness** - Code should be self-explanatory
2. **Type hints everywhere** - Use Python 3.10+ type annotations
3. **Docstrings for all functions** - Google style format
4. **Error handling** - Try-except with informative messages
5. **Logging over print** - Use logging module for production code
6. **Modular functions** - Each function does one thing well
7. **Constants in CAPS** - Magic numbers as named constants

### Example Function Template:
```python
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def example_function(
    data: pd.DataFrame,
    param1: str,
    param2: int = 10,
    optional_param: Optional[dict] = None
) -> tuple[pd.DataFrame, dict]:
    """
    Short one-line description.

    Longer explanation of what the function does, edge cases,
    and any important implementation details.

    Args:
        data: Description of data parameter
        param1: Description of param1
        param2: Description with default value explanation
        optional_param: Description of optional parameter

    Returns:
        Tuple containing:
            - DataFrame with processed results
            - Dictionary with metadata/statistics

    Raises:
        ValueError: If data is empty
        KeyError: If required columns missing

    Example:
        >>> df = pd.DataFrame({'col': [1, 2, 3]})
        >>> result_df, stats = example_function(df, "test")
        >>> print(stats['count'])
        3
    """
    # Input validation
    if data.empty:
        raise ValueError("Input DataFrame cannot be empty")

    logger.info(f"Processing {len(data)} records with param1={param1}")

    try:
        # Main logic here
        result = data.copy()
        # ... processing ...

        stats = {
            'count': len(result),
            'param_used': param1
        }

        logger.info(f"Successfully processed {stats['count']} records")
        return result, stats

    except Exception as e:
        logger.error(f"Error in example_function: {str(e)}")
        raise
```

### Naming Conventions:
- **Functions:** `snake_case` - `calculate_salary_range()`
- **Classes:** `PascalCase` - `JobAnalyzer`
- **Variables:** `snake_case` - `job_count`, `user_profile`
- **Constants:** `UPPER_SNAKE_CASE` - `MAX_RETRIES`, `DEFAULT_LOCATION`
- **Private methods:** `_leading_underscore` - `_internal_helper()`

### Import Organization:
```python
# Standard library
import os
import sys
from typing import List, Dict, Optional

# Third-party libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# LangChain imports
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Local imports
from src.utils import clean_job_data
from src.ml_analyzer import cluster_job_roles
```

---

## 🔐 ENVIRONMENT VARIABLES (.env)

```bash
# Google Gemini API
GEMINI_API_KEY=your_api_key_here

# Optional: For extended functionality
OPENAI_API_KEY=backup_if_needed

# Scraping settings (optional overrides)
MAX_RETRIES=3
REQUEST_DELAY=2.0
USER_AGENT="Mozilla/5.0 (compatible; JobBot/1.0)"

# ChromaDB settings
CHROMA_PERSIST_DIR=data/embeddings/chroma_db

# Logging
LOG_LEVEL=INFO
```

---

## 📊 ML MODEL SPECIFICATIONS

### K-Means Clustering:
```python
# Configuration
N_CLUSTERS = 6  # Can tune with elbow method
MAX_FEATURES = 500  # TF-IDF vocabulary size
NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
STOP_WORDS = 'english'
RANDOM_STATE = 42  # For reproducibility

# Expected clusters (may vary):
# 1. Technical Product Manager
# 2. Growth/Marketing PM
# 3. B2B/Enterprise PM
# 4. Data Analyst (Technical)
# 5. Business Analyst (Domain-focused)
# 6. Management Consultant
```

### Salary Prediction Model:
```python
# Random Forest Configuration
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Features:
SALARY_FEATURES = [
    'years_experience',
    'num_skills',
    'has_technical_skills',
    'has_mba',
    'location',  # One-hot encoded
    'experience_level',  # Ordinal encoded
    'cluster_id'
]

# Target: salary_mid (average of min and max)
```

### Skills Extraction:
```python
# Comprehensive skill dictionary
SKILL_CATEGORIES = {
    'programming': ['Python', 'R', 'SQL', 'JavaScript', 'Java'],
    'data_tools': ['Tableau', 'Power BI', 'Excel', 'Looker', 'Metabase'],
    'product_tools': ['Figma', 'JIRA', 'Confluence', 'Miro', 'Notion'],
    'analytics': ['A/B Testing', 'Google Analytics', 'Mixpanel', 'Amplitude'],
    'ml_ai': ['Machine Learning', 'NLP', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Prompt Engineering'],
    'cloud': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'],
    'soft_skills': ['Communication', 'Leadership', 'Problem Solving', 'Stakeholder Management'],
    'domain': ['Financial Modeling', 'Market Research', 'Strategy', 'Operations']
}

# Extraction method: Case-insensitive regex with word boundaries
```

---

## 🤖 RAG SYSTEM CONFIGURATION

### Embedding Model:
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Embedding dimensions: 768
# Context window: 2048 tokens per document
```

### Vector Store (ChromaDB):
```python
from langchain.vectorstores import Chroma

vectorstore = Chroma(
    collection_name="job_postings_2026",
    embedding_function=embeddings,
    persist_directory="data/embeddings/chroma_db"
)

# Retrieval settings:
RETRIEVAL_K = 8  # Number of documents to retrieve
SEARCH_TYPE = "similarity"  # or "mmr" for diversity
```

### LLM Configuration:
```python
from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,  # Lower = more focused, higher = more creative
    max_output_tokens=2048
)

# Rate limits (free tier):
# - 15 requests per minute
# - 1500 requests per day
# - 1 million tokens per day
```

### Prompt Template:
```python
CAREER_ADVISOR_TEMPLATE = """You are an expert career advisor with access to real-time job market data from February 2026.

Your role is to provide:
1. Data-driven insights backed by statistics from the context
2. Actionable, specific recommendations (not generic advice)
3. Relevant examples from actual job postings
4. Realistic timelines and learning paths

Context from job market database:
{context}

User Profile:
{user_profile}

User Question: {question}

Provide a comprehensive answer using markdown formatting:
- Use **bold** for key insights
- Use bullet points for lists
- Include emojis for visual appeal (📊 💡 🎯 ✅ ⚠️)
- Cite specific data points (e.g., "87% of Product Manager roles require...")
- If discussing salaries, provide ranges with context
- End with 2-3 actionable next steps

Answer:"""
```

---

## 🎨 STREAMLIT UI SPECIFICATIONS

### Page Configuration:
```python
st.set_page_config(
    page_title="Job Intelligence Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Color Scheme:
```python
PRIMARY_COLOR = "#1E88E5"  # Blue
SECONDARY_COLOR = "#43A047"  # Green
ACCENT_COLOR = "#FFA726"  # Orange
BG_COLOR = "#F5F5F5"  # Light gray
TEXT_COLOR = "#212121"  # Dark gray
```

### Custom CSS:
```python
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .skill-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background-color: #E3F2FD;
        border-radius: 1rem;
        font-size: 0.875rem;
    }
</style>
"""
```

### Session State Management:
```python
# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if 'ml_analyzer' not in st.session_state:
    st.session_state.ml_analyzer = None
```

---

## 📈 VISUALIZATION STANDARDS

### Plotly Chart Configuration:
```python
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
}

PLOTLY_LAYOUT = {
    'font': {'family': 'Arial, sans-serif', 'size': 12},
    'plot_bgcolor': '#FFFFFF',
    'paper_bgcolor': '#F5F5F5',
    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
}
```

### Chart Types:
1. **Cluster Visualization:** 2D scatter (PCA reduced)
2. **Skills Analysis:** Horizontal bar chart
3. **Salary Distribution:** Box plot + violin plot
4. **Trend Analysis:** Line chart with confidence intervals
5. **Skills Co-occurrence:** Heatmap

---

## 🔍 SAMPLE CAREER GUIDE DOCUMENTS

These should be added to the vector store for RAG context:

### Guide 1: SDE to PM Transition
```markdown
# Transitioning from Software Engineer to Product Manager

Common path for technical professionals moving into product roles.

**Key Advantages:**
- Technical credibility with engineering teams
- Understanding of feasibility and trade-offs
- Data-driven decision making skills

**Skill Gaps to Address:**
1. Product strategy and vision
2. User research and empathy
3. Business metrics and P&L understanding
4. Stakeholder communication (non-technical)
5. UI/UX fundamentals

**Learning Path (3-6 months):**
- Week 1-4: Product thinking frameworks (CIRCLES, AARRR)
- Week 5-8: User research methods, conduct 10+ user interviews
- Week 9-12: Analytics tools (Mixpanel, Amplitude)
- Week 13-16: Design tools (Figma basics)
- Week 17+: Side project - build and launch a small product

**Target Roles:**
Start with Associate PM or Technical PM positions at product companies.

**Salary Expectations:**
15-25 LPA for entry-level PM with 2-3 years SDE experience in Bangalore/Mumbai.
```

### Guide 2: MBA Graduate Job Search
```markdown
# Job Search Strategy for MBA Graduates (2026)

**High-Demand Roles:**
1. Product Management (40% of MBA placements)
2. Strategy & Consulting (25%)
3. Business Analytics (20%)
4. Marketing (10%)
5. Operations (5%)

**Skills That Matter in 2026:**
- Technical literacy (SQL, basic Python) - differentiator
- Data storytelling and visualization
- AI/ML awareness (not deep expertise)
- Prompt engineering for productivity
- Cross-functional collaboration

**Application Strategy:**
- Target 30-50 companies based on career goals
- Customize resume for each role category
- Leverage alumni networks (80% of jobs come through referrals)
- Build online presence (LinkedIn, Medium articles)
- Prepare case studies portfolio

**Interview Prep Timeline:**
- 2 months before target start date
- Practice 40-50 case studies
- Mock interviews: 10-15 sessions
- Company research: 2-3 hours per company
```

### Guide 3: Data Analyst Skills 2026
```markdown
# Essential Skills for Data Analysts in 2026

**Technical Skills (Must-Have):**
1. SQL (Advanced) - 95% of jobs require
2. Python/R - Choose one, focus on pandas, NumPy
3. Data visualization - Tableau or Power BI
4. Excel - Still relevant for 70% of roles
5. Statistical concepts - hypothesis testing, regression

**Emerging Requirements:**
1. Basic ML understanding - Random Forests, clustering
2. Cloud platforms - Bigquery, Snowflake, Redshift
3. Version control - Git basics
4. ETL/Data pipelines - Understanding of dbt, Airflow
5. LLM integration - Using AI for analysis automation

**Domain Knowledge:**
Choose specialization: E-commerce, Fintech, SaaS, Healthcare

**Salary Benchmarks (2026, India):**
- Entry (0-2 yrs): 6-12 LPA
- Mid (2-5 yrs): 12-20 LPA
- Senior (5+ yrs): 20-35 LPA

**Learning Resources:**
- Mode Analytics SQL Tutorial
- DataCamp Python for Data Science
- Kaggle competitions for practice
```

*(Add 12-15 more such guides covering various topics)*

---

## 🚨 ERROR HANDLING PATTERNS

### Web Scraping Errors:
```python
try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if response.status_code == 429:
        logger.warning("Rate limit hit, waiting 60 seconds...")
        time.sleep(60)
        # Retry logic
    elif response.status_code == 403:
        logger.error("Access forbidden, try rotating user agents")
        # Switch to backup scraper
except requests.exceptions.Timeout:
    logger.error(f"Timeout while scraping {url}")
    # Skip this URL, continue with next
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    # Log but don't crash
```

### API Errors:
```python
try:
    response = llm.invoke(prompt)
except Exception as e:
    if "quota" in str(e).lower():
        logger.error("API quota exceeded")
        st.error("⚠️ API limit reached. Please try again later.")
        # Fall back to cached response or simplified answer
    elif "invalid api key" in str(e).lower():
        logger.error("Invalid API key")
        st.error("🔑 Please check your API key configuration")
    else:
        logger.error(f"API error: {str(e)}")
        st.error("❌ Service temporarily unavailable")
```

### Data Processing Errors:
```python
try:
    df = pd.read_csv(filepath)
    assert not df.empty, "DataFrame is empty"
    assert 'job_title' in df.columns, "Missing required column"
except FileNotFoundError:
    logger.error(f"File not found: {filepath}")
    # Generate synthetic data as fallback
    df = generate_synthetic_job_data()
except AssertionError as e:
    logger.error(f"Data validation failed: {str(e)}")
    # Clean and retry
except Exception as e:
    logger.error(f"Data processing error: {str(e)}")
    raise
```

---

## 📊 SUCCESS METRICS & VALIDATION

### Data Quality Checks:
```python
def validate_scraped_data(df: pd.DataFrame) -> dict:
    """Validate scraped job data quality"""
    checks = {
        'total_records': len(df),
        'duplicates': df.duplicated().sum(),
        'missing_titles': df['job_title'].isna().sum(),
        'missing_companies': df['company'].isna().sum(),
        'missing_descriptions': df['job_description'].isna().sum(),
        'valid_urls': df['job_url'].apply(lambda x: x.startswith('http')).sum(),
        'date_range': (df['posted_date'].min(), df['posted_date'].max())
    }

    # Quality gates
    assert checks['total_records'] >= 500, "Insufficient data"
    assert checks['duplicates'] / checks['total_records'] < 0.1, "Too many duplicates"
    assert checks['missing_titles'] == 0, "Missing job titles"

    return checks
```

### ML Model Evaluation:
```python
# Clustering quality
SILHOUETTE_SCORE_THRESHOLD = 0.3  # Minimum acceptable
CALINSKI_HARABASZ_THRESHOLD = 100

# Salary prediction
MIN_R2_SCORE = 0.6
MAX_MAE_LPA = 5.0  # Maximum acceptable error in lakhs

# Skills extraction
MIN_RECALL = 0.7  # Should catch 70%+ of actual skills
```

### RAG System Quality:
```python
# Test queries with expected response characteristics
TEST_QUERIES = [
    {
        'query': "What skills do Product Managers need?",
        'expected_keywords': ['SQL', 'A/B testing', 'Figma', 'analytics'],
        'expected_stats': True  # Should include percentages
    },
    {
        'query': "Salary for Data Analyst in Bangalore",
        'expected_keywords': ['LPA', 'Bangalore', 'range'],
        'expected_numbers': True
    }
]

def validate_rag_response(response: str, expected: dict) -> bool:
    """Check if RAG response meets quality standards"""
    # Check for expected keywords
    keyword_match = sum(kw.lower() in response.lower() for kw in expected['expected_keywords'])

    # Check for data citations (numbers, percentages)
    has_stats = bool(re.search(r'\d+%', response)) if expected.get('expected_stats') else True

    # Check response length (not too short)
    adequate_length = len(response) > 200

    return keyword_match >= len(expected['expected_keywords']) * 0.7 and has_stats and adequate_length
```

---

## 🎯 DEMO SCENARIOS

### Scenario 1: MBA Student Profile (Your Use Case)
```python
DEMO_PROFILE_1 = {
    'name': "MBA Student (You)",
    'current_role': "2nd Year MBA, Former SDE",
    'target_role': "Product Manager",
    'years_experience': 2.0,
    'current_skills': ['Python', 'SQL', 'Data Analysis', 'Git'],
    'education': "MBA + B.Tech Computer Science",
    'preferred_locations': ["Bangalore", "Mumbai"],
    'salary_expectation': (18, 28)  # LPA
}

DEMO_QUERIES_1 = [
    "What skills do Product Managers need in 2026?",
    "I'm an SDE wanting to become a PM. What should I learn?",
    "What salary should I expect as an Associate PM in Bangalore?",
    "Show me job matches for my profile"
]
```

### Scenario 2: Career Switcher
```python
DEMO_PROFILE_2 = {
    'name': "Career Switcher",
    'current_role': "Business Analyst",
    'target_role': "Data Analyst",
    'years_experience': 3.0,
    'current_skills': ['Excel', 'PowerPoint', 'SQL (Basic)', 'Market Research'],
    'education': "MBA",
    'preferred_locations': ["Mumbai", "Pune"],
    'salary_expectation': (12, 18)
}

DEMO_QUERIES_2 = [
    "How do I transition from Business Analyst to Data Analyst?",
    "What technical skills do I need to learn?",
    "What's the typical salary difference between BA and DA roles?"
]
```

---

## ⏱️ DAILY DEVELOPMENT CHECKPOINTS

### Day 1 Checkpoint:
- [ ] ≥1000 job records scraped (or synthetic data generated)
- [ ] CSV saved with all required columns
- [ ] Data cleaning functions working
- [ ] No critical errors in scraper

### Day 2 Checkpoint:
- [ ] K-Means clustering produces 5-7 interpretable clusters
- [ ] Skills extraction identifies 50+ unique skills
- [ ] Salary model achieves R² > 0.6
- [ ] All visualizations render correctly

### Day 3 Checkpoint:
- [ ] ChromaDB vector store created and persisted
- [ ] Test query returns relevant results
- [ ] ML + RAG integration shows combined insights
- [ ] Response quality meets validation checks

### Day 4 Checkpoint:
- [ ] Streamlit app launches without errors
- [ ] All 4 tabs functional
- [ ] Chatbot maintains conversation context
- [ ] Charts display interactively

### Day 5 Checkpoint:
- [ ] Full pipeline runs end-to-end
- [ ] Can reproduce results from scratch
- [ ] Error handling prevents crashes
- [ ] Logging captures important events

### Day 6 Checkpoint:
- [ ] Demo script rehearsed
- [ ] Documentation complete
- [ ] Known bugs documented/worked around
- [ ] Backup plan tested

### Day 7 Checkpoint:
- [ ] Deployed to Streamlit Cloud
- [ ] Public URL accessible
- [ ] Presentation deck ready
- [ ] Team confident in demo

---

## 🎓 COURSE ALIGNMENT MATRIX

Map project components to course sessions (for project report):

| Session | Topic | Our Implementation |
|---------|-------|-------------------|
| 2 | Generating code with AI | Used Gemini CLI for 80% of codebase |
| 3 | ML in Retail/Business | Applied ML to career/job domain |
| 4 | ML Learning Process | K-Means clustering, hyperparameter tuning |
| 5 | Model Evaluation | Silhouette score, R², MAE metrics |
| 6 | Linear Regression | Salary prediction (baseline model) |
| 9 | k-NN Classification | Job similarity matching |
| 10 | **ML + Gen AI Integration** | Core innovation: ML insights → LLM synthesis |
| 11 | Random Forest | Salary prediction (primary model) |
| 13 | K-Means Clustering | Job role pattern discovery |
| 14 | PCA/Dimensionality Reduction | Cluster visualization in 2D |
| 15 | Recommender Systems | Job matching based on profile |
| 16 | NLP | Skills extraction, text processing |
| 17 | Ensemble Methods | Random Forest ensemble for predictions |
| 19 | Data Governance | Ethical scraping, privacy considerations |

---

## 📝 RESPONSE FORMATTING GUIDELINES

### For Career Advice Responses:
```markdown
# Use this structure:

🎯 **KEY INSIGHT**
One-sentence main takeaway

📊 **DATA-DRIVEN ANALYSIS**
• Statistic 1 (cite percentage)
• Statistic 2 (cite count)
• Trend observation

💡 **ACTIONABLE RECOMMENDATIONS**
1. **Priority 1** (High Impact)
   - Specific action
   - Expected outcome
   - Timeline

2. **Priority 2** (Quick Win)
   - Specific action
   - Expected outcome

3. **Priority 3** (Strategic)
   - Specific action
   - Expected outcome

📚 **RELEVANT JOBS**
• [Job Title] at [Company] - [Key requirement]
• [Job Title] at [Company] - [Key requirement]

🎯 **NEXT STEPS**
Clear 3-step action plan with timeline
```

### For Skills Gap Analysis:
```markdown
✅ **SKILLS YOU HAVE** (X/Y match)
• Skill 1
• Skill 2

⚠️ **CRITICAL GAPS** (Priority order)
1. **Skill Name** ⭐ HIGH PRIORITY
   → Found in Z% of target roles
   → Learning path: [Specific resource]
   → Time investment: X weeks

2. **Skill Name** 
   → Found in Z% of target roles
   → Learning path: [Specific resource]

🎯 **MATCH SCORE:** XX%
Estimated prep time: X-Y months
```

---

## 🚀 DEPLOYMENT CHECKLIST

### Pre-Deployment:
- [ ] requirements.txt includes all dependencies with versions
- [ ] .env variables documented in .env.example
- [ ] Secrets configured in Streamlit Cloud
- [ ] Data files under 100MB (Streamlit limit)
- [ ] No hardcoded API keys in codebase
- [ ] .gitignore includes data/, .env, __pycache__

### Post-Deployment:
- [ ] Public URL loads without errors
- [ ] Test all demo scenarios
- [ ] Check mobile responsiveness
- [ ] Monitor API usage (stay under free tier)
- [ ] Share URL in project report

---

## 💡 OPTIMIZATION TIPS

### If Running Slow:
1. **Reduce retrieval size:** k=8 → k=5
2. **Cache ML models:** Use @st.cache_resource
3. **Smaller dataset:** 1000 jobs instead of 1500
4. **Simplify embeddings:** Use smaller documents

### If API Quota Exceeded:
1. **Increase temperature:** More creative = fewer API calls for same quality
2. **Cache responses:** Store common queries
3. **Batch processing:** Process in groups rather than one-by-one
4. **Use Gemini Flash:** Faster, cheaper model

### If Out of Memory:
1. **Stream processing:** Don't load all data at once
2. **Chunk embeddings:** Process 100 jobs at a time
3. **Clear session state:** Reset periodically

---

## 🎤 PRESENTATION TALKING POINTS

### Opening (30 seconds):
"Imagine you're an MBA student with 1,000 job postings open in browser tabs. You're trying to figure out: What skills should I learn? What salary should I expect? Which companies should I target? This process takes 40+ hours of manual research.

We built an AI system that does this analysis in 15 minutes—combining machine learning for pattern discovery with large language models for personalized advice."

### Demo Introduction (20 seconds):
"Let me show you how it works. I'll use my own profile as an example: 2nd year MBA, former software engineer, targeting Product Manager roles."

### Technical Innovation Highlight (30 seconds):
"The key innovation here is the integration between machine learning and generative AI—exactly what we learned in Session 10 of our course. Our ML models analyze 1,000+ jobs to find patterns, and the LLM translates those insights into actionable career advice. Together, they're more powerful than either alone."

### Business Impact Closing (20 seconds):
"This isn't just a course project—it's a tool we're actually using for our own placements. If deployed campus-wide, it could save 5,000 MBA students over 200,000 hours of research time annually."

---

## 📚 QUICK REFERENCE: COMMON COMMANDS

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Run full pipeline
python src/pipeline.py --keywords "Product Manager,Data Analyst" --locations "Bangalore,Mumbai" --pages 10

# Generate synthetic data (fallback)
python src/generate_sample_data.py --n_jobs 1000 --output data/raw/synthetic_jobs.csv

# Start Streamlit app
streamlit run app.py

# Run tests
pytest tests/

# Deploy to Streamlit Cloud
git push origin main
# Then connect repo in Streamlit Cloud UI
```

---

## 🎯 FINAL CHECKLIST BEFORE SUBMISSION

### Code Quality:
- [ ] All functions have docstrings
- [ ] Type hints added to function signatures
- [ ] No hardcoded paths (use relative paths)
- [ ] Error handling for all external calls
- [ ] Logging instead of print statements
- [ ] Code passes basic tests

### Documentation:
- [ ] README.md complete with setup instructions
- [ ] ARCHITECTURE.md explains technical design
- [ ] DEMO.md has presentation script
- [ ] DEPLOYMENT.md has deployment guide
- [ ] Code comments explain complex logic

### Functionality:
- [ ] Scraper or synthetic data works
- [ ] ML models train successfully
- [ ] RAG system returns relevant answers
- [ ] Streamlit app loads and works
- [ ] All demo scenarios tested

### Presentation:
- [ ] Deck ready (10-12 slides)
- [ ] Demo rehearsed (3-5 minutes)
- [ ] Backup screenshots captured
- [ ] Public URL tested on mobile/desktop
- [ ] Q&A prep done

---

**END OF GLOBAL CONTEXT**

*Use this document as reference for all Gemini CLI prompts. Include relevant sections in your prompts by saying "According to GEMINI.md specifications..." to ensure consistency.*
