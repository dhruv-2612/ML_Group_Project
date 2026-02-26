# GEMINI.md - Global Context for Job Market Intelligence Assistant

## 🎯 PROJECT OVERVIEW

**Project Name:** Job Market Intelligence Assistant  
**Timeline:** 1 week sprint (Feb 11-17, 2026) - Finalized: Feb 19, 2026  
**Course:** Leveraging Machine Learning in Business Applications (MBA, 6th Trimester)  
**Objective:** Build an AI-powered career guidance chatbot that integrates ML clustering, RAG (Retrieval Augmented Generation), and LLMs to provide personalized, strictly data-driven job market insights.

---

## 📋 PROJECT REQUIREMENTS & CONSTRAINTS

### Must-Have Features:
1. **Autonomous Deep Scraping**:
   - Selenium-based "Deep Scraper" that visits individual job URLs for full detail extraction.
   - Smart pagination logic with `start_page` support to avoid redundant scanning.
   - **Autonomous Feedback Loop**: Dynamic scraping triggers when data sufficiency thresholds (e.g., <15 jobs) are not met for a specific role/location.
2. **Strictly Live Data**:
   - **NO synthetic data fallback**. All insights are grounded in the last 14 days of job postings.
   - Automated cleanup of stale data (older than 14 days) to maintain market relevance.
3. **Supervised Classification**:
   - Jobs are classified into **20 industry-standard clusters** using TF-IDF and Cosine Similarity against golden profiles.
   - Replaces unsupervised K-Means to ensure professional, accurate categorization.
4. **Hybrid Salary Intelligence**:
   - **Statistical Benchmarking**: Provides robust market stats (Median, IQR, Mean) as the primary source of truth.
   - **Predictive Engine**: A **Random Forest Regressor** trained on real-time data to provide specific salary estimates based on experience and skill count.
5. **Career Transition Modeling**:
   - **Logistic Regression Model**: Predicts the likelihood (probability %) of a successful transition between roles based on skill overlap, experience gaps, and location matches.
6. **Efficient RAG & Indexing**:
   - **Incremental Indexing**: Only new, unique job postings are embedded, saving 80%+ of API quota.
   - **Dynamic Career Guide Generation**: Automatically generates and indexes expert career roadmaps for new roles using Gemini if a guide doesn't exist.
   - Uses **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) for cost-effective vectorization.
7. **Unified User Experience**:
   - **6-Tab Layout**: "Job Inquiry" (Chat), "ML Insights" (Stats), "Profile Analysis" (Resume Gap), "Job Matches", "Career Transition" (LogReg), "Skill Associations" (Graph).
   - **Resume Analysis**: Automated skill extraction from PDF resumes for gap analysis.
   - **Context Sync**: Chatbot queries automatically update the target role and location for all dashboards.
8. **Skill Co-occurrence Network Analysis**:
   - **Market Basket Analysis**: Uses Apriori to find skill associations (Support, Confidence, Lift).
   - **Network Visualization**: Interactive Graph Theory nodes showing skill clusters and centrality.

### Technical Constraints:
- **Model**: `gemini-2.5-flash` (via `google-genai` and `langchain-classic`).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Stack**: LangChain 1.x, Streamlit, Selenium, ChromaDB, Scikit-learn, mlxtend, networkx.
- **Environment**: Managed via `.env` with `MODEL_NAME` and `HUGGINGFACE_API_TOKEN`.

---

## 🏗️ SYSTEM ARCHITECTURE

### Data Flow:
```
User Query (Tab 1) → Intent & Entity Detection (Role/Location) → Context Update
                                      ↓
If Low Data → Dynamic Scraping Loop (Selenium) → Cleaning → Classification → RAG Update
                                      ↓
Tab 2/3/4/5/6 Update (Filtered by Context) ←→ LLM Response (Augmented with Stats + Real Jobs)
```

### Component Stack:
- **Scraping:** Selenium (Headless Chrome), BeautifulSoup4, WebDriver Manager.
- **ML Engine:** Scikit-learn (TF-IDF, Cosine Similarity, RandomForest, LogisticRegression), mlxtend (Apriori).
- **Network Analysis:** NetworkX (Degree Centrality, Modularity Communities).
- **NLP/RAG:** LangChain 1.x, Google Generative AI (Chat), HuggingFace (Embeddings).
- **Vector DB:** ChromaDB (Persistent storage with Soft Refresh logic).
- **Frontend:** Streamlit 1.35+.

---

## 📁 PROJECT STRUCTURE

```
job-intelligence-assistant/
├── data/
│   ├── raw/                      # Scraped deep-detailed CSVs
│   ├── processed/                # Cleaned data & Persistent Dictionary
│   │   ├── cleaned_jobs.csv      # Unified source of truth (Fresh 14 days)
│   │   ├── job_clusters.csv      # Classified data with 20 categories
│   │   ├── known_roles.json      # Learned Professional Titles (Dynamic Dict)
│   │   └── *.html                # Plotly Visualizations (PCA, Network, Skills)
│   └── embeddings/               # ChromaDB vector store
├── src/
│   ├── scraper_selenium.py       # Deep detailed scraper with start_page support
│   ├── utils.py                  # Cleaning, Skills Extraction (Blacklist), Freshness
│   ├── ml_analyzer.py            # Supervised Classif, RF Salary, LogReg Transition, Apriori
│   ├── rag_system.py             # Incremental Indexing, Dynamic Guide Gen, Role Discovery
│   ├── resume_parser.py          # PDF/Text Resume analysis and skill extraction
│   ├── chatbot.py                # LangChain 1.x Conversational Logic
│   └── pipeline.py               # Orchestration with Autonomous Scraping triggers
├── app.py                        # Main Streamlit UI (6-Tab Layout)
├── requirements.txt              # Locked 2026 stable dependencies
└── PROJECT_REPORT.md             # Comprehensive technical documentation
```

---

## 🔧 CORE LOGIC SIGNATURES

### src/ml_analyzer.py
```python
def cluster_job_roles(df: pd.DataFrame, n_clusters: int = 20) -> Tuple[pd.DataFrame, Any, List]:
    """Supervised classification using Cosine Similarity against 20 Professional Profiles."""

class SalaryPredictionEngine:
    """Random Forest Regressor for skill-based salary estimation."""

class CareerTransitionModel:
    """Logistic Regression for transition probability and factor analysis."""

def analyze_skill_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Market Basket Analysis using Apriori for skill rules."""
```

---

## 📝 RESPONSE FORMATTING GUIDELINES

### Accuracy & Transparency:
Responses MUST state the specific sample size used for analysis to ensure the user knows exactly how grounded the insights are.

**Template:**
"Analysis based on **{role_count}** specific job postings for '{role}' in '{location}' (Last 14 days)."

---

## 🚀 DEPLOYMENT & MAINTENANCE

### Data Freshness:
The system automatically purges expired jobs (older than 14 days) during every pipeline run.

---

**END OF GLOBAL CONTEXT**
