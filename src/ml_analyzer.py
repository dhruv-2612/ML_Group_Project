import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import plotly.express as px
import plotly.io as pio
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.utils import parse_skills_robust, extract_education
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cluster_job_roles(
    df: pd.DataFrame, 
    n_clusters: int = 20, 
    max_features: int = 2000
) -> Tuple[pd.DataFrame, Any, List[List[str]]]:
    """
    Classifies job roles into 20 predefined categories using Semantic/Keyword Similarity.
    """
    if df.empty or 'job_description' not in df.columns:
        logger.error("DataFrame is empty or missing 'job_description'.")
        return df, None, []
    
    # Fill NaN descriptions
    docs = df['job_description'].fillna("")
    
    # --- Predefined Cluster Definitions ---
    CLUSTER_DEFINITIONS = {
        "Software Engineering – Backend & Systems": "backend java python c++ golang rust node.js api microservices distributed systems architecture server side spring boot",
        "Software Engineering – Frontend & Mobile": "frontend react angular vue javascript typescript android ios flutter swift mobile native web react-native redux ui development",
        "Full Stack & Web Development": "full stack fullstack web developer mern mean stack django flask php laravel wordpress website development html css",
        "Data Science & Machine Learning": "data scientist machine learning ml ai deep learning nlp pytorch tensorflow computer vision generative ai llm neural networks predictive modeling",
        "Data Engineering & Big Data": "data engineer etl pipeline hadoop spark kafka airflow snowflake databricks sql nosql data warehousing big data hive",
        "Cloud, DevOps & Platform Engineering": "devops cloud aws azure gcp kubernetes docker terraform ci/cd site reliability sre infrastructure platform engineer linux jenkins",
        "Cybersecurity & Information Security": "security cyber infosec penetration testing vulnerability soc siem firewall network security ethical hacking ciso",
        "QA, Testing & Automation": "qa quality assurance testing automation selenium cypress junit manual testing sdet test automation appium",
        "Product Management": "product manager pm roadmap product owner agile scrum product strategy user story feature prioritization backlog",
        "Project / Program Management": "project manager program manager pmp delivery manager scrum master agile coach timeline stakeholder management",
        "Business Analysis & Strategy": "business analyst ba strategy consultant requirements gathering process improvement use case brd frd strategic planning",
        "Finance & Accounting": "accountant finance tax audit ca cpa ledger financial analyst reporting budgeting reconciliation accounts payable receivable",
        "Investment Banking & Capital Markets": "investment banking equity research capital markets m&a valuation portfolio management trader asset management private equity venture capital",
        "Risk, Compliance & Governance": "risk compliance regulatory aml kyc governance internal audit legal risk fraud prevention risk management",
        "Sales & Business Development (B2B/B2C)": "sales business development bdm account manager inside sales presales lead generation client relation revenue growth b2b b2c",
        "Marketing & Growth (Digital/Brand/Performance)": "marketing digital marketing seo sem social media content strategy brand manager growth hacking campaign management google ads",
        "HR & Talent Management": "hr human resources recruitment talent acquisition payroll employee relations hrbp sourcing interviewing performance management",
        "Operations & Supply Chain": "operations supply chain logistics procurement warehouse inventory admin office manager logistics coordinator supply planning",
        "Legal & Corporate Secretarial": "legal lawyer advocate corporate secretary company secretary contract litigation compliance officer llb llm drafting agreements",
        "Design & Creative (UI/UX/Graphic/Content)": "designer graphic ui ux creative art director video editor animator content creator adobe figma photoshop illustrator"
    }
    
    cluster_names = list(CLUSTER_DEFINITIONS.keys())
    cluster_keywords = list(CLUSTER_DEFINITIONS.values())
    
    # 1. TF-IDF Vectorization
    all_text = pd.concat([docs, pd.Series(cluster_keywords)], ignore_index=True)
    
    from sklearn.feature_extraction import text
    custom_stop_words = list(text.ENGLISH_STOP_WORDS.union({
        'job', 'description', 'role', 'responsibilities', 'requirements', 'experience',
        'years', 'skills', 'key', 'preferred', 'qualification', 'candidate', 'work',
        'team', 'knowledge', 'strong', 'ability', 'good', 'looking', 'hiring', 'company',
        'industry', 'apply', 'location', 'salary', 'benefits', 'opportunity'
    }))

    logger.info(f"Vectorizing job descriptions against {len(cluster_names)} predefined clusters...")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words=custom_stop_words,
        ngram_range=(1, 2)
    )
    
    tfidf.fit(all_text)
    
    job_vectors = tfidf.transform(docs)
    cluster_vectors = tfidf.transform(cluster_keywords)
    
    # 2. Compute Similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(job_vectors, cluster_vectors)
    best_cluster_indices = similarity_matrix.argmax(axis=1)
    
    df['cluster_id'] = best_cluster_indices
    df['cluster_label'] = [cluster_names[i] for i in best_cluster_indices]
    
    # 3. Generate Top Keywords (for compatibility)
    top_keywords_list = []
    feature_names = np.array(tfidf.get_feature_names_out())
    
    for i in range(len(cluster_names)):
        cluster_docs_indices = np.where(best_cluster_indices == i)[0]
        if len(cluster_docs_indices) > 0:
            mean_vector = job_vectors[cluster_docs_indices].mean(axis=0)
            mean_vector_arr = np.asarray(mean_vector).flatten()
            top_indices = mean_vector_arr.argsort()[::-1][:10]
            top_keywords_list.append(list(feature_names[top_indices]))
        else:
            top_keywords_list.append(["no_data"])

    logger.info("Classification complete using predefined clusters.")
    return df, None, top_keywords_list

def get_cluster_characteristics(df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
    """
    Analyzes a specific cluster to return its key stats and features.
    """
    cluster_df = df[df['cluster_id'] == cluster_id]
    
    if cluster_df.empty:
        return {}
        
    # 1. Avg Salary (if available)
    avg_salary = cluster_df['salary_mid'].mean() if 'salary_mid' in cluster_df.columns else 0.0
    
    # 2. Top Companies
    top_companies = cluster_df['company'].value_counts().head(5).index.tolist()
    
    # 3. Top Locations (Handle lists)
    all_locs = []
    for locs in cluster_df['location']:
        if isinstance(locs, list):
            all_locs.extend(locs)
        elif isinstance(locs, str):
            # Try to parse string list or just add string
            if locs.startswith('[') and locs.endswith(']'):
                try:
                    import ast
                    all_locs.extend(ast.literal_eval(locs))
                except:
                    all_locs.append(locs)
            else:
                all_locs.append(locs)
                
    top_locations = pd.Series(all_locs).value_counts().head(5).index.tolist() if all_locs else []
    
    # 4. Top Skills (aggregating skills_extracted lists)
    all_skills = []
    if 'skills_extracted' in cluster_df.columns:
        # Flatten list of lists, handling potential string representations if read from CSV
        for skills in cluster_df['skills_extracted']:
            if isinstance(skills, list):
                all_skills.extend(skills)
            elif isinstance(skills, str):
                # Simple cleanup if it's a string repr like "['Python', 'SQL']"
                cleaned = skills.strip("[]").replace("'", "").split(", ")
                all_skills.extend([s for s in cleaned if s])
                
    top_skills = pd.Series(all_skills).value_counts().head(10).index.tolist()
    
    return {
        'cluster_id': int(cluster_id),
        'count': len(cluster_df),
        'avg_salary_lpa': round(avg_salary, 2),
        'top_companies': top_companies,
        'top_locations': top_locations,
        'top_skills': top_skills
    }

def analyze_skills_by_role(df: pd.DataFrame, role_filter: str = None, skill_col: str = 'skills_extracted') -> pd.DataFrame:
    """
    Analyzes skill frequency using high-quality extracted skills by default.
    """
    if df.empty:
        return pd.DataFrame()
        
    target_df = df.copy()
    if role_filter:
        target_df = target_df[target_df['job_title'].str.contains(role_filter, case=False, na=False)]
        
    if target_df.empty:
        return pd.DataFrame()

    # Priority: skills_extracted (Clean/Deduplicated) > skills_list (Raw Portal)
    actual_col = skill_col
    if actual_col not in target_df.columns or target_df[actual_col].isna().all():
        actual_col = 'skills_list' if 'skills_list' in target_df.columns else None
            
    if not actual_col:
        return pd.DataFrame()

    # Create a working copy for exploding
    temp_df = target_df[['job_id', 'salary_mid', actual_col]].copy()
    temp_df['skill_name'] = temp_df[actual_col].apply(parse_skills_robust)
    exploded = temp_df.explode('skill_name')
    
    # Filter out empty skill names
    exploded = exploded[exploded['skill_name'].notna() & (exploded['skill_name'] != "")]
    
    if exploded.empty:
        return pd.DataFrame()

    # Group by skill
    skill_stats = exploded.groupby('skill_name').agg(
        count=('job_id', 'count'),
        avg_salary=('salary_mid', 'mean')
    ).reset_index()
    
    skill_stats['percentage'] = (skill_stats['count'] / len(target_df)) * 100
    skill_stats = skill_stats.sort_values('count', ascending=False)
    
    return skill_stats

def identify_emerging_skills(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Identifies high-demand skills based on frequency in recent job postings.
    Returns: DataFrame sorted by count.
    """
    current_stats = analyze_skills_by_role(df)
    
    if current_stats.empty:
        return pd.DataFrame()
        
    current_stats['growth_rate'] = current_stats['percentage'] 
    
    return current_stats.head(top_n)

# --- Career Transition Intelligence (Reliable Logistic Regression) ---

class CareerTransitionModel:
    """
    Predicts career transition likelihood using a scaled Logistic Regression model 
    trained on a realistic synthetic dataset derived from real job distributions.
    """
    def __init__(self):
        self.model = LogisticRegression(C=0.5, class_weight='balanced')
        self.scaler = StandardScaler()
        self.feature_names = [
            'Skill Match', 
            'Experience Fit', 
            'Location Alignment', 
            'Education Compatibility',
            'Role Complexity'
        ]
        self.is_trained = False

    def _normalize_skill(self, s: str) -> str:
        """Helper for synonym-aware skill matching."""
        s = str(s).lower().strip()
        syn_map = {
            'analytics': 'analysis',
            'ai': 'artificial intelligence',
            'artificial intelligence': 'ai',
            'ml': 'machine learning',
            'machine learning': 'ml',
            'genai': 'generative ai',
            'generative ai': 'genai',
            'llm': 'genai',
            'fullstack': 'full stack',
            'frontend': 'front end',
            'backend': 'back end',
            'ui': 'user interface',
            'ux': 'user experience'
        }
        for k, v in syn_map.items():
            if k == s: return v
        return s

    def _match_education(self, user_edu: List[str], job_edu: List[str]) -> float:
        """Helper to match education levels including hierarchies."""
        if not job_edu:
            return 1.0 # No specific requirement
        
        user_set = set(e.lower() for e in user_edu)
        hierarchy = {
            'phd': ['phd', 'doctorate'],
            'master\'s': ['master\'s', 'm.tech/m.e', 'mba', 'postgraduate', 'm.sc', 'm.com', 'm.a'],
            'mba': ['mba', 'pgdm'],
            'b.tech/b.e': ['b.tech/b.e', 'bachelor\'s'],
            'bachelor\'s': ['bachelor\'s', 'undergraduate', 'b.sc', 'b.com', 'b.a']
        }
        
        for req in job_edu:
            req_lower = req.lower()
            if req_lower in user_set:
                return 1.0
            for key, aliases in hierarchy.items():
                if req_lower == key:
                    if any(alias in user_set for alias in aliases):
                        return 1.0
        return 0.0

    def train(self, jobs_df: pd.DataFrame):
        """
        Trains the scaler and model on a multi-set synthetic dataset designed to 
        achieve precise coefficient weights for realistic career advice.
        """
        if jobs_df.empty:
            logger.warning("Jobs DataFrame is empty. Skipping model training.")
            return

        logger.info("Training Precision-Tuned Career Transition Model...")
        
        # Realistic distributions for complexity
        num_skills_dist = jobs_df['num_skills'].dropna().values
        if len(num_skills_dist) < 5: num_skills_dist = [5, 8, 12, 18, 25]

        training_data, labels = [], []
        
        # SET 1: The 'Experience Veteran' (Fixes User Case)
        # Low/Mid Skills (35-50%), but Experience Surplus or Perfect Match
        for _ in range(1000):
            training_data.append([
                np.random.uniform(0.35, 0.55), # Mid skills
                np.random.uniform(-5.0, 0.0),  # Exp Surplus (Match)
                1.0, 1.0, 
                np.random.choice(num_skills_dist)
            ])
            labels.append(1) # Experience carries the transition

        # SET 2: The 'Skill Savant'
        # High Skills (70-100%), but slight Experience Gap (1-3 years)
        for _ in range(800):
            training_data.append([
                np.random.uniform(0.7, 1.0), 
                np.random.uniform(0.5, 3.0), # Small Gap
                1.0, 1.0, 
                np.random.choice(num_skills_dist)
            ])
            labels.append(1) # Skills carry the transition

        # SET 3: 'Almost There' (Balanced Middle)
        for _ in range(800):
            skill = np.random.uniform(0.4, 0.6)
            gap = np.random.uniform(0.0, 2.0)
            training_data.append([skill, gap, 1.0, 1.0, np.random.choice(num_skills_dist)])
            labels.append(1 if np.random.rand() < 0.6 else 0) # 60% success chance

        # SET 4: Hard Failures (Skills < 25% OR Gap > 6y OR Loc/Edu Mismatch)
        for _ in range(1200):
            case = np.random.choice(['skill', 'exp', 'loc', 'edu'])
            if case == 'skill':
                training_data.append([np.random.uniform(0.0, 0.25), 0.0, 1.0, 1.0, 10])
            elif case == 'exp':
                training_data.append([0.8, np.random.uniform(6.0, 12.0), 1.0, 1.0, 10])
            elif case == 'loc':
                training_data.append([0.9, 0.0, 0.0, 1.0, 10])
            else:
                training_data.append([0.9, 0.0, 1.0, 0.0, 10])
            labels.append(0)

        # SET 5: Complexity Scaling (High complexity needs higher skills)
        for _ in range(600):
            complexity = np.random.randint(20, 40)
            skill = np.random.uniform(0.4, 0.6)
            training_data.append([skill, 0.0, 1.0, 1.0, complexity])
            labels.append(0) # Fails because 40-60% skills isn't enough for VERY complex roles

        X = np.array(training_data)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        coefs = self.model.coef_[0]
        logger.info(f"Updated Learned Coefficients: {dict(zip(self.feature_names, np.round(coefs, 3)))}")
        self.is_trained = True

        X = np.array(training_data)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Eval
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
        logger.info(f"Career Transition Model Trained. ROC-AUC: {auc:.4f}")
        self.is_trained = True

    def predict(self, user_profile: Dict[str, Any], target_job: Dict[str, Any], market_skills: pd.DataFrame = None) -> Dict[str, Any]:
        if not self.is_trained:
            return {"error": "Model not trained"}
            
        # 1. Skill Match (Weighted by Market Frequency if market_skills provided)
        user_skills_raw = user_profile.get('skills', [])
        user_skills_norm = set(self._normalize_skill(s) for s in user_skills_raw)
        
        job_skills_raw = target_job.get('skills_list', target_job.get('skills_extracted', []))
        job_skills_list = parse_skills_robust(job_skills_raw)
        
        if market_skills is not None and not market_skills.empty:
            # Use Top 20 skills from market stats (Tab 3) with weights
            top_20 = market_skills.head(20).copy()
            total_freq = top_20['percentage'].sum()
            
            # Match user skills against top 20 market skills
            top_20['is_matched'] = top_20['skill_name'].apply(lambda s: self._normalize_skill(s) in user_skills_norm)
            user_weighted_sum = top_20[top_20['is_matched']]['percentage'].sum()
            
            skill_ratio = user_weighted_sum / total_freq if total_freq > 0 else 0.0
            
            # For feedback, missing skills come from the market top 20
            missing_skills = top_20[~top_20['is_matched']]['skill_name'].tolist()
        else:
            # Fallback to direct job skill match
            job_skills_norm = set(self._normalize_skill(s) for s in job_skills_list)
            skill_ratio = len(user_skills_norm.intersection(job_skills_norm)) / len(job_skills_norm) if job_skills_norm else 1.0
            missing_skills = [s for s in job_skills_list if self._normalize_skill(s) not in user_skills_norm]
        
        # 2. Exp
        exp_gap = float(target_job.get('experience_min', 0)) - float(user_profile.get('years_experience', 0))
        
        # 3. Location
        user_locs = [l.lower() for l in user_profile.get('locations', [])]
        job_locs = [l.lower() for l in target_job.get('location', [])]
        loc_match = 1.0 if any(loc in job_locs for loc in user_locs) or not job_locs else 0.0
        
        # 4. Education
        user_edu = user_profile.get('education', [])
        if isinstance(user_edu, str): user_edu = [user_edu]
        job_edu = target_job.get('education_extracted', [])
        if not job_edu and 'job_description' in target_job:
            job_edu = extract_education(target_job['job_description'])
        edu_match = self._match_education(user_edu, job_edu)
        
        # 5. Difficulty
        difficulty = float(len(job_skills_list))
        
        features = np.array([[skill_ratio, exp_gap, loc_match, edu_match, difficulty]])
        features_scaled = self.scaler.transform(features)
        prob = self.model.predict_proba(features_scaled)[0][1]
        
        missing_skills = []
        for s in job_skills_list:
            if self._normalize_skill(s) not in user_skills_norm:
                missing_skills.append(s)
        
        return {
            'probability': round(max(0.0, min(100.0, prob * 100)), 1),
            'confidence': 'High' if prob > 0.75 else 'Moderate' if prob > 0.4 else 'Low',
            'factors': {
                'Skill Match': f"{skill_ratio*100:.1f}%",
                'Experience Fit': f"{exp_gap:.1f} years",
                'Location Alignment': "Yes" if loc_match > 0.5 else "No",
                'Education Compatibility': "High" if edu_match > 0.5 else "Mismatch",
                'Role Complexity': "High" if difficulty > 12 else "Medium" if difficulty > 6 else "Normal"
            },
            'missing_skills': missing_skills[:8],
            'recommendation': self._get_recommendation(prob, missing_skills)
        }

    def _get_recommendation(self, prob, missing_skills):
        if prob > 0.8: return "🚀 High success potential! You are a strong candidate."
        if prob > 0.5: return f"💡 Moderate potential. Consider learning: {', '.join(missing_skills[:3])}."
        return "⚠️ Difficult transition. Focus on skill building or lower experience roles."

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained: return {}
        coefs = self.model.coef_[0]
        return {name: float(val) for name, val in zip(self.feature_names, coefs)}

def calculate_skill_salary_impact(df: pd.DataFrame, skill_list: List[str] = None) -> pd.DataFrame:
    """
    Calculates the salary premium associated with each skill.
    Filtered to show only positive 'boosts' for better UI clarity.
    If skill_list is provided, only analyzes those specific skills (e.g., Top 20).
    """
    if df.empty or 'salary_mid' not in df.columns:
        return pd.DataFrame()
        
    # Find best skill column
    skill_col = next((c for col in ['skills_list', 'skills_extracted', 'skills_text'] if (c := col) in df.columns and not df[col].isna().all()), None)
    if not skill_col: return pd.DataFrame()
        
    if skill_list:
        unique_skills = [s for s in skill_list if s]
    else:
        all_skills_series = df[skill_col].apply(parse_skills_robust)
        unique_skills = list(set([s for sublist in all_skills_series for s in sublist]))
    
    impact_data = []
    
    for skill in unique_skills:
        # Check presence
        has_skill = df[skill_col].apply(lambda x: skill in str(x))
        with_skill = df[has_skill]['salary_mid'].dropna()
        without_skill = df[~has_skill]['salary_mid'].dropna()
        
        if len(with_skill) < 3 or len(without_skill) < 3:
            continue
            
        avg_with = with_skill.mean()
        avg_without = without_skill.mean()
        premium = avg_with - avg_without
        
        if premium <= 0:
            continue
            
        t_stat, p_val = stats.ttest_ind(with_skill, without_skill, equal_var=False)
        
        impact_data.append({
            'skill': skill,
            'avg_salary_with': round(avg_with, 2),
            'avg_salary_without': round(avg_without, 2),
            'premium': round(premium, 2),
            'p_value': round(p_val, 4),
            'count': len(with_skill)
        })
        
    if not impact_data:
        return pd.DataFrame(columns=['skill', 'avg_salary_with', 'avg_salary_without', 'premium', 'p_value', 'count'])
        
    return pd.DataFrame(impact_data).sort_values('premium', ascending=False)

# --- Salary Prediction Engine (Session 11: Random Forest) ---

class SalaryPredictionEngine:
    """
    Random Forest Regressor to predict salary based on structured features.
    """
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_cols = []

    def train(self, df: pd.DataFrame):
        """
        Trains the RF model on available job data.
        """
        if df.empty or 'salary_mid' not in df.columns:
            return
            
        logger.info("Training Salary Prediction Engine (Random Forest)...")
        
        # Prepare Features
        train_df = df.dropna(subset=['salary_mid']).copy()
        if len(train_df) < 20:
             logger.warning("Insufficient data to train salary predictor.")
             return

        # Simple encoding for experience
        train_df['exp_num'] = train_df['experience_min'].fillna(train_df['experience_min'].median())
        
        # One-hot encoding for Cluster
        cluster_dummies = pd.get_dummies(train_df['cluster_label'], prefix='c')
        
        # Skill count as a feature - Find best skill column
        skill_col = next((c for col in ['skills_list', 'skills_extracted', 'skills_text'] if (c := col) in train_df.columns and not train_df[col].isna().all()), None)
        if skill_col:
            train_df['num_skills'] = train_df[skill_col].apply(lambda x: len(parse_skills_robust(x)))
        else:
            train_df['num_skills'] = 0
        
        X = pd.concat([train_df[['exp_num', 'num_skills']], cluster_dummies], axis=1)
        y = train_df['salary_mid']
        
        self.feature_cols = X.columns.tolist()
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"Salary Prediction Engine trained on {len(train_df)} records.")

    def predict(self, profile: Dict[str, Any], cluster_label: str) -> float:
        """
        Predicts salary for a user profile in a specific cluster.
        """
        if not self.is_trained:
            return 0.0
            
        # Create input vector
        input_data = {col: 0 for col in self.feature_cols}
        input_data['exp_num'] = float(profile.get('years_experience', 2))
        input_data['num_skills'] = len(profile.get('skills', []))
        
        cluster_col = f'c_{cluster_label}'
        if cluster_col in input_data:
            input_data[cluster_col] = 1
            
        X_input = pd.DataFrame([input_data])[self.feature_cols]
        prediction = self.model.predict(X_input)[0]
        
        return round(float(prediction), 2)

def generate_skills_report(df: pd.DataFrame, role: str = None) -> Dict[str, Any]:
    """
    Generates a comprehensive skills report for a specific role.
    """
    if role is None:
        role = "the selected role"
    report = {}
    
    # 1. Skill Analysis
    skill_stats = analyze_skills_by_role(df, role_filter=role)
    report['top_skills'] = skill_stats.head(10).to_dict('records')
    
    # 2. Salary Impact (global for now, to ensure sample size)
    impact_df = calculate_skill_salary_impact(df)
    report['salary_premiums'] = impact_df.head(5).to_dict('records')
    
    # 3. Visualization
    if not skill_stats.empty:
        fig = px.bar(
            skill_stats.head(10),
            x='count',
            y='skill_name',
            orientation='h',
            title=f"Top Skills for {role}",
            labels={'skill_name': 'Skill', 'count': 'Job Count'},
            template='plotly_white'
        )
        output_path = f"data/processed/skills_viz_{role.lower().replace(' ', '_')}.html"
        fig.write_html(output_path)
        report['viz_path'] = output_path
        
    return report

def analyze_salary_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates statistical salary benchmarks (Median, IQR) instead of predictive modeling.
    Returns: Dictionary of salary stats by Cluster and Experience Level.
    """
    if df.empty or 'salary_mid' not in df.columns:
        logger.warning("DataFrame empty or missing 'salary_mid'.")
        return {}
        
    # Drop rows with missing salary
    valid_sal = df.dropna(subset=['salary_mid']).copy()
    
    if valid_sal.empty:
        return {}

    stats_report = {
        'global_median': round(valid_sal['salary_mid'].median(), 2),
        'global_mean': round(valid_sal['salary_mid'].mean(), 2),
        'by_cluster': {},
        'by_experience': {}
    }
    
    # 1. By Cluster
    if 'cluster_label' in valid_sal.columns:
        cluster_stats = valid_sal.groupby('cluster_label')['salary_mid'].describe()
        stats_report['by_cluster'] = cluster_stats[['50%', '25%', '75%']].round(2).to_dict('index')
        
    # 2. By Experience Level
    if 'experience_level' in valid_sal.columns:
        exp_stats = valid_sal.groupby('experience_level')['salary_mid'].describe()
        stats_report['by_experience'] = exp_stats[['50%', '25%', '75%']].round(2).to_dict('index')
        
    logger.info("Salary statistical analysis complete.")
    return stats_report

def get_salary_benchmark(df: pd.DataFrame, role_filter: str = None, location_filter: str = None) -> Dict[str, float]:
    """
    Retrieves real-time salary stats for a specific slice of data.
    """
    target_df = df.copy()
    
    if role_filter:
        target_df = target_df[target_df['job_title'].str.contains(role_filter, case=False, na=False)]
    
    if location_filter and location_filter != "All Locations":
        def loc_match(loc_entry):
            if isinstance(loc_entry, list): return location_filter in loc_entry
            if isinstance(loc_entry, str): return location_filter in loc_entry
            return False
        target_df = target_df[target_df['location'].apply(loc_match)]
        
    if target_df.empty or 'salary_mid' not in target_df.columns:
        return {'median': 0, 'min': 0, 'max': 0, 'count': 0}
        
    salaries = target_df['salary_mid'].dropna()
    if salaries.empty:
        return {'median': 0, 'min': 0, 'max': 0, 'count': 0}
        
    return {
        'median': round(salaries.median(), 2),
        'mean': round(salaries.mean(), 2),
        'min': round(salaries.min(), 2),
        'max': round(salaries.max(), 2),
        'count': len(salaries)
    }

def plot_clusters_visualization(df: pd.DataFrame, title: str = "Job Role Clusters"):
    """
    Visualizes clusters using PCA (2D) and saves to HTML.
    """
    if 'cluster_id' not in df.columns:
        logger.error("Clusters not found. Run cluster_job_roles first.")
        return

    logger.info("Generating PCA visualization...")
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    matrix = tfidf.fit_transform(df['job_description'].fillna(""))
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(matrix.toarray())
    df['pca_x'] = components[:, 0]
    df['pca_y'] = components[:, 1]
    fig = px.scatter(
        df, x='pca_x', y='pca_y', color='cluster_label',
        hover_data=['job_title', 'company', 'location'],
        title=title, template='plotly_white', width=1000, height=700
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    output_path = "data/processed/clusters_viz.html"
    fig.write_html(output_path)

def analyze_skill_associations(df: pd.DataFrame, min_support: float = 0.05, allowed_skills: List[str] = None) -> pd.DataFrame:
    """
    Finds which skills appear together using Apriori.
    If allowed_skills is provided, restricts analysis to those specific skills.
    """
    from mlxtend.frequent_patterns import apriori, association_rules
    if df.empty: return pd.DataFrame()
    
    skill_col = next((c for col in ['skills_list', 'skills_extracted', 'skills_text'] if (c := col) in df.columns and not df[col].isna().all()), None)
    if not skill_col: return pd.DataFrame()
    
    skills_series = df[skill_col].apply(parse_skills_robust)
    
    # Filter skills if allowed_skills provided
    if allowed_skills:
        allowed_set = set(allowed_skills)
        skills_series = skills_series.apply(lambda x: [s for s in x if s in allowed_set])
        
    skills_series = skills_series[skills_series.map(len) > 0]
    if skills_series.empty: return pd.DataFrame()
    
    # One-hot encoding
    skills_matrix = pd.get_dummies(skills_series.apply(pd.Series).stack()).groupby(level=0).max().astype(bool)
    if skills_matrix.empty: return pd.DataFrame()
    
    try:
        # Use apriori as requested, but optimized by the restricted skills matrix
        frequent_itemsets = apriori(skills_matrix, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty: return pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules = rules.sort_values('lift', ascending=False)
        rules['antecedents_list'] = rules['antecedents'].apply(lambda x: list(x))
        rules['consequents_list'] = rules['consequents'].apply(lambda x: list(x))
        return rules
    except Exception as e:
        logger.error(f"Error in association analysis: {e}")
        return pd.DataFrame()

def calculate_cooccurrence_matrix(df: pd.DataFrame, skills_list: List[str]) -> pd.DataFrame:
    """
    Calculates a symmetric matrix where each cell (i, j) is the count of job 
    postings containing both Skill i and Skill j.
    """
    if df.empty or not skills_list:
        return pd.DataFrame()
        
    skill_col = next((c for col in ['skills_list', 'skills_extracted', 'skills_text'] if (c := col) in df.columns and not df[col].isna().all()), None)
    if not skill_col: return pd.DataFrame()
    
    # Parse skills for each job
    parsed_skills = df[skill_col].apply(parse_skills_robust)
    
    # Initialize matrix
    matrix = pd.DataFrame(0, index=skills_list, columns=skills_list)
    
    # Count occurrences
    for skills in parsed_skills:
        # Filter to only the skills we care about
        matched = [s for s in skills if s in skills_list]
        for i in range(len(matched)):
            for j in range(i + 1, len(matched)):
                matrix.loc[matched[i], matched[j]] += 1
                matrix.loc[matched[j], matched[i]] += 1
                
    return matrix

def create_skill_network_graph(rules: pd.DataFrame, all_nodes: List[str] = None):
    """(DEPRECATED) Kept for backward compatibility, uses association rules."""
    import networkx as nx
    G = nx.Graph()
    if all_nodes:
        for node in all_nodes: G.add_node(node)
    if rules.empty: return G, {}, []
    top_rules = rules.head(150)
    for _, rule in top_rules.iterrows():
        for a in rule['antecedents_list']:
            for b in rule['consequents_list']:
                if all_nodes and (a not in all_nodes or b not in all_nodes): continue
                if G.has_edge(a, b): G[a][b]['weight'] = max(G[a][b]['weight'], rule['lift'])
                else: G.add_edge(a, b, weight=rule['lift'])
    centrality = nx.degree_centrality(G)
    communities = list(nx.community.greedy_modularity_communities(G))
    community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    return G, centrality, community_map

def create_cooccurrence_network(df: pd.DataFrame, top_skills: List[str]):
    """
    Creates a network graph based on raw pairwise frequency of the top skills.
    Edge weight = frequency of co-occurrence.
    Node size = total frequency of the skill.
    """
    import networkx as nx
    G = nx.Graph()
    
    if not top_skills:
        return G, {}, {}
        
    # Add all nodes
    for skill in top_skills:
        G.add_node(skill)
        
    # Calculate pairwise frequencies
    matrix = calculate_cooccurrence_matrix(df, top_skills)
    
    if matrix.empty:
        return G, {}, {}
        
    # Add edges where frequency > 0
    for i in range(len(top_skills)):
        for j in range(i + 1, len(top_skills)):
            freq = matrix.iloc[i, j]
            if freq > 0:
                G.add_edge(top_skills[i], top_skills[j], weight=float(freq))
                
    # Calculate metrics
    # Node importance based on total connections (degree)
    centrality = nx.degree_centrality(G)
    
    # Community detection
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    except:
        community_map = {node: 0 for node in G.nodes()}
        
    return G, centrality, community_map

def plot_skill_network(G, centrality, community_map):
    import networkx as nx
    import plotly.graph_objects as go
    if not G.nodes(): return None
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{node}</b><br>Connections: {G.degree(node)}")
        node_color.append(community_map.get(node, 0))
        node_size.append(15 + centrality.get(node, 0) * 100)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=[n for n in G.nodes()],
        textposition="top center", marker=dict(
            showscale=True, colorscale='Viridis', reversescale=True, color=node_color, size=node_size,
            colorbar=dict(thickness=15, title=dict(text='Skill Cluster', side='right'), xanchor='left'), line_width=2))
    node_trace.text = node_text
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=dict(text='Skill Co-occurrence Network', font=dict(size=16)), showlegend=False, hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40), annotations=[ dict(text="Node size = Centrality | Colors = Skill Clusters", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

def analyze_experience_alignment(df: pd.DataFrame, user_experience: float) -> Dict[str, Any]:
    """
    Analyzes how the user's experience aligns with the market volume for a specific role.
    """
    if df.empty or 'experience_min' not in df.columns:
        return {}
        
    exp_min_series = df['experience_min'].dropna()
    if exp_min_series.empty:
        return {}
        
    # Calculate volume distribution
    counts = exp_min_series.value_counts().sort_index()
    
    # Calculate percentiles of experience_min in the market
    p25 = exp_min_series.quantile(0.25)
    p50 = exp_min_series.median()
    p75 = exp_min_series.quantile(0.75)
    
    # Alignment logic based on where most jobs are (Volume)
    if user_experience < p25:
        status = "Under-qualified"
        color = "#D32F2F" # Red
        message = f"You are below the core market volume. 75% of available roles require at least {p25} years of experience."
    elif user_experience > p75 + 2: # Give some buffer for overqualification
        status = "Over-qualified"
        color = "#FFA000" # Orange
        message = f"You exceed the core market volume for this specific search. Most roles ({p75}%) require less experience than you have."
    else:
        status = "Perfectly Matched"
        color = "#2E7D32" # Green
        message = "Your experience perfectly aligns with the current peak market volume for this role."

    # Percent of market accessible
    # A user can realistically apply for jobs requiring up to their exp + 1 or 2 years
    accessible_mask = exp_min_series <= (user_experience + 1)
    accessible_pct = accessible_mask.mean() * 100
    
    return {
        'status': status,
        'color': color,
        'message': message,
        'user_experience': user_experience,
        'market_p25': p25,
        'market_p50': p50,
        'market_p75': p75,
        'accessible_volume_pct': round(accessible_pct, 1),
        'exp_distribution': counts.to_dict()
    }

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/cleaned_jobs.csv")
        clustered_df, kmeans_model, keywords = cluster_job_roles(df, n_clusters=20)
        plot_clusters_visualization(clustered_df)
        clustered_df.to_csv("data/processed/job_clusters.csv", index=False)
    except Exception as e:
        logger.error(f"Test failed: {e}")
