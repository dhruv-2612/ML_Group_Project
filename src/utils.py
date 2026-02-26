import pandas as pd
import numpy as np
import re
import logging
import time
from typing import List, Dict, Tuple, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Comprehensive Skill Dictionary for Extraction
SKILL_CATEGORIES = {
    'programming': ['Python', 'R', 'SQL', 'JavaScript', 'Java', r'C\+\+', 'Scala'],
    'data_tools': ['Tableau', 'Power BI', 'Excel', 'Looker', 'Metabase', 'SAS', 'SPSS', 'Qlik'],
    'product_tools': ['Figma', 'JIRA', 'Confluence', 'Miro', 'Notion', 'Asana', 'Trello'],
    'analytics': ['A/B Testing', 'Google Analytics', 'Mixpanel', 'Amplitude', 'Segment', 'Hotjar'],
    'ml_ai': ['Machine Learning', 'NLP', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Prompt Engineering', 'GenAI'],
    'cloud': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Snowflake', 'Databricks'],
    'soft_skills': ['Communication', 'Leadership', 'Problem Solving', 'Stakeholder Management', 'Critical Thinking', 'Collaboration'],
    'domain': ['Financial Modeling', 'Market Research', 'Strategy', 'Operations', 'Supply Chain', 'User Research', 'Product Roadmap']
}

def load_all_skills() -> List[str]:
    """Loads and cleans skills from skills.csv, master_skills.csv and SKILL_CATEGORIES."""
    skills = set()
    # 1. Add base categories
    for sublist in SKILL_CATEGORIES.values():
        skills.update(sublist)
    
    # 2. Add from skills.csv (Root directory)
    root_skills_path = "skills.csv"
    if os.path.exists(root_skills_path):
        try:
            # Handle potential header or single column format
            df_skills = pd.read_csv(root_skills_path)
            # Find the correct column name (could be 'Skills' or first column)
            col_name = 'Skills' if 'Skills' in df_skills.columns else df_skills.columns[0]
            raw_names = df_skills[col_name].dropna().tolist()
            for name in raw_names:
                name = str(name).strip()
                if name:
                    skills.add(name)
        except Exception as e:
            logger.warning(f"Could not load skills.csv from root: {e}")

    # 3. Add from master_skills.csv
    csv_path = "data/processed/master_skills.csv"
    if os.path.exists(csv_path):
        try:
            df_skills = pd.read_csv(csv_path)
            raw_names = df_skills['SkillName'].tolist()
            for name in raw_names:
                # Add original
                skills.add(name)
                # Add cleaned (remove common redundant words)
                cleaned = re.sub(r'\s+(Programming|Skills|Knowledge|Principles|Software|Tools|Fundamentals|Methodology|Methodologies|Frameworks)$', '', name, flags=re.IGNORECASE)
                if cleaned != name:
                    skills.add(cleaned)
        except Exception as e:
            logger.warning(f"Could not load master_skills.csv: {e}")
            
    return sorted(list(skills), key=len, reverse=True)

import os
ALL_SKILLS = load_all_skills()

def clean_text(text: Any) -> str:
    """Basic text cleaning for descriptions and titles."""
    if not isinstance(text, str) or text == "":
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters but keep some punctuation
    text = re.sub(r'[^\w\s\.,\-\&\/]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_experience(exp_str: Any) -> Tuple[float, float]:
    """
    Extracts min and max years of experience from strings like '2-5 years' or '5+ years'.
    Returns: (min_exp, max_exp)
    """
    if not isinstance(exp_str, str) or exp_str == "":
        return 0.0, 0.0
    
    # Find all numbers in the string
    nums = re.findall(r'\d+\.?\d*', exp_str)
    
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        return float(nums[0]), float(nums[0]) + 2.0  # Assume +2 if only one number (e.g., '5+ years')
    
    return 0.0, 0.0

def parse_salary(salary_str: Any) -> Tuple[float, float]:
    """
    Parses salary strings into LPA (Lakhs Per Annum).
    Handles: '10-20 Lacs PA', '500,000 - 800,000', '15 LPA'
    Returns: (min_salary, max_salary) in LPA
    """
    if not isinstance(salary_str, str) or "Not Disclosed" in salary_str or salary_str == "":
        return np.nan, np.nan
    
    salary_str = salary_str.lower().replace(",", "")
    nums = re.findall(r'\d+\.?\d*', salary_str)
    
    if not nums:
        return np.nan, np.nan
    
    vals = [float(n) for n in nums]
    
    # Heuristic to detect if value is in absolute INR (e.g. 500000) vs LPA (e.g. 5)
    processed_vals = []
    for v in vals:
        if v > 1000: # Likely absolute INR
            processed_vals.append(v / 100000.0)
        else: # Likely already in LPA
            processed_vals.append(v)
            
    if len(processed_vals) >= 2:
        return processed_vals[0], processed_vals[1]
    return processed_vals[0], processed_vals[0]

def extract_skills(text: str) -> List[str]:
    """
    Extracts skills from text using regex matching against the skill dictionary.
    Includes filtering for short/ambiguous terms and a blacklist.
    """
    if not text:
        return []
    
    # Blacklist of terms that cause false positives (common words or ambiguous abbreviations)
    BLACKLIST = {'CS', 'IT', 'AS', 'IN', 'R', 'OR', 'FOR', 'AND', 'THE', 'TO', 'WITH', 'BY'}
    
    found_skills = set()
    text_lower = text.lower()
    
    for skill in ALL_SKILLS:
        skill_upper = skill.upper()
        if skill_upper in BLACKLIST:
            continue
            
        # For very short skills (<= 3 chars), be more strict (e.g., case sensitive or specific context)
        if len(skill) <= 3:
            # Only match if it's a known high-confidence tech skill like 'SQL', 'PHP', 'AWS', 'C++', 'Git'
            TECH_SHORT = {'SQL', 'PHP', 'AWS', 'C++', 'GIT', 'R', 'SAP', 'ERP', 'ML', 'AI', 'NLP', 'UX', 'UI'}
            if skill_upper not in TECH_SHORT:
                continue
            
            # Use word boundaries and check for exact case-insensitive match
            pattern = rf'\b{re.escape(skill)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_skills.add(skill)
        else:
            pattern = rf'\b{re.escape(skill)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_skills.add(skill)
            
    return sorted(list(found_skills))

def categorize_experience_level(years: float) -> str:
    """Maps years of experience to a category label."""
    if years < 2:
        return "Entry Level"
    elif 2 <= years < 5:
        return "Mid Level"
    elif 5 <= years < 10:
        return "Senior Level"
    else:
        return "Lead/Principal"

def parse_skills_robust(x):
    """Unified helper to parse skills with Case Normalization and Cleaning."""
    if isinstance(x, list): 
        # Standardize Case (Title Case for display consistency)
        return sorted(list(set([str(s).strip().title() for s in x if s])))
    if isinstance(x, str):
        if x.startswith('[') and x.endswith(']'):
            try:
                import ast
                val = ast.literal_eval(x)
                return sorted(list(set([str(s).strip().title() for s in val if s])))
            except: pass
        # Handle comma separated strings
        return sorted(list(set([s.strip(" '\"").title() for s in x.split(",") if s.strip()])))
    return []

def extract_education(text: str) -> List[str]:
    """
    Extracts degrees/education from text using regex.
    """
    if not text:
        return []
    
    edu_patterns = {
        'MBA': [r'\bMBA\b', r'\bPGDM\b', r'\bMaster of Business Administration\b'],
        'B.Tech/B.E': [r'\bB\.?Tech\b', r'\bB\.?E\.?\b', r'\bBachelor of Engineering\b', r'\bBachelor of Technology\b'],
        'M.Tech/M.E': [r'\bM\.?Tech\b', r'\bM\.?E\.?\b', r'\bMaster of Engineering\b', r'\bMaster of Technology\b'],
        'Bachelor\'s': [r'\bBachelors?\b', r'\bDegree\b', r'\bUndergraduate\b', r'\bB\.?Sc\b', r'\bB\.?Com\b', r'\bB\.?A\b'],
        'Master\'s': [r'\bMasters?\b', r'\bPostgraduate\b', r'\bM\.?Sc\b', r'\bM\.?Com\b', r'\bM\.?A\b'],
        'PhD': [r'\bPhD\b', r'\bPh\.?D\b', r'\bDoctorate\b']
    }
    
    found_edu = set()
    for edu, patterns in edu_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_edu.add(edu)
                break
    return sorted(list(found_edu))

def clean_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning pipeline: dedup, standardize, parse fields.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return df
    
    clean_df = df.copy()
    
    # 0. Ensure job_id exists
    if 'job_id' not in clean_df.columns:
        clean_df['job_id'] = [f"job_{i:04d}_{int(time.time())}" for i in range(len(clean_df))]
    
    # 1. Basic Cleaning (Deduplication moved to step 3)
    clean_df['job_title'] = clean_df['job_title'].apply(clean_text)
    clean_df['company'] = clean_df['company'].apply(clean_text)
    if 'job_description' in clean_df.columns:
        clean_df['job_description'] = clean_df['job_description'].apply(clean_text)
    
    # 3. Location Standardization
    location_map = {
        'bengaluru': 'Bangalore',
        'bangalore': 'Bangalore',
        'gurgaon': 'Gurgaon',
        'gurugram': 'Gurgaon',
        'mumbai': 'Mumbai',
        'bombay': 'Mumbai',
        'new delhi': 'Delhi',
        'delhi ncr': 'Delhi'
    }
    
    def standardize_loc(loc_entry):
        # Normalize to list
        if isinstance(loc_entry, str):
            # Check if it's a string representation of a list
            if loc_entry.startswith('[') and loc_entry.endswith(']'):
                try:
                    import ast
                    locs = ast.literal_eval(loc_entry)
                except:
                    locs = [loc_entry]
            else:
                locs = [loc_entry]
        elif isinstance(loc_entry, list):
            locs = loc_entry
        else:
            locs = ["Other"]
            
        standardized = set()
        for loc in locs:
            if not isinstance(loc, str): continue
            loc_lower = loc.lower()
            found = False
            for key, val in location_map.items():
                if key in loc_lower:
                    standardized.add(val)
                    found = True
                    break
            if not found:
                standardized.add(loc.capitalize())
        
        return list(standardized)

    clean_df['location'] = clean_df['location'].apply(standardize_loc)
    
    # Merge duplicates (combine location lists)
    # Group by URL (most reliable) or Title+Company
    if 'job_url' in clean_df.columns:
        # Fill NA urls to allow grouping
        clean_df['job_url'] = clean_df['job_url'].fillna('')
        
        # Aggregation logic
        def merge_locs(series):
            merged = set()
            for x in series:
                merged.update(x)
            return list(merged)
            
        # We want to keep the FIRST of other columns, but MERGE locations
        agg_dict = {col: 'first' for col in clean_df.columns if col != 'location' and col != 'job_url'}
        agg_dict['location'] = merge_locs
        
        # Reset index after groupby
        clean_df = clean_df.groupby('job_url', as_index=False).agg(agg_dict)
    
    # 4. Experience Parsing
    exp_parsed = clean_df['experience_required'].apply(parse_experience)
    clean_df['experience_min'] = [x[0] for x in exp_parsed]
    clean_df['experience_max'] = [x[1] for x in exp_parsed]
    clean_df['experience_level'] = clean_df['experience_min'].apply(categorize_experience_level)
    
    # 5. Salary Parsing
    salary_parsed = clean_df['salary'].apply(parse_salary)
    clean_df['salary_min'] = [s[0] for s in salary_parsed]
    clean_df['salary_max'] = [s[1] for s in salary_parsed]
    # Calculate mid-point for ML models
    clean_df['salary_mid'] = (clean_df['salary_min'] + clean_df['salary_max']) / 2
    
    # 6. Skill Extraction
    # Combine title and description for better coverage
    if 'job_description' in clean_df.columns:
        clean_df['skills_extracted'] = (clean_df['job_title'] + " " + clean_df['job_description'].fillna("")).apply(extract_skills)
        clean_df['education_extracted'] = clean_df['job_description'].fillna("").apply(extract_education)
    else:
        clean_df['skills_extracted'] = clean_df['job_title'].apply(extract_skills)
        clean_df['education_extracted'] = clean_df['job_title'].apply(extract_education)
        
    clean_df['skills_text'] = clean_df['skills_extracted'].apply(lambda x: ", ".join(x))
    clean_df['num_skills'] = clean_df['skills_extracted'].apply(len)
    
    # 7. Final Polish
    clean_df['scraped_date'] = pd.to_datetime(clean_df['scraped_date'])
    
    # --- DATA FRESHNESS FILTER ---
    # Remove jobs older than 14 days from the current date
    # Note: Using scraped_date as a proxy for freshness
    now = pd.Timestamp.now()
    fourteen_days_ago = now - pd.Timedelta(days=14)
    
    initial_count = len(clean_df)
    clean_df = clean_df[clean_df['scraped_date'] >= fourteen_days_ago]
    removed_count = initial_count - len(clean_df)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale job postings (older than 14 days).")
    
    return clean_df

if __name__ == "__main__":
    # Test with sample data
    test_data = {
        'job_title': ['Product Manager', 'Data Analyst II', 'Data Analyst II'],
        'company': ['Flipkart', 'Amazon', 'Amazon'],
        'location': ['Bengaluru', 'Gurgaon', 'Gurgaon'],
        'experience_required': ['2-5 years', '5+ years', '5+ years'],
        'salary': ['15-25 Lacs PA', 'Not Disclosed', 'Not Disclosed'],
        'job_description': ['Looking for Python and SQL skills.', 'Need Tableau and Stakeholder Management.', 'Need Tableau and Stakeholder Management.'],
        'scraped_date': ['2026-02-11T12:00:00', '2026-02-11T12:05:00', '2026-02-11T12:05:00']
    }
    df = pd.DataFrame(test_data)
    cleaned_df = clean_job_data(df)
    print(cleaned_df[['job_title', 'location', 'experience_min', 'experience_level', 'salary_min', 'skills_extracted']])
