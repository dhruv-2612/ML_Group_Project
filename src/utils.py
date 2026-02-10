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

# Flatten skill list for regex matching
ALL_SKILLS = [skill for sublist in SKILL_CATEGORIES.values() for skill in sublist]

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
    Returns: Unique list of matched skills.
    """
    if not text:
        return []
    
    found_skills = set()
    for skill in ALL_SKILLS:
        # Use word boundaries to avoid matching 'R' in 'Product'
        pattern = rf'\b{re.escape(skill)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(skill.replace('', '')) # Clean up escaped chars
            
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
    
    # 1. Deduplication
    initial_len = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=['job_title', 'company'], keep='first')
    logger.info(f"Removed {initial_len - len(clean_df)} duplicates.")
    
    # 2. Basic Cleaning
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
    
    def standardize_loc(loc):
        if not isinstance(loc, str): return "Other"
        loc = loc.lower()
        for key, val in location_map.items():
            if key in loc:
                return val
        return loc.capitalize()

    clean_df['location'] = clean_df['location'].apply(standardize_loc)
    
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
    else:
        clean_df['skills_extracted'] = clean_df['job_title'].apply(extract_skills)
        
    clean_df['skills_text'] = clean_df['skills_extracted'].apply(lambda x: ", ".join(x))
    clean_df['num_skills'] = clean_df['skills_extracted'].apply(len)
    
    # 7. Final Polish
    clean_df['scraped_date'] = pd.to_datetime(clean_df['scraped_date'])
    
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
