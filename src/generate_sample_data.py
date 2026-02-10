import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_synthetic_job_data(n_jobs: int = 1000):
    """
    Generates a realistic synthetic dataset for job postings to ensure 
    the pipeline can be tested even if scrapers are blocked.
    """
    roles = [
        "Product Manager", "Data Analyst", "Business Analyst", 
        "Management Consultant", "Growth PM", "Technical PM",
        "Business Intelligence Analyst", "Strategy Manager"
    ]
    
    companies = [
        "Flipkart", "Amazon", "Zomato", "Swiggy", "Google", "Microsoft",
        "McKinsey & Co", "BCG", "Deloitte", "Paytm", "PhonePe", "Razorpay",
        "Uber", "Ola", "Reliance Industries", "Tata Consultancy Services"
    ]
    
    locations = ["Bangalore", "Mumbai", "Gurgaon", "Pune", "Hyderabad", "Delhi", "Bengaluru", "Gurugram"]
    
    skills_pool = [
        "Python", "SQL", "Tableau", "Power BI", "A/B Testing", "Product Roadmap",
        "Stakeholder Management", "JIRA", "Figma", "Excel", "Market Research",
        "Financial Modeling", "Communication", "Leadership", "Machine Learning",
        "NLP", "Google Analytics", "Mixpanel", "Amplitude", "User Research"
    ]
    
    data = []
    
    start_date = datetime(2026, 1, 1)
    
    for i in range(n_jobs):
        role = random.choice(roles)
        company = random.choice(companies)
        loc = random.choice(locations)
        
        # Experience
        min_exp = random.randint(0, 8)
        max_exp = min_exp + random.randint(2, 5)
        
        # Salary (LPA)
        if "Manager" in role or "Consultant" in role:
            s_min = random.randint(15, 25)
            s_max = s_min + random.randint(5, 15)
        else:
            s_min = random.randint(6, 12)
            s_max = s_min + random.randint(4, 10)
            
        # Randomly make some salaries "Not Disclosed"
        salary_text = f"{s_min}-{s_max} Lacs PA" if random.random() > 0.2 else "Not Disclosed"
        
        # Randomly choose 3-7 skills
        job_skills = random.sample(skills_pool, random.randint(3, 7))
        
        # Description
        description = (
            f"Looking for a {role} at {company} in {loc}. "
            f"The ideal candidate should have experience in {', '.join(job_skills[:2])}. "
            f"Key responsibilities include {random.choice(['driving growth', 'data-driven decision making', 'strategic planning', 'product lifecycle management'])}."
        )
        
        posted_days_ago = random.randint(0, 30)
        posted_date = start_date + timedelta(days=posted_days_ago)
        
        data.append({
            'job_id': f"job_{i:04d}",
            'job_title': role,
            'company': company,
            'location': loc,
            'experience_required': f"{min_exp}-{max_exp} years",
            'salary': salary_text,
            'job_description': description,
            'posted_date': posted_date.strftime("%Y-%m-%d"),
            'job_url': f"https://example.com/jobs/{i}",
            'source': 'Synthetic',
            'scraped_date': datetime.now().isoformat()
        })
        
    df = pd.DataFrame(data)
    
    output_path = "data/raw/synthetic_jobs.csv"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic job records at {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_job_data(1200)