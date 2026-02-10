import pandas as pd
from src.utils import clean_job_data
import os
import glob

def verify_live_data():
    # Find the latest deep scraped file
    files = glob.glob("data/raw/naukri_deep_scraped_*.csv")
    if not files:
        # Fallback to general selenium files if deep scrape not found
        files = glob.glob("data/raw/naukri_selenium_*.csv")
    
    if not files:
        print("No live data found!")
        return
        
    latest_file = max(files, key=os.path.getctime)
    print(f"Cleaning live data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    print(f"Original records: {len(df)}")
    
    # Run cleaning
    cleaned_df = clean_job_data(df)
    print(f"Cleaned records: {len(cleaned_df)}")
    
    print("\nLive Data Sample (Cleaned):")
    cols = ['job_title', 'company', 'location', 'experience_min', 'skills_extracted']
    if all(c in cleaned_df.columns for c in cols):
        print(cleaned_df[cols].head())
    else:
        print(cleaned_df.head())
    
    # Save to processed
    cleaned_df.to_csv("data/processed/live_cleaned_jobs.csv", index=False)
    print("\nSaved to data/processed/live_cleaned_jobs.csv")

if __name__ == "__main__":
    verify_live_data()