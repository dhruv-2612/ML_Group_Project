import pandas as pd
from src.utils import clean_job_data
import os

def test_pipeline():
    # Load raw synthetic data
    raw_path = "data/raw/synthetic_jobs.csv"
    if not os.path.exists(raw_path):
        print("Error: Raw data file not found!")
        return
        
    df = pd.read_csv(raw_path)
    print(f"Original shape: {df.shape}")
    
    # Run cleaning pipeline
    cleaned_df = clean_job_data(df)
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    # Check location standardization
    locations = cleaned_df['location'].unique()
    print(f"Standardized Locations: {locations}")
    
    # Check numeric parsing
    print("\nSample Parsed Data:")
    cols = ['job_title', 'experience_min', 'experience_level', 'salary_mid', 'num_skills']
    print(cleaned_df[cols].head())
    
    # Save cleaned data
    output_path = "data/processed/cleaned_jobs.csv"
    os.makedirs("data/processed", exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to {output_path}")

if __name__ == "__main__":
    test_pipeline()