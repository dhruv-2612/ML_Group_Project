import os
import glob
import pandas as pd
import logging
from src.utils import clean_job_data
import src.ml_analyzer as ml

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_all_raw_data():
    logger.info("Starting processing of all raw data files...")
    raw_files = glob.glob("data/raw/naukri_*.csv")
    if not raw_files:
        logger.warning("No raw data files found.")
        return

    df_list = []
    for f in raw_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
            logger.info(f"Loaded {f} with {len(df)} records.")
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")

    if not df_list:
        return

    raw_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total raw records: {len(raw_df)}")
    
    # Clean data
    cleaned_df = clean_job_data(raw_df)
    logger.info(f"Cleaned records: {len(cleaned_df)}")
    
    os.makedirs("data/processed", exist_ok=True)
    cleaned_df.to_csv("data/processed/cleaned_jobs.csv", index=False)
    
    # ML Analysis
    logger.info("Running ML Analysis (Clustering)...")
    clustered_df, kmeans, keywords = ml.cluster_job_roles(cleaned_df, n_clusters=12)
    clustered_df.to_csv("data/processed/job_clusters.csv", index=False)
    
    # Salary Analysis
    logger.info("Analyzing Salary Trends (Statistical)...")
    try:
        salary_stats = ml.analyze_salary_trends(clustered_df)
        logger.info(f"Global Median Salary: {salary_stats.get('global_median', 'N/A')} LPA")
    except Exception as e:
        logger.error(f"Salary analysis failed: {e}")

    logger.info("Processing complete. Files saved in data/processed/")

if __name__ == "__main__":
    process_all_raw_data()
