import os
import sys
import glob
import logging
import argparse
import pandas as pd
import time
from datetime import datetime

# Adjust path to allow imports from src
sys.path.append(os.getcwd())

try:
    from src.scraper import JobScraper
    from src.scraper_selenium import NaukriSeleniumScraper
    from src.utils import clean_job_data
    import src.ml_analyzer as ml
    import src.rag_system as rag
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running from the project root directory.")
    sys.exit(1)

# Setup Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_full_pipeline(keywords_list: list, locations_list: list, num_pages: int = 2, use_selenium: bool = False, start_page: int = 1):
    """
    Executes the complete Job Intelligence pipeline.
    Returns: count of TRULY NEW jobs added to the system.
    """
    start_time = time.time()
    logger.info("🚀 Starting Job Intelligence Pipeline")
    
    # 0. Load existing data count for comparison
    existing_count = 0
    if os.path.exists("data/processed/cleaned_jobs.csv"):
        try:
            existing_count = len(pd.read_csv("data/processed/cleaned_jobs.csv"))
        except: pass

    # --- STEP 1: SCRAPING ---
    # ... (Scraping logic) ...
    if use_selenium:
        scraper = NaukriSeleniumScraper(headless=True)
        try:
            scraper.scrape_naukri(keywords_list, locations_list, num_pages=num_pages, deep_scrape=True, start_page=start_page)
        except Exception as e:
            logger.error(f"Selenium Scraping failed: {e}")
    
    # --- STEP 2: DATA CLEANING & PROCESSING ---
    logger.info("\n--- STEP 2: DATA CLEANING & PROCESSING ---")
    try:
        raw_files = glob.glob("data/raw/naukri_*.csv")
        if not raw_files:
            logger.warning("No raw data files found.")
            return 0

        df_list = [pd.read_csv(f) for f in raw_files]
        raw_df = pd.concat(df_list, ignore_index=True)
        
        # Freshness filter included in clean_job_data
        cleaned_df = clean_job_data(raw_df)
        
        new_total_count = len(cleaned_df)
        truly_new_added = new_total_count - existing_count
        
        if truly_new_added <= 0:
            logger.info("ℹ️ No new unique job postings found after cleaning. Skipping ML and RAG update.")
            return 0
            
        cleaned_df.to_csv("data/processed/cleaned_jobs.csv", index=False)
        logger.info(f"✅ Success: {truly_new_added} truly new jobs added. Proceeding to ML/RAG...")
        
    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        return 0

    # --- STEP 3: ML ANALYSIS ---
    logger.info("\n--- STEP 3: ML MODEL TRAINING & ANALYSIS ---")
    try:
        # Clustering
        clustered_df, kmeans, keywords = ml.cluster_job_roles(cleaned_df, n_clusters=6)
        clustered_df.to_csv("data/processed/job_clusters.csv", index=False)
        
        # Visualization
        ml.plot_clusters_visualization(clustered_df)
        
    except Exception as e:
        logger.error(f"ML Analysis failed: {e}")

    # --- STEP 4: VECTOR STORE CREATION ---
    logger.info("\n--- STEP 4: RAG SYSTEM INDEXING ---")
    try:
        final_df = pd.read_csv("data/processed/job_clusters.csv")
        vectorstore = rag.create_vector_store(final_df)
        logger.info("Vector store successfully updated.")
    except Exception as e:
        logger.error(f"Vector store update failed: {e}")

    # Final Summary
    elapsed = time.time() - start_time
    logger.info(f"\n✅ Pipeline Complete. Added {truly_new_added} new records in {elapsed:.1f}s.")
    return truly_new_added

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Intelligence Pipeline Runner")
    parser.add_argument("--keywords", type=str, default="Product Manager,Data Analyst", help="Comma-separated job titles")
    parser.add_argument("--locations", type=str, default="Bangalore,Mumbai", help="Comma-separated locations")
    parser.add_argument("--pages", type=int, default=2, help="Pages per keyword/location")
    parser.add_argument("--start-page", type=int, default=1, help="Page number to start scraping from")
    parser.add_argument("--use-selenium", action="store_true", help="Use Selenium for deep scraping")
    
    args = parser.parse_args()
    
    kw_list = [k.strip() for k in args.keywords.split(",")]
    loc_list = [l.strip() for l in args.locations.split(",")]
    
    run_full_pipeline(kw_list, loc_list, args.pages, args.use_selenium, args.start_page)
