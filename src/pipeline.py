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
    from src.generate_sample_data import generate_synthetic_job_data
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

def run_full_pipeline(keywords_list: list, locations_list: list, num_pages: int = 2, use_selenium: bool = False):
    """
    Executes the complete Job Intelligence pipeline:
    Scrape -> Clean -> ML Analysis -> RAG Indexing -> Validation
    """
    start_time = time.time()
    logger.info("🚀 Starting Job Intelligence Pipeline")
    logger.info(f"Parameters: Keywords={keywords_list}, Locations={locations_list}, Pages={num_pages}, Selenium={use_selenium}")
    
    # --- STEP 1: SCRAPING ---
    logger.info("\n--- STEP 1: DATA ACQUISITION ---")
    
    if use_selenium:
        logger.info("Using Selenium Scraper (Deep Scrape enabled)")
        scraper = NaukriSeleniumScraper(headless=True)
        try:
            scraper.scrape_naukri(keywords_list, locations_list, num_pages=num_pages, deep_scrape=True)
        except Exception as e:
            logger.error(f"Selenium Scraping failed: {e}")
    else:
        scraper = JobScraper()
        # Try scrapping
        try:
            scraper.scrape_naukri_jobs(keywords_list, locations_list, num_pages=num_pages)
        except Exception as e:
            logger.error(f"Standard Scraping failed: {e}")
        
    # Check if data was collected
    raw_files = glob.glob("data/raw/naukri_*.csv")
    
    if not raw_files:
        logger.error("❌ No scraped data found in data/raw/. Please check your scraper or internet connection.")
        logger.info("Pipeline stopped: insufficient data to proceed without synthetic fallback.")
        return
    else:
        logger.info(f"Found {len(raw_files)} raw data files.")

    # --- STEP 2: DATA CLEANING ---
    logger.info("\n--- STEP 2: DATA CLEANING & PROCESSING ---")
    try:
        df_list = []
        for f in raw_files:
            try:
                df_list.append(pd.read_csv(f))
            except Exception as e:
                logger.warning(f"Could not read {f}: {e}")
                
        if not df_list:
            logger.error("No data to process. Exiting.")
            return

        raw_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Total raw records: {len(raw_df)}")
        
        cleaned_df = clean_job_data(raw_df)
        
        output_path = "data/processed/cleaned_jobs.csv"
        os.makedirs("data/processed", exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        
        logger.info(f"Cleaned data saved to {output_path} ({len(cleaned_df)} records)")
        logger.info(f"Top Companies: {', '.join(cleaned_df['company'].value_counts().head(3).index)}")
        
    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        return

    # --- STEP 3: ML ANALYSIS ---
    logger.info("\n--- STEP 3: ML MODEL TRAINING & ANALYSIS ---")
    try:
        # Clustering
        clustered_df, kmeans, keywords = ml.cluster_job_roles(cleaned_df, n_clusters=6)
        clustered_df.to_csv("data/processed/job_clusters.csv", index=False)
        logger.info(f"Clustering complete. Data saved to data/processed/job_clusters.csv")
        
        # Visualization
        ml.plot_clusters_visualization(clustered_df)
        
        # Skills Analysis (for a sample role)
        sample_role = keywords_list[0] if keywords_list else "Product Manager"
        ml.generate_skills_report(clustered_df, role=sample_role)
        logger.info(f"Generated skills report for {sample_role}")
        
        # Salary Model
        model, feat_imp, metrics, cols = ml.train_salary_predictor(clustered_df)
        logger.info(f"Salary Model Trained. Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"ML Analysis failed: {e}")
        # Continue to RAG even if ML fails partially? Yes, RAG can work without full ML artifacts if needed.

    # --- STEP 4: VECTOR STORE CREATION ---
    logger.info("\n--- STEP 4: RAG SYSTEM INDEXING ---")
    try:
        # Reload clustered data to ensure we have cluster IDs in vector store
        final_df = pd.read_csv("data/processed/job_clusters.csv")
        
        # Create Store
        vectorstore = rag.create_vector_store(final_df)
        
        # Add Guides
        rag.add_career_guides_to_store(vectorstore)
        
        logger.info("Vector store successfully created and persisted.")
        
    except Exception as e:
        logger.error(f"Vector store creation failed: {e}")
        return

    # --- STEP 5: VALIDATION ---
    logger.info("\n--- STEP 5: SYSTEM VALIDATION ---")
    try:
        # Load chain
        chain, retriever = rag.create_rag_chain(vectorstore)
        
        test_q = f"What skills do {keywords_list[0]}s need?"
        logger.info(f"Testing Query: {test_q}")
        
        # Use simple query first
        result = rag.query_job_intelligence(test_q, chain)
        
        print("\n" + "="*40)
        print("🤖 SAMPLE RESPONSE")
        print("="*40)
        print(result['answer'])
        print("="*40 + "\n")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")

    # Final Summary
    elapsed = time.time() - start_time
    logger.info(f"\n✅ Pipeline Complete in {elapsed:.1f} seconds.")
    logger.info("Ready to launch Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Intelligence Pipeline Runner")
    parser.add_argument("--keywords", type=str, default="Product Manager,Data Analyst", help="Comma-separated job titles")
    parser.add_argument("--locations", type=str, default="Bangalore,Mumbai", help="Comma-separated locations")
    parser.add_argument("--pages", type=int, default=2, help="Pages per keyword/location")
    parser.add_argument("--use-selenium", action="store_true", help="Use Selenium for deep scraping")
    
    args = parser.parse_args()
    
    kw_list = [k.strip() for k in args.keywords.split(",")]
    loc_list = [l.strip() for l in args.locations.split(",")]
    
    run_full_pipeline(kw_list, loc_list, args.pages, args.use_selenium)
