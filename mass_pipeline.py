import os
import json
import pandas as pd
import time
import logging
from typing import List, Dict, Set

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Local imports
try:
    from src.scraper_selenium import NaukriSeleniumScraper
    from src.utils import clean_job_data
    from src.rag_system import get_known_roles
except ImportError as e:
    logger.error(f"Import Error: {e}. Please ensure the script is run from the project root and 'src' is accessible.")
    exit()

# --- CONFIGURATION ---
TARGET_LOCATIONS: Dict[str, List[str]] = {
    "Mumbai": ["Mumbai"],
    "Delhi": ["Delhi", "New Delhi"],
    "Pune": ["Pune"],
    "Bangalore": ["Bangalore", "Bengaluru"]
}
JOBS_PER_PAIR_QUOTA = 50
STATE_FILE = "mass_scraper_state.json"
CLEANED_JOBS_FILE = "data/processed/cleaned_jobs.csv"
MAX_PAGES_PER_PAIR = 20 # Safety break to avoid infinite loops

# --- STATE MANAGEMENT ---
def load_state() -> Set[str]:
    """Loads the set of completed 'role|location' pairs."""
    if not os.path.exists(STATE_FILE):
        return set()
    try:
        with open(STATE_FILE, 'r') as f:
            return set(json.load(f))
    except (json.JSONDecodeError, IOError):
        logger.warning("Could not read state file, starting fresh.")
        return set()

def save_state(completed_pair: str, state: Set[str]):
    """Saves a newly completed pair to the state file."""
    state.add(completed_pair)
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(list(state), f, indent=2)
        logger.info(f"Progress saved. Completed: {completed_pair}")
    except IOError as e:
        logger.error(f"Failed to save state: {e}")

# --- DATA HANDLING ---
def get_existing_urls() -> Set[str]:
    """Gets all job URLs from the existing cleaned jobs CSV."""
    if not os.path.exists(CLEANED_JOBS_FILE):
        return set()
    try:
        df = pd.read_csv(CLEANED_JOBS_FILE)
        if 'job_url' in df.columns:
            return set(df['job_url'].dropna())
    except pd.errors.EmptyDataError:
        return set()
    except Exception as e:
        logger.error(f"Error reading existing jobs CSV: {e}")
        return set()
    return set()

# --- MAIN SCRAPING LOGIC ---
def main():
    logger.info("--- Starting Mass Job Scraper ---")
    
    # 1. Load roles and state
    all_roles = get_known_roles(pd.DataFrame()) # Pass empty df to get all known roles
    completed_state = load_state()
    
    logger.info(f"Found {len(all_roles)} roles to scrape for {len(TARGET_LOCATIONS)} location groups.")
    logger.info(f"{len(completed_state)} role-location pairs already completed.")

    # Main loops
    for role in all_roles:
        for loc_group_name, loc_aliases in TARGET_LOCATIONS.items():
            
            pair_key = f"{role}|{loc_group_name}"
            if pair_key in completed_state:
                logger.info(f"Skipping already completed pair: {pair_key}")
                continue

            logger.info(f"--- Starting new pair: Role='{role}', Location Group='{loc_group_name}' ---")
            
            # 2. Scrape until quota is met
            unique_new_jobs_for_pair = []
            page_num = 1
            existing_urls = get_existing_urls()
            
            while len(unique_new_jobs_for_pair) < JOBS_PER_PAIR_QUOTA and page_num <= MAX_PAGES_PER_PAIR:
                logger.info(f"Scraping page {page_num} for '{role}' in '{loc_group_name}'...")
                
                try:
                    # Instantiate and use the scraper class
                    scraper = NaukriSeleniumScraper(headless=True)
                    scraped_jobs_df = scraper.scrape_naukri(
                        keywords=[role],
                        locations=loc_aliases,
                        num_pages=1, # We scrape one page at a time
                        start_page=page_num,
                        deep_scrape=True # Ensure we get full details
                    )
                    
                    if scraped_jobs_df.empty:
                        logger.warning(f"No more jobs found for '{role}' in '{loc_group_name}'. Moving to next pair.")
                        break
                        
                    # Filter out jobs we already have
                    scraped_jobs_df['job_url'] = scraped_jobs_df['job_url'].astype(str)
                    new_jobs_df = scraped_jobs_df[~scraped_jobs_df['job_url'].isin(existing_urls)]
                    
                    if not new_jobs_df.empty:
                        new_job_list = new_jobs_df.to_dict('records')
                        unique_new_jobs_for_pair.extend(new_job_list)
                        
                        # Add newly found URLs to existing_urls to avoid duplicates within the same run
                        existing_urls.update(new_jobs_df['job_url'])
                        
                        logger.info(f"Found {len(new_jobs_df)} new unique jobs. Total for this pair: {len(unique_new_jobs_for_pair)}/{JOBS_PER_PAIR_QUOTA}")
                    else:
                        logger.info("Page contained no new unique jobs.")
                        
                except Exception as e:
                    logger.error(f"An error occurred during scraping page {page_num} for {pair_key}: {e}")
                    # Decide whether to break or continue
                    break # Safer to break on error
                    
                page_num += 1
                
                # Rate limiting
                logger.info("Waiting 3 seconds before next page...")
                time.sleep(3)

            # 3. Process and save the batch
            if unique_new_jobs_for_pair:
                logger.info(f"Processing and saving {len(unique_new_jobs_for_pair)} new jobs for {pair_key}...")
                
                new_jobs_batch_df = pd.DataFrame(unique_new_jobs_for_pair)
                
                # Clean the data
                cleaned_df = clean_job_data(new_jobs_batch_df)
                
                # Append to main CSV
                if not cleaned_df.empty:
                    try:
                        header = not os.path.exists(CLEANED_JOBS_FILE)
                        cleaned_df.to_csv(CLEANED_JOBS_FILE, mode='a', header=header, index=False)
                        logger.info(f"Successfully appended {len(cleaned_df)} cleaned jobs to {CLEANED_JOBS_FILE}")
                    except Exception as e:
                        logger.error(f"Failed to write to CSV: {e}")
                else:
                    logger.warning("No jobs remained after cleaning process.")

            # 4. Update state
            save_state(pair_key, completed_state)
            logger.info("--- Pair completed. Waiting 3 seconds before starting next pair... ---")
            time.sleep(3)
            
    logger.info("--- Mass Scraper Finished ---")

if __name__ == "__main__":
    main()
