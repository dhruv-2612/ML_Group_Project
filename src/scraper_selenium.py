from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import random
import os
from datetime import datetime
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NaukriSeleniumScraper:
    def __init__(self, headless=True):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
        self.driver = None

    def start_driver(self):
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.options)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            logger.error(f"Could not start Chrome Driver: {e}")
            raise

    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def scrape_job_details(self, url):
        """Visits a job URL and extracts full description and skills."""
        try:
            self.driver.get(url)
            time.sleep(random.uniform(4, 5))
            
            # 1. Handle "Read More" if present
            try:
                # Look for 'read more' button specifically within the job description section
                read_more = self.driver.find_elements(By.XPATH, "//span[contains(text(), 'read more')] | //button[contains(text(), 'read more')]")
                if read_more:
                    self.driver.execute_script("arguments[0].click();", read_more[0])
                    time.sleep(1)
            except:
                pass

            # 2. Extract Full Description
            description = ""
            try:
                jd_container = self.driver.find_element(By.CSS_SELECTOR, "section[class*='job-desc-container'], .styles_job-desc-container__txpYf")
                description = jd_container.text
            except:
                # Fallback
                try:
                    sections = self.driver.find_elements(By.TAG_NAME, "section")
                    for sec in sections:
                        if "Job description" in sec.text:
                            description = sec.text
                            break
                except:
                    description = "Description not found"

            # 3. Extract Skills
            skills = []
            try:
                headings = self.driver.find_elements(By.CSS_SELECTOR, "div[class*='heading'], .styles_heading__veHpg, h2, h3")
                for h in headings:
                    if "Key Skills" in h.text:
                        parent = h.find_element(By.XPATH, "..")
                        skill_tags = parent.find_elements(By.CSS_SELECTOR, "a, span")
                        skills = [s.text for s in skill_tags if s.text and 2 < len(s.text) < 30]
                        skills = list(set(skills))
                        break
            except:
                pass
                
            return description, skills
        except Exception as e:
            logger.error(f"Error scraping details for {url}: {e}")
            return "Error", []

    def scrape_naukri(self, keywords, locations, num_pages=2, deep_scrape=True, start_page=1):
        self.start_driver()
        # Dictionary to handle duplicates and merge locations: {job_url: job_dict}
        jobs_by_url = {} 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load existing URLs to avoid redundant deep scraping
        existing_urls = set()
        df_path = "data/processed/cleaned_jobs.csv"
        if os.path.exists(df_path):
            try:
                existing_df = pd.read_csv(df_path)
                if 'job_url' in existing_df.columns:
                    existing_urls = set(existing_df['job_url'].dropna().unique())
                    logger.info(f"Loaded {len(existing_urls)} existing URLs to skip deep scraping.")
            except Exception as e:
                logger.warning(f"Could not load existing jobs for filtering: {e}")

        try:
            for kw in keywords:
                for loc in locations:
                    # Retry logic: if we get too many duplicates, we try more pages
                    max_attempts = 2
                    current_attempt = 1
                    local_num_pages = num_pages if num_pages > 0 else 2
                    current_start_page = start_page
                    
                    while current_attempt <= max_attempts:
                        logger.info(f"Attempt {current_attempt} for {kw} in {loc} (Pages: {current_start_page} to {current_start_page + local_num_pages - 1})")
                        kw_slug = kw.lower().replace(" ", "-")
                        loc_slug = loc.lower().replace(" ", "-")
                        
                        jobs_found_in_attempt = 0
                        
                        for page in range(current_start_page, current_start_page + local_num_pages):
                            url = f"https://www.naukri.com/{kw_slug}-jobs-in-{loc_slug}-{page}"
                            logger.info(f"Scraping Page: {url}")
                            
                            self.driver.get(url)
                            time.sleep(random.uniform(3, 4))
                            
                            cards = self.driver.find_elements(By.CSS_SELECTOR, ".srp-jobtuple-wrapper, [data-job-id], .jobTuple")
                            logger.info(f"Found {len(cards)} job summaries on page {page}")
                            
                            jobs_found_in_attempt += len(cards)
                            
                            for card in cards:
                                try:
                                    title_el = card.find_element(By.CSS_SELECTOR, "a.title, .title")
                                    comp_el = card.find_element(By.CSS_SELECTOR, ".comp-name, .companyName")
                                    
                                    job_url = title_el.get_attribute("href")
                                    
                                    # Handle Location Merging
                                    if job_url in jobs_by_url:
                                        # Already found this job, just update location
                                        if loc not in jobs_by_url[job_url]['location']:
                                            jobs_by_url[job_url]['location'].append(loc)
                                        continue # Skip creating new entry
                                    
                                    # New Job
                                    job = {
                                        'job_title': title_el.text,
                                        'company': comp_el.text,
                                        'location': [loc], # Initialize as list
                                        'job_url': job_url,
                                        'source': 'Naukri_Selenium',
                                        'scraped_date': datetime.now().isoformat()
                                    }
                                    
                                    # Basic info from summary
                                    try:
                                        exp_el = card.find_element(By.CSS_SELECTOR, ".exp-wrap, .exp")
                                        job['experience_required'] = exp_el.text
                                    except: job['experience_required'] = "N/A"
                                    
                                    try:
                                        sal_el = card.find_element(By.CSS_SELECTOR, ".sal-wrap, .salary")
                                        job['salary'] = sal_el.text
                                    except: job['salary'] = "Not Disclosed"

                                    jobs_by_url[job_url] = job
                                except Exception:
                                    continue

                        # Check if we should push deeper
                        # Compare detected new jobs vs existing system data
                        new_unique_count = 0
                        for url, job in jobs_by_url.items():
                            clean_url = url.split('?')[0]
                            if clean_url not in existing_urls:
                                new_unique_count += 1
                                
                        logger.info(f"Current session has {len(jobs_by_url)} unique jobs. {new_unique_count} are truly new to system.")

                        # Heuristic: If we found very few *truly new* jobs in this attempt, try next pages
                        # Only applicable if we actually found cards (to avoid infinite loops on empty results)
                        if jobs_found_in_attempt > 0 and new_unique_count < 3 and current_attempt < max_attempts:
                             logger.warning("Low volume of new data. Pushing deeper into pagination...")
                             current_start_page += local_num_pages 
                             current_attempt += 1
                             continue
                        else:
                             break

            # Deep Scrape Details (ONLY for jobs not already in system)
            all_jobs_list = list(jobs_by_url.values())
            
            if deep_scrape and all_jobs_list:
                # Filter for truly new jobs to deep scrape
                truly_new_jobs = []
                for job in all_jobs_list:
                    clean_url = job['job_url'].split('?')[0]
                    if clean_url not in existing_urls:
                        truly_new_jobs.append(job)
                
                logger.info(f"Starting deep scrape for {len(truly_new_jobs)} NEW unique jobs...")
                for job in tqdm(truly_new_jobs, desc="Extracting full details"):
                    if job['job_url']:
                        desc, skills = self.scrape_job_details(job['job_url'])
                        job['job_description'] = desc
                        job['skills_list'] = skills
                        time.sleep(random.uniform(2, 4)) # Responsible scraping
                
                # We return ALL jobs found in this session (even if not deep scraped, they might have location updates)
                # But typically pipeline expects deep details.
                # Ideally, we should merge this with existing data, but for now we just return what we found.
                # Ideally we only save the ones we processed + any that were just location updates?
                # Simpler: Return all found in this session. The pipeline will merge.
            
            if all_jobs_list:
                df = pd.DataFrame(all_jobs_list)
                os.makedirs("data/raw", exist_ok=True)
                filename = f"data/raw/naukri_deep_scraped_{timestamp}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Successfully saved {len(df)} detailed jobs to {filename}")
                return df
            
        finally:
            self.close_driver()

if __name__ == "__main__":
    scraper = NaukriSeleniumScraper(headless=True)
    # Test with 5 jobs to verify deep scraping
    scraper.scrape_naukri(["Product Manager"], ["Bangalore"], num_pages=1, deep_scrape=True)
