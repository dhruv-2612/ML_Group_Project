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
            time.sleep(random.uniform(4, 6))
            
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

    def scrape_naukri(self, keywords, locations, num_pages=1, deep_scrape=True):
        self.start_driver()
        all_jobs = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            for kw in keywords:
                for loc in locations:
                    kw_slug = kw.lower().replace(" ", "-")
                    loc_slug = loc.lower().replace(" ", "-")
                    
                    for page in range(1, num_pages + 1):
                        url = f"https://www.naukri.com/{kw_slug}-jobs-in-{loc_slug}-{page}"
                        logger.info(f"Scraping Page: {url}")
                        
                        self.driver.get(url)
                        time.sleep(random.uniform(5, 7))
                        
                        cards = self.driver.find_elements(By.CSS_SELECTOR, ".srp-jobtuple-wrapper, [data-job-id], .jobTuple")
                        logger.info(f"Found {len(cards)} job summaries on page {page}")
                        
                        for card in cards:
                            try:
                                title_el = card.find_element(By.CSS_SELECTOR, "a.title, .title")
                                comp_el = card.find_element(By.CSS_SELECTOR, ".comp-name, .companyName")
                                
                                job = {
                                    'job_title': title_el.text,
                                    'company': comp_el.text,
                                    'location': loc,
                                    'job_url': title_el.get_attribute("href"),
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

                                all_jobs.append(job)
                            except Exception:
                                continue

            # Deep Scrape Details
            if deep_scrape and all_jobs:
                logger.info(f"Starting deep scrape for {len(all_jobs)} jobs...")
                for job in tqdm(all_jobs, desc="Extracting full details"):
                    if job['job_url']:
                        desc, skills = self.scrape_job_details(job['job_url'])
                        job['job_description'] = desc
                        job['skills_list'] = skills
                        time.sleep(random.uniform(2, 4)) # Responsible scraping
            
            if all_jobs:
                df = pd.DataFrame(all_jobs)
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
