import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
import json
from datetime import datetime
from tqdm import tqdm
import logging
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseScraper:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
        ]

    def get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def clean_text(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'<[^>]+>', '', text)
        return re.sub(r'\s+', ' ', text).strip()

class JobScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        self.output_dir = "data/raw"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def scrape_naukri_jobs(self, keywords: List[str], locations: List[str], num_pages: int = 10):
        """Scrape jobs from Naukri.com using initial state JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for keyword in keywords:
            all_jobs = []
            kw_slug = keyword.lower().replace(" ", "-")
            for location in locations:
                loc_slug = location.lower().replace(" ", "-")
                for page in tqdm(range(1, num_pages + 1), desc=f"Naukri: {keyword} in {location}"):
                    url = f"https://www.naukri.com/{kw_slug}-jobs-in-{loc_slug}-{page}"
                    try:
                        resp = requests.get(url, headers=self.get_headers(), timeout=15)
                        resp.raise_for_status()
                        soup = BeautifulSoup(resp.content, 'html.parser')
                        scripts = soup.find_all('script')
                        for script in scripts:
                            if script.string and 'window._initialState' in script.string:
                                json_text = script.string.split('window._initialState = ')[1].split(';')[0]
                                data = json.loads(json_text)
                                tuples = data.get('searchPageState', {}).get('details', {}).get('jobTuple', [])
                                for job in tuples:
                                    all_jobs.append({
                                        'job_title': job.get('title'),
                                        'company': job.get('companyName'),
                                        'location': job.get('location'),
                                        'experience_required': job.get('experience'),
                                        'skills_list': job.get('tagsAndSkills'),
                                        'job_description': self.clean_text(job.get('jobDescription', '')),
                                        'salary': job.get('salaryRange', 'Not Disclosed'),
                                        'posted_date': job.get('footerPlaceholderLabel'),
                                        'job_url': job.get('jdURL'),
                                        'source': 'Naukri',
                                        'scraped_date': datetime.now().isoformat()
                                    })
                                break
                    except Exception as e:
                        logger.error(f"Naukri Error {url}: {e}")
                    time.sleep(random.uniform(2.0, 4.0))
            
            if all_jobs:
                df = pd.DataFrame(all_jobs).drop_duplicates(subset=['job_url'])
                filename = f"{self.output_dir}/naukri_jobs_{kw_slug}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(df)} Naukri jobs for {keyword}")

    def scrape_instahyre_jobs(self, keywords: List[str], locations: List[str], num_pages: int = 5):
        """
        Scrape jobs from Instahyre.com.
        Note: Instahyre is heavily JS-dependent. This attempts to find searchable patterns 
        or common elements if available via requests.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for keyword in keywords:
            all_jobs = []
            kw_query = keyword.replace(" ", "+")
            for location in locations:
                loc_query = location.replace(" ", "+")
                for page in tqdm(range(1, num_pages + 1), desc=f"Instahyre: {keyword} in {location}"):
                    # Instahyre's URL pattern for search
                    url = f"https://www.instahyre.com/search-jobs/?q={kw_query}&l={loc_query}"
                    try:
                        # Note: This is a simplified BS4 approach. 
                        # Real-world Instahyre often requires Selenium or their internal API.
                        resp = requests.get(url, headers=self.get_headers(), timeout=15)
                        resp.raise_for_status()
                        soup = BeautifulSoup(resp.content, 'html.parser')
                        
                        # Looking for job cards (CSS selectors may vary based on site updates)
                        job_cards = soup.select('.job-listing') or soup.select('[id^="job-"]')
                        
                        for card in job_cards:
                            all_jobs.append({
                                'job_title': self.clean_text(card.select_one('.job-title').text) if card.select_one('.job-title') else "N/A",
                                'company': self.clean_text(card.select_one('.company-name').text) if card.select_one('.company-name') else "N/A",
                                'location': location,
                                'experience_required': self.clean_text(card.select_one('.exp').text) if card.select_one('.exp') else "N/A",
                                'skills_list': [s.text.strip() for s in card.select('.skill')],
                                'job_description': "Visit URL for description",
                                'salary': "Not Disclosed",
                                'posted_date': "Recently",
                                'job_url': "https://www.instahyre.com" + card.find('a')['href'] if card.find('a') else "N/A",
                                'source': 'Instahyre',
                                'scraped_date': datetime.now().isoformat()
                            })
                    except Exception as e:
                        logger.error(f"Instahyre Error {url}: {e}")
                    time.sleep(random.uniform(2.0, 4.0))

            if all_jobs:
                df = pd.DataFrame(all_jobs).drop_duplicates(subset=['job_url'])
                filename = f"{self.output_dir}/instahyre_jobs_{keyword.lower().replace(' ', '_')}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(df)} Instahyre jobs for {keyword}")

if __name__ == "__main__":
    target_keywords = ["Product Manager", "Data Analyst"]
    target_locations = ["Bangalore", "Mumbai"]
    
    scraper = JobScraper()
    
    # Run Naukri Scraper
    scraper.scrape_naukri_jobs(keywords=target_keywords, locations=target_locations, num_pages=2)
    
    # Run Instahyre Scraper (Backup)
    scraper.scrape_instahyre_jobs(keywords=target_keywords, locations=target_locations, num_pages=1)