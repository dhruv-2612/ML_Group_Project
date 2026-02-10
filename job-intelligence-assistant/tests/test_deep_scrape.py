from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_deep_scrape(url):
    options = Options()
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        logger.info(f"Visiting: {url}")
        driver.get(url)
        time.sleep(5)  # Wait for load
        
        # 1. Extract Job Description
        try:
            jd_element = driver.find_element(By.CSS_SELECTOR, ".job-desc, .dang-inner-html")
            print("\n--- JOB DESCRIPTION EXTRACTED ---")
            print(jd_element.text[:500] + "...") 
        except Exception as e:
            print(f"\nCould not find Job Description: {e}")

        # 2. Extract Key Skills
        try:
            skills_elements = driver.find_elements(By.CSS_SELECTOR, "a.chip, .key-skill")
            skills = [s.text for s in skills_elements if s.text]
            print("\n--- SKILLS EXTRACTED ---")
            print(skills)
        except Exception as e:
            print(f"\nCould not find Skills: {e}")

    finally:
        driver.quit()

if __name__ == "__main__":
    import pandas as pd
    try:
        import glob
        import os
        files = glob.glob("data/processed/live_cleaned_jobs.csv")
        if files:
            df = pd.read_csv(files[0])
            if not df.empty and 'job_url' in df.columns:
                test_url = df['job_url'].iloc[0]
                test_deep_scrape(test_url)
            else:
                print("CSV is empty or missing job_url column.")
        else:
            print("No scraped CSV found. Please provide a URL.")
    except Exception as e:
        print(f"Error reading CSV: {e}")