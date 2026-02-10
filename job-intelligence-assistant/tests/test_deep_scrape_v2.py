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

def test_deep_scrape_refined(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        logger.info(f"Visiting: {url}")
        driver.get(url)
        time.sleep(7)  # Increased wait for full rendering
        
        # 1. Extract Job Description
        try:
            jd_container = driver.find_element(By.CSS_SELECTOR, "section[class*='job-desc-container'], .styles_job-desc-container__txpYf")
            print("\n--- JOB DESCRIPTION FOUND ---")
            print(jd_container.text[:800] + "...") 
        except Exception as e:
            print(f"\nCould not find Job Description via specific class. Trying fallback...")
            try:
                sections = driver.find_elements(By.TAG_NAME, "section")
                for sec in sections:
                    if "Job description" in sec.text:
                        print("Found Job description in a generic section.")
                        print(sec.text[:800] + "...")
                        break
            except:
                print("Fallback also failed.")

        # 2. Extract Key Skills
        try:
            headings = driver.find_elements(By.CSS_SELECTOR, "div[class*='heading'], .styles_heading__veHpg, h2, h3")
            for h in headings:
                if "Key Skills" in h.text:
                    print("\n--- KEY SKILLS HEADING FOUND ---")
                    parent = h.find_element(By.XPATH, "..")
                    skill_tags = parent.find_elements(By.CSS_SELECTOR, "a, span")
                    skills = [s.text for s in skill_tags if s.text and 2 < len(s.text) < 30]
                    print(f"Extracted Skills: {list(set(skills))}")
                    break
        except Exception as e:
            print(f"\nCould not find Skills: {e}")

    finally:
        driver.quit()

if __name__ == "__main__":
    import pandas as pd
    try:
        import glob
        files = glob.glob("data/processed/live_cleaned_jobs.csv")
        if files:
            df = pd.read_csv(files[0])
            if not df.empty and 'job_url' in df.columns:
                test_url = df['job_url'].iloc[0]
                test_deep_scrape_refined(test_url)
    except Exception as e:
        print(f"Error: {e}")