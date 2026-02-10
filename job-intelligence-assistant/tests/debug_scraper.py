import requests
from bs4 import BeautifulSoup
import json
import random

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
]

def test_naukri_single_page():
    url = "https://www.naukri.com/product-manager-jobs-in-bangalore"
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    
    print(f"Testing URL: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            scripts = soup.find_all('script')
            found_state = False
            for script in scripts:
                if script.string and 'window._initialState' in script.string:
                    found_state = True
                    print("Found window._initialState JSON!")
                    # Just print first 200 chars of the JSON
                    json_start = script.string.find('window._initialState = ') + len('window._initialState = ')
                    print(f"JSON preview: {script.string[json_start:json_start+200]}...")
                    break
            
            if not found_state:
                print("Could not find window._initialState. Page might be protected or structure changed.")
                # Save a snippet of HTML for debugging
                with open("debug_naukri.html", "w", encoding="utf-8") as f:
                    f.write(soup.prettify()[:2000])
                print("Saved HTML snippet to debug_naukri.html")
        else:
            print(f"Failed to fetch page. Status: {response.status_code}")
            
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    test_naukri_single_page()
