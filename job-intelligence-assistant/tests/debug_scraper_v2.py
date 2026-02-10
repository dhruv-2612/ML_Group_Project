import requests
import random

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
]

def test_naukri_full_html():
    url = "https://www.naukri.com/product-manager-jobs-in-bangalore"
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open("debug_full.html", "w", encoding="utf-8") as f:
                f.write(response.text)
            print("Saved full HTML to debug_full.html")
            
            # Search for typical data indicators in the text
            if "__NEXT_DATA__" in response.text:
                print("Found __NEXT_DATA__")
            if "jobTuple" in response.text:
                print("Found jobTuple")
            if "searchPageState" in response.text:
                print("Found searchPageState")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_naukri_full_html()
