from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
from bs4 import BeautifulSoup
import time
from pathlib import Path

SEARCH_QUERY = "AI content marketing"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_CSV = DATA_DIR / "pinterest_posts_detailed.csv"

driver_path = r"C:\Users\Prabha\Desktop\AI_Content_Optimizerr\AI_Powered_Content_Marketing_Optimizer\chromedriver.exe"

options = Options()
# options.add_argument("--headless")  # comment out for debugging
options.add_argument("--start-maximized")
options.add_argument("--headless=new")   # run completely in background
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")
# Use Service object instead of executable_path
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=options)

driver.get(f"https://www.pinterest.com/search/pins/?q={SEARCH_QUERY}")

scroll_pause_time = 5
num_scrolls = 10

last_height = driver.execute_script("return document.body.scrollHeight")
for i in range(num_scrolls):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

pins = []
for tag in soup.find_all("a", href=True):
    href = tag["href"]
    if "/pin/" in href:
        img_tag = tag.find("img")
        pins.append({
            "title": tag.get("aria-label") or tag.text.strip(),
            "pin_url": f"https://www.pinterest.com{href}",
            "image_url": img_tag.get("src") if img_tag else None,
            "alt_text": img_tag.get("alt") if img_tag and img_tag.has_attr("alt") else None
        })

df = pd.DataFrame(pins).drop_duplicates(subset="pin_url")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Fetched {len(df)} Pinterest postss for '{SEARCH_QUERY}'")
