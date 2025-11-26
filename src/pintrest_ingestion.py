from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
from bs4 import BeautifulSoup
import time
from pathlib import Path
import requests
import re
from datetime import datetime, timezone

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
        pin_url = f"https://www.pinterest.com{href}"
        img_tag = tag.find("img")
        pins.append({
            "pin_id": href.rstrip("/").split("/")[-1],
            "title": tag.get("aria-label") or tag.text.strip(),
            "pin_url": pin_url,
            "image_url": img_tag.get("src") if img_tag else None,
            "alt_text": img_tag.get("alt") if img_tag and img_tag.has_attr("alt") else None
        })

details = []
for pin in pins:
    meta = {}
    try:
        resp = requests.get(
            "https://www.pinterest.com/oembed.json",
            params={"url": pin["pin_url"]},
            timeout=10,
        )
        if resp.ok:
            meta = resp.json()
    except requests.RequestException:
        pass

    description = meta.get("title") or pin["title"] or ""
    tags = "|".join(
        {t.lower().lstrip("#") for t in re.findall(r"#\w+", description)}
    )
    fetch_ts = datetime.now(timezone.utc).isoformat()
    details.append({
        "pin_id": pin["pin_id"],
        "title": (pin["title"] or "").strip(),
        "description": description.strip(),
        "author": meta.get("author_name", "").strip(),
        "created_at": "",
        "repin_count": 0,
        "comment_count": 0,
        "link": meta.get("url") or pin["pin_url"],
        "tags": tags,
        "image_url": pin["image_url"],
        "alt_text": pin["alt_text"],
        "pin_url": pin["pin_url"],
        "fetch_ts": fetch_ts,
    })

df = pd.DataFrame(details).drop_duplicates(subset="pin_id")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Fetched {len(df)} Pinterest posts for '{SEARCH_QUERY}'")
