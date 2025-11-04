import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ================= CONFIG =================
SUBGENRES_BEATPORT = {
    "hard-techno": 2,
    "minimal-deep-tech": 14,
    "tech-house": 11,
    "melodic-house-and-techno": 90,
    "trance": 7
}

PAGES_PER_SUBGENRE = 2        # nombre de pages à parcourir (chaque page ~50 tracks)
OUTPUT_CSV = "beatport_tracks.csv"
DOWNLOAD_PREVIEWS = True      # mettre False si tu veux juste le CSV
DOWNLOAD_DIR = "data/beatport_previews"
SLEEP_BETWEEN_PAGES = 2.0     # attendre entre les chargements
HEADLESS = True               # afficher ou non Chrome
# ===========================================


def setup_driver():
    """Initialise Chrome headless."""
    chrome_options = Options()
    if HEADLESS:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=chrome_options)
    return driver


def fetch_beatport_tracks(driver, slug, genre_id, pages=1):
    """Utilise Selenium pour charger les pages Beatport et extraire les morceaux."""
    all_tracks = []

    for page in range(1, pages + 1):
        url = f"https://www.beatport.com/genre/{slug}/{genre_id}/tracks?page={page}"
        print(f"🌐 Fetching {url}")
        driver.get(url)
        time.sleep(SLEEP_BETWEEN_PAGES)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        items = soup.select("li.bucket-item.ec-item.track")

        if not items:
            print(f"❌ No tracks found on page {page} ({slug})")
            continue

        for item in items:
            try:
                title = item.select_one("p.ec-item-track").get_text(strip=True)
                artist = item.select_one("p.ec-item-artist").get_text(strip=True)
                label = item.select_one("p.ec-item-label").get_text(strip=True)
                bpm = item.select_one("p.ec-item-bpm").get_text(strip=True)
                release_date = item.select_one("p.ec-item-released").get_text(strip=True)
                preview_tag = item.select_one("a.playable-play")
                preview_url = preview_tag.get("data-preview-url") if preview_tag else None

                all_tracks.append({
                    "title": title,
                    "artist": artist,
                    "label": label,
                    "bpm": bpm,
                    "release_date": release_date,
                    "preview_url": preview_url,
                    "subgenre": slug.replace("-", " ")
                })
            except Exception:
                continue

        print(f"✅ Found {len(items)} tracks on page {page} ({slug})")

    print(f"🎶 Total {len(all_tracks)} tracks collected for {slug}")
    return all_tracks


def download_previews(df):
    """Télécharge les previews MP3 (optionnel)."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for _, row in df.iterrows():
        genre_folder = os.path.join(DOWNLOAD_DIR, row["subgenre"].replace(" ", "_"))
        os.makedirs(genre_folder, exist_ok=True)

        if not row["preview_url"]:
            continue

        filename = os.path.join(genre_folder, f"{row['artist']} - {row['title']}.mp3")
        if os.path.exists(filename):
            continue

        try:
            r = requests.get(row["preview_url"], timeout=10)
            with open(filename, "wb") as f:
                f.write(r.content)
        except Exception:
            continue


def main():
    driver = setup_driver()
    all_tracks = []

    for slug, gid in SUBGENRES_BEATPORT.items():
        tracks = fetch_beatport_tracks(driver, slug, gid, pages=PAGES_PER_SUBGENRE)
        all_tracks.extend(tracks)

    driver.quit()

    df = pd.DataFrame(all_tracks)
    df.drop_duplicates(subset=["title", "artist"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved {len(df)} total tracks to {OUTPUT_CSV}")

    if DOWNLOAD_PREVIEWS:
        print("\n🎵 Downloading preview MP3s...")
        download_previews(df)
        print("✅ All previews downloaded.")


if __name__ == "__main__":
    main()
