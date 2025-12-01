import os
import pandas as pd
import subprocess
from pathlib import Path
from rapidfuzz import fuzz
from tqdm import tqdm
import unicodedata
import re
import yt_dlp
import requests
from bs4 import BeautifulSoup
from src.config import download, audio # Adapt to your config

# TODO : Problème sur le merge de l'ancien et nouveau sur les subgenres (pas update)

INPUT_CSV = download.input_soundcloud_csv
OUTPUT_METADATA = download.output_metadata_csv
DOWNLOAD_DIR = download.download_dir
FFMPEG_PATH = "ffmpeg"
DURATION_MAX = audio.max_track_duration

def sanitize_filename(filename):
    """
    Sanitize filename to prevent path traversal and invalid characters.
    Removes or replaces characters that could be interpreted as path separators.
    """
    # Remove leading/trailing whitespace
    filename = filename.strip()

    # Replace path separators with safe alternatives
    filename = filename.replace('/', '-')
    filename = filename.replace('\\', '-')

    # Remove problematic characters for filesystems
    filename = re.sub(r'[<>:"|?*]', '', filename)

    # Replace multiple spaces with single space
    filename = re.sub(r'\s+', ' ', filename)

    # Remove leading/trailing dots and spaces (problematic on Windows)
    filename = filename.strip('. ')

    # Ensure filename is not empty
    if not filename:
        filename = "untitled"

    return filename

def find_matching_file(expected, existing_files, threshold=95):
    for f in existing_files:
        score = fuzz.ratio(expected, f.lower())
        if score >= threshold:
            return f
    return None

def check_download_available(soundcloud_url):
    """
    Check if a track has download enabled on SoundCloud
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(soundcloud_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Check if download button exists
        download_btn = soup.find('button', {'aria-label': 'Download'})
        return download_btn is not None
    except:
        return False

def download_from_soundcloud_direct(url, output_path):
    """
    Download audio directly from SoundCloud if download is enabled
    """
    outtmpl = output_path

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'outtmpl': outtmpl,
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 120,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"Error downloading from SoundCloud: {e}")
        return False

def search_youtube(title, artist):
    """
    Search YouTube for a track and return the best match URL
    """
    query = f"{artist} {title}"

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'default_search': 'ytsearch1',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if result and 'entries' in result and len(result['entries']) > 0:
                video_id = result['entries'][0]['id']
                return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print(f"Error searching for {query}: {e}")

    return None

def download_track_from_youtube(url, output_path):
    """
    Download audio from YouTube using yt-dlp
    """
    outtmpl = output_path

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'outtmpl': outtmpl,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        return False

def download_tracks(df):
    """
    Download SoundCloud tracks either directly or via YouTube search
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        title = str(row["title"])
        artist = str(row["artist"])
        subgenres = str(row["search_genres"])
        soundcloud_url = str(row["soundcloud_url"])

        # Sanitize artist and title before using in filenames
        artist_clean = sanitize_filename(artist)
        title_clean = sanitize_filename(title)

        filename = f"{artist_clean} - {title_clean}.mp3"
        filepath = Path(DOWNLOAD_DIR, filename).as_posix()

        print(f"\nProcessing: {artist} - {title}")
        print(f"Sanitized filename: {filename}")

        # Check if already downloaded
        # Normalize expected filename
        expected_base = f"{artist_clean} - {title_clean}".lower()
        existing_files = os.listdir(DOWNLOAD_DIR)
        matched = find_matching_file(expected_base, existing_files)
        print("Matched :", matched)

        if matched:
            true_path = Path(DOWNLOAD_DIR, matched).as_posix()
            print(f"Already downloaded (fuzzy match: {matched})")
            metadata.append({
                "path": true_path,
                "title": title,
                "artist": artist,
                "subgenres": subgenres
            })
            continue

        # Try downloading directly from SoundCloud first
        print("Attempting direct SoundCloud download...")
        # Use sanitized names for the output template
        output_template = str(Path(DOWNLOAD_DIR, f"{artist_clean} - {title_clean}"))

        success = download_from_soundcloud_direct(soundcloud_url, output_template)

        if success:
            possible_path = f"{output_template}.mp3"
            if os.path.exists(possible_path):
                metadata.append({
                    "path": possible_path,
                    "title": title,
                    "artist": artist,
                    "subgenres": subgenres
                    #"soundcloud_url": soundcloud_url,
                    #"download_source": "soundcloud_direct"
                })
                print("Downloaded directly from SoundCloud!")
                continue

        # If direct download fails, try YouTube
        print("Direct download failed. Searching on YouTube...")
        youtube_url = search_youtube(title, artist)

        if not youtube_url:
            print(f"Could not find on YouTube: {artist} - {title}")
            continue

        print(f"Found on YouTube: {youtube_url}")

        # Check duration
        try:
            info = yt_dlp.YoutubeDL({"quiet": True}).extract_info(youtube_url, download=False)
            duration = info.get("duration", 0)
            title_yt = info.get("title", "").lower()
            print(f"YouTube duration: {duration} sec")

            # Skip long videos
            if duration and duration > DURATION_MAX:
                print("Skipping: duration too long → probable DJ set.")
                continue
        except Exception as e:
            print(f"Could not get duration: {e}")
            continue

        # Download from YouTube
        print("Downloading from YouTube...")
        success = download_track_from_youtube(youtube_url, output_template)

        if success:
            possible_path = f"{output_template}.mp3"
            if os.path.exists(possible_path):
                metadata.append({
                    "path": possible_path,
                    "title": title,
                    "artist": artist,
                    "subgenres": subgenres
                    #"soundcloud_url": soundcloud_url,
                    #"youtube_url": youtube_url,
                    #"download_source": "youtube"
                })
                print("Download successful!")
            else:
                print(f"File not found after download: {possible_path}")
        else:
            print(f"Failed to download: {artist} - {title}")

    return metadata

def main():

    # Load CSV
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        print("Please run fetch_soundcloud_tracks.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} SoundCloud tracks from {INPUT_CSV}")

    # Download tracks
    metadata = download_tracks(df)

    # Save metadata
    if os.path.exists(OUTPUT_METADATA):
        old = pd.read_csv(OUTPUT_METADATA)
        df_new = pd.DataFrame(metadata)
        # Merge and avoid duplicates via file path
        df_meta = pd.concat([old, df_new], ignore_index=True)
        df_meta.drop_duplicates(subset=["path"], inplace=True)
    else:
        df_meta = pd.DataFrame(metadata)

    # Normalize paths
    if len(df_meta) > 0:
        df_meta["path"] = df_meta["path"].apply(
            lambda p: p.replace("\\", "/") if isinstance(p, str) else p
        )

    df_meta.to_csv(OUTPUT_METADATA, index=False)
    print(f"\nSaved metadata for {len(df_meta)} tracks to {OUTPUT_METADATA}")

if __name__ == "__main__":
    main()
