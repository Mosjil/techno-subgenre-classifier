import os
import pandas as pd
import argparse
import asyncio
import aiohttp
import logging
import shlex
from rapidfuzz import fuzz
from dotenv import load_dotenv
from soundcloudpy import SoundcloudAsyncAPI

# TODO : On skip les dups dans la recherche mais du coup on pourra jamais concater les labels. Faut rajouter l'ajout du label
# a la musique qui est trouvé 2 fois.

try:
    from src.config import fetch, audio

    OUTPUT_CSV = fetch.output_soundcloud_tracks
    MAX_TRACKS_PER_GENRE = fetch.tracks_per_subgenre
    MAX_DURATION_SECONDS = audio.max_track_duration
    GENRES = audio.subgenres
    SOUNDCLOUD_BASE_URL = "https://api-v2.soundcloud.com"
except ImportError:
    OUTPUT_CSV = "soundcloud_tracks.csv"
    MAX_TRACKS_PER_GENRE = 200
    MAX_DURATION_SECONDS = 600
    GENRES = ["Tech House", "Hard Techno", "Minimal Techno", "Melodic Techno", "Trance"]
    SOUNDCLOUD_BASE_URL = "https://api-v2.soundcloud.com"

load_dotenv()
CLIENT_ID = os.getenv("SOUNDCLOUD_CLIENT_ID")
AUTH_TOKEN = os.getenv("SOUNDCLOUD_AUTH_TOKEN")

PLAYLISTS_BY_GENRE = {
    "Tech House": [
        "https://soundcloud.com/housemusic-district/sets/tech-house-2025",
        "https://soundcloud.com/laurosanchezzz/sets/techno-house"
    ],
    "Hard Techno": ["https://soundcloud.com/revisedrecords/sets/hard-techno-essentials"],
    "Minimal Techno": [
        "https://soundcloud.com/matthias-welte-157726484/sets/minimal-techno-classics",
        "https://soundcloud.com/eoinheaney123-5/sets/minimal-techno",
        "https://soundcloud.com/dj-ray-c/sets/minimal-dark-techno-2024"
    ],
    "Melodic Techno": [
        "https://soundcloud.com/melodic_techno/sets/tamborder-aldebaran-betelgeuse",
        "https://soundcloud.com/melodic_techno/sets/malindi-by-adrien-kepler-guava",
        "https://soundcloud.com/elevinsound/sets/melodic-techno"
    ],
    "Trance": ["https://soundcloud.com/user-716720440/sets/90s-2000s-trance-classics"],
}

CUE_WORDS_DJ_SET = [" dj set ", " live set ", " full set ", " guest set ", " mix ", " set ", " podcast ", " radio ",
                    " hour ", " session "]



def looks_like_dj_set(title: str) -> bool:
    if not title: return False
    title_lower = title.lower()
    for cue in CUE_WORDS_DJ_SET:
        if cue in title_lower: return True
    return False


def tag_matches_genre(tags: list, target_genre: str) -> bool:
    if not tags: return False
    target = target_genre.lower()
    for t in tags:
        if not t: continue
        if fuzz.partial_ratio(str(t).lower(), target) >= 80: return True
    return False


def ms_to_duration_string(ms: int) -> str:
    if not ms: return "0:00"
    seconds = int(ms / 1000)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


async def resolve_url(session, client_id, url):
    try:
        api_url = f"{SOUNDCLOUD_BASE_URL}/resolve?url={url}&client_id={client_id}"
        async with session.get(api_url) as resp:
            if resp.status == 200: return await resp.json()
            print(f"Resolve failed: HTTP {resp.status}")
            return None
    except Exception as e:
        print(f"Resolve error: {e}")
        return None



async def process_track_object(track_data: dict, genre: str, source: str, trust: bool = False) -> dict | None:
    """Traite une track brute : filtres DJ set, durée et tags."""
    title = track_data.get("title")
    if not title: return None

    if looks_like_dj_set(title): return None

    duration_ms = track_data.get("duration", 0)
    if duration_ms > (MAX_DURATION_SECONDS * 1000): return None

    raw_tags = track_data.get("tag_list", "")
    api_genre = track_data.get("genre", "")
    tags_list = []
    if api_genre: tags_list.append(str(api_genre).lower())
    if raw_tags:
        try:
            cleaned = shlex.split(raw_tags)
        except:
            cleaned = raw_tags.replace('"', '').split(" ")
        tags_list.extend([str(t).lower() for t in cleaned if t])

    if not trust and not tag_matches_genre(tags_list, genre):
        return None

    return {
        "title": title,
        "artist": track_data.get("user", {}).get("username", "Unknown"),
        "soundcloud_url": track_data.get("permalink_url"),
        "search_genre": genre,
        "soundcloud_tags": tags_list,
        "duration": ms_to_duration_string(duration_ms),
        "source": source
    }


async def fetch_tracks_from_playlist_api(sc_client: SoundcloudAsyncAPI, playlist_url: str, genre: str,
                                         limit_needed: int, existing_urls: set, trust: bool):
    """Récupère les tracks d'une playlist en ignorant les doublons et en gérant le format hybride."""
    print(f"\nFetching playlist: {playlist_url}")

    resolved = await resolve_url(sc_client.http_session, sc_client.client_id, playlist_url)
    if not resolved or "id" not in resolved: return []

    playlist_id = resolved["id"]
    full_playlist = await sc_client.get_playlist_details(playlist_id)

    if not full_playlist or "tracks" not in full_playlist:
        print("Empty playlist or error.")
        return []

    raw_tracks = full_playlist["tracks"]
    print(f"Scanned {len(raw_tracks)} items in playlist.")

    valid_tracks = []
    ids_to_fetch = []
    skipped_dupes = 0

    for item in raw_tracks:
        if len(valid_tracks) >= limit_needed: break

        if isinstance(item, int):
            ids_to_fetch.append(str(item))

        elif isinstance(item, dict):

            if "permalink_url" in item and "title" in item:
                if item["permalink_url"] in existing_urls:
                    skipped_dupes += 1
                    continue

                processed = await process_track_object(item, genre, "playlist", trust=trust)
                if processed:
                    valid_tracks.append(processed)
                    existing_urls.add(processed["soundcloud_url"])

            elif "id" in item:
                ids_to_fetch.append(str(item["id"]))

    print(f"Found immediately: {len(valid_tracks)} new tracks")
    if skipped_dupes > 0:
        print(f"Skipped {skipped_dupes} already existing tracks")

    if ids_to_fetch and len(valid_tracks) < limit_needed:
        remaining = limit_needed - len(valid_tracks)
        print(f"Batch fetching {len(ids_to_fetch)} IDs to find {remaining} more...")

        def chunk_list(lst, n):
            for i in range(0, len(lst), n): yield lst[i:i + n]

        chunks = list(chunk_list(ids_to_fetch, 50))

        for i, chunk in enumerate(chunks):
            if len(valid_tracks) >= limit_needed: break

            ids_string = ",".join(chunk)
            try:
                batch_url = f"{SOUNDCLOUD_BASE_URL}/tracks"
                params = {"ids": ids_string, "client_id": sc_client.client_id, "app_version": sc_client.app_version}

                async with sc_client.http_session.get(batch_url, params=params, headers=sc_client.headers) as resp:
                    if resp.status == 200:
                        fetched_tracks = await resp.json()
                        for track in fetched_tracks:
                            if len(valid_tracks) >= limit_needed: break

                            if track.get("permalink_url") in existing_urls:
                                continue

                            processed = await process_track_object(track, genre, "playlist", trust=trust)
                            if processed:
                                valid_tracks.append(processed)
                                existing_urls.add(processed["soundcloud_url"])
                                print(f"New: {processed['title'][:40]}")
            except Exception as e:
                print(f"Batch error: {e}")

    return valid_tracks


async def fetch_tracks_search_api(sc_client: SoundcloudAsyncAPI, genre: str, limit: int, existing_urls: set):
    """Search API robuste avec logs d'erreurs."""
    print(f"\nSearching API for '{genre}' (Target: {limit} new)...")

    try:
        url = f"{SOUNDCLOUD_BASE_URL}/search/tracks"
        request_limit = min(limit * 4 + 50, 200)

        params = {
            "q": genre,
            "client_id": sc_client.client_id,
            "limit": str(request_limit),
            "app_version": sc_client.app_version,
            "sort": "popular"
        }

        async with sc_client.http_session.get(url, params=params, headers=sc_client.headers) as response:
            if response.status != 200:
                print(f"API Error: HTTP {response.status} - {response.reason}")
                return []
            data = await response.json()

        collection = data.get("collection", [])
        print(f"API returned {len(collection)} candidates.")

        valid_tracks = []
        skipped_dupes = 0
        skipped_filter = 0

        for track in collection:
            if len(valid_tracks) >= limit: break

            if track.get("permalink_url") in existing_urls:
                skipped_dupes += 1
                continue

            processed = await process_track_object(track, genre, "search", trust=False)

            if processed:
                valid_tracks.append(processed)
                existing_urls.add(processed["soundcloud_url"])
            else:
                skipped_filter += 1

        print(f"Accepted: {len(valid_tracks)}")
        print(f"Skipped: {skipped_dupes} duplicates | {skipped_filter} filtered (tags/duration)")
        return valid_tracks

    except Exception as e:
        print(f"Search execution error: {e}")
        return []



def count_tracks_per_label(csv_path: str):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: return {}
    df = pd.read_csv(csv_path)
    if "search_genres" not in df.columns: return {}
    label_counts = {}
    for genres_str in df["search_genres"]:
        if pd.isna(genres_str): continue
        labels = [label.strip() for label in str(genres_str).split(",")]
        for label in labels:
            if label: label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts


def load_label_distribution(csv_path: str):
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    distribution = {}
    for _, row in df.iterrows(): distribution[row["label"]] = int(row["count"])
    return distribution


def merge_tracks_by_url(all_tracks):
    merged = {}
    for track in all_tracks:
        url = track["soundcloud_url"]
        if url not in merged:
            merged[url] = track.copy()
            merged[url]["search_genres"] = [track["search_genre"]]
        else:
            if track["search_genre"] not in merged[url]["search_genres"]:
                merged[url]["search_genres"].append(track["search_genre"])
    for t in merged.values():
        t["search_genres"] = sorted(set(t["search_genres"]))
    return list(merged.values())



async def run_scraper(use_playlists_first: bool, label_dist_csv: str, trust_playlists: bool):
    print(f"\n{'=' * 60}")
    print(f"SOUNDCLOUD API SCRAPER")
    print(f"{'=' * 60}")

    existing_urls = set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        try:
            df_exist = pd.read_csv(OUTPUT_CSV)
            if "soundcloud_url" in df_exist.columns:
                existing_urls = set(df_exist["soundcloud_url"].dropna().tolist())
            print(f"Loaded {len(existing_urls)} existing URLs (will be ignored).")
        except Exception as e:
            print(f"Error loading existing CSV: {e}")

    target_counts = {genre: MAX_TRACKS_PER_GENRE for genre in GENRES}

    if label_dist_csv:
        current_counts = load_label_distribution(label_dist_csv)
    else:
        current_counts = count_tracks_per_label(OUTPUT_CSV) or {}

    missing_counts = {}
    for label, target in target_counts.items():
        current = current_counts.get(label, 0)
        if current < target:
            missing_counts[label] = target - current

    print(f"Target: {MAX_TRACKS_PER_GENRE} per genre")

    if not missing_counts:
        print("\nAll quotas reached! Exiting.")
        return

    if not CLIENT_ID or not AUTH_TOKEN:
        print("\nERROR: CLIENT_ID and SOUNDCLOUD_AUTH_TOKEN missing in .env")
        return

    async with aiohttp.ClientSession() as http_session:
        sc_client = SoundcloudAsyncAPI(AUTH_TOKEN, CLIENT_ID, http_session)
        try:
            await sc_client.login()
            print(f"Logged in as: ClientID ends in ...{sc_client.client_id[-4:]}")
        except Exception as e:
            print(f"\nLogin failed: {e}")
            return

        all_new_tracks = []

        for genre, needed in missing_counts.items():
            print(f"\n{'=' * 60}")
            print(f"Processing: {genre}")
            print(f"Status: Have {current_counts.get(genre, 0)} | Need {needed} NEW tracks")
            print(f"{'=' * 60}")

            genre_new_tracks = []

            if use_playlists_first and genre in PLAYLISTS_BY_GENRE:
                for pl_url in PLAYLISTS_BY_GENRE[genre]:
                    if len(genre_new_tracks) >= needed: break

                    tracks = await fetch_tracks_from_playlist_api(
                        sc_client, pl_url, genre,
                        needed - len(genre_new_tracks),
                        existing_urls,
                        trust_playlists
                    )
                    genre_new_tracks.extend(tracks)

            if len(genre_new_tracks) < needed:
                remaining = needed - len(genre_new_tracks)
                print(f"\nPlaylist exhausted/insufficient. Switching to Search API...")

                search_tracks = await fetch_tracks_search_api(
                    sc_client, genre, remaining, existing_urls
                )
                genre_new_tracks.extend(search_tracks)

            count = len(genre_new_tracks)
            if count > 0:
                all_new_tracks.extend(genre_new_tracks[:needed])
                print(f"\nAcquired {min(count, needed)} new tracks for {genre}")
            else:
                print(f"\nNo new tracks found for {genre} (Try checking filters or API limits)")

        if not all_new_tracks:
            print(f"\n{'=' * 60}")
            print("No new tracks to save.")
            print(f"{'=' * 60}")
            return

        print(f"\n{'=' * 60}")
        print(f"Saving {len(all_new_tracks)} new tracks to CSV...")

        merged_tracks = merge_tracks_by_url(all_new_tracks)
        new_df = pd.DataFrame(merged_tracks)

        if "search_genres" in new_df.columns:
            new_df["search_genres"] = new_df["search_genres"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x)
        if "soundcloud_tags" in new_df.columns:
            new_df["soundcloud_tags"] = new_df["soundcloud_tags"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
            old_df = pd.read_csv(OUTPUT_CSV)
            final_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        final_df.drop_duplicates(subset=["soundcloud_url"], inplace=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"SUCCESS! Total tracks in file: {len(final_df)}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--playlists-first", action="store_true")
    parser.add_argument("--label-distribution", type=str, default=None)
    parser.add_argument("--trust-playlists", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_scraper(args.playlists_first, args.label_distribution, args.trust_playlists))


if __name__ == "__main__":
    main()