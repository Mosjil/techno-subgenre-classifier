import os
import pandas as pd
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
from rapidfuzz import fuzz
from dotenv import load_dotenv
from src.config import fetch, audio

# TODO : fuzz score dynamique
# TODO : Problème de duplicate sur les noms (ID différents, albums... mais mm musique)
# TODO : Gérer le fait de pouvoir télécharger de nouveaux sous genres sans écraser le dataset existant

load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
OUTPUT_CSV = fetch.output_spotify_tracks
MAX_MUSIC_DURATION = audio.max_track_duration

SUBGENRES = audio.subgenres

TRACKS_PER_GENRE = fetch.tracks_per_subgenre
SLEEP_BETWEEN_CALLS = fetch.sleep_between_calls

FUZZ_SCORE_RATIO = fetch.fuzz_score_ratio

def setup_spotify():
    """Auth to spotify client"""
    auth_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def fetch_artist_genres(sp, artist_id):
    """Genre for an artist"""
    try:
        artist = sp.artist(artist_id)
        return artist.get("genres", [])
    except Exception:
        return []

def fetch_tracks_for_genre(sp, genre, limit=100):

    results = []
    offset = 0
    total_fetched = 0

    print(f"\nSearching for {genre} tracks...")

    while total_fetched < limit:
        try:
            resp = sp.search(q=f'genre:"{genre}"', type='track', limit=50, offset=offset)
        except Exception as e:
            print(f"API error for genre {genre}: {e}")
            break

        items = resp["tracks"]["items"]
        if not items:
            break

        for track in items:

            artist_id = track["artists"][0]["id"]
            artist_genres = fetch_artist_genres(sp, artist_id)

            # Only subgenres allowed
            artist_genres_possible = []
            for artist_genre in artist_genres:
                for allowed_subgenre in SUBGENRES:
                    score = fuzz.token_sort_ratio(artist_genre.lower(), allowed_subgenre.lower())
                    if score > FUZZ_SCORE_RATIO:
                        artist_genres_possible.append(allowed_subgenre)

            combined_genres = list(set(artist_genres_possible + [genre]))

            track_info = {
                "id": track["id"],
                "title": track["name"],
                "artist": track["artists"][0]["name"],
                "artist_id": track["artists"][0]["id"],
                "spotify_url": track["external_urls"]["spotify"],
                "preview_url": track["preview_url"],
                "album": track["album"]["name"],
                "release_date": track["album"]["release_date"],
                "duration_s": track["duration_ms"]/1000,
                "subgenres": combined_genres
            }
            results.append(track_info)
            total_fetched += 1

            if total_fetched >= limit:
                break

        offset += 50
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"{len(results)} tracks found for {genre}")
    return results

def fetch_tracks_from_playlists(sp, genre, max_playlists=3, max_tracks_per_playlist=200, sleep=0.2):

    print(f"\nSearching playlists for '{genre}'...")
    results = []

    try:
        search = sp.search(q=genre, type="playlist", limit=max_playlists)
        playlists = [
            p for p in search.get("playlists", {}).get("items", [])
            if p and genre.lower() in p.get("name", "").lower()
        ]
    except Exception as e:
        print(f"Playlist search error for '{genre}': {e}")
        return results

    for p in playlists:
        try:
            pname = p.get("name", "Unnamed")
            ptotal = p.get("tracks", {}).get("total", 0)
            pid = p.get("id")
            if pid is None:
                continue
            print(f"   → Playlist: {pname} ({ptotal} tracks)")
        except Exception:
            continue

        fetched_from_this_playlist = 0
        offset = 0
        limit = 100

        while fetched_from_this_playlist < max_tracks_per_playlist:
            try:
                resp = sp.playlist_items(
                    playlist_id=pid,
                    offset=offset,
                    limit=min(limit, max_tracks_per_playlist - fetched_from_this_playlist),
                    additional_types=["track"],  # évite podcasts
                )
            except Exception as e:
                print(f"Error fetching items for '{pname}': {e}")
                break

            items = resp.get("items", [])
            if not items:
                break

            for it in items:
                t = it.get("track")
                if not t or t.get("id") is None or t.get("type") != "track" or t.get("is_local"):
                    continue

                try:
                    results.append({
                        "id": t["id"],
                        "title": t["name"],
                        "artist": (t["artists"][0]["name"] if t.get("artists") else "Unknown"),
                        "artist_id": (t["artists"][0]["id"] if t.get("artists") else None),
                        "spotify_url": t["external_urls"]["spotify"],
                        "preview_url": t.get("preview_url"),
                        "album": t["album"]["name"],
                        "release_date": t["album"].get("release_date"),
                        "duration_s": t["duration_ms"] / 1000,
                        "subgenres": [genre],
                    })
                    fetched_from_this_playlist += 1
                    if fetched_from_this_playlist >= max_tracks_per_playlist:
                        break
                except Exception:
                    continue

            if fetched_from_this_playlist >= max_tracks_per_playlist:
                break

            offset += len(items)
            time.sleep(sleep)

    print(f"{len(results)} tracks collected from playlists for {genre}")
    return results

def merge_tracks_by_query(all_tracks):
    """
    Merge tracks that appear across multiple subgenre queries.
    If a track ID is duplicated, its subgenres are concatenated (multi-label).
    """
    merged = {}

    for track in all_tracks:
        tid = track["id"]
        if tid not in merged:
            merged[tid] = track
        else:
            # DEBUG
            prev_sub = merged[tid]["subgenres"]
            new_sub = track["subgenres"]
            print(f"Duplicate found for {track['title']} by {track['artist']}")
            print(f"    Existing: {prev_sub}")
            print(f"    Adding : {new_sub}\n")
            # END DEBUG
            merged[tid]["subgenres"].extend(track["subgenres"])


    # Remove duplicates and sort
    for t in merged.values():
        t["subgenres"] = sorted(set(t["subgenres"]))

    print(f"\nMerged {len(all_tracks)} raw entries into {len(merged)} unique tracks.")
    return list(merged.values())

def main():
    sp = setup_spotify()
    all_tracks = []

    for genre in SUBGENRES:
        genre_tracks = fetch_tracks_for_genre(sp, genre, TRACKS_PER_GENRE)

        break_while = 0
        while len(genre_tracks) < TRACKS_PER_GENRE:
            if break_while >= 5:
                break
            missing = TRACKS_PER_GENRE - len(genre_tracks)
            print(f"Only {len(genre_tracks)} results for {genre}, fetching from playlists...")
            playlist_tracks = fetch_tracks_from_playlists(
                sp, genre, max_playlists=10, max_tracks_per_playlist=200
            )
            seen_ids = {t["id"] for t in genre_tracks}
            for pt in playlist_tracks:
                if pt["id"] not in seen_ids:
                    genre_tracks.append(pt)
                    seen_ids.add(pt["id"])
                    if len(genre_tracks) >= TRACKS_PER_GENRE:
                        break
            break_while += 1

        all_tracks.extend(genre_tracks)

    merged_tracks = merge_tracks_by_query(all_tracks)

    # Save CSV
    df = pd.DataFrame(merged_tracks)
    df.drop_duplicates(subset=["id"], inplace=True)
    df = df[df["duration_s"] <= MAX_MUSIC_DURATION]
    df["subgenres"] = df["subgenres"].apply(lambda lst: ", ".join(lst))
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} tracks to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
