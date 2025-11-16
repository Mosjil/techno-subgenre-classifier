import os
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
from src.config import download


# TODO : Gérer le fait de pouvoir télécharger de nouveaux sous genres sans écraser le dataset existant
# TODO : Ajouter une écriture auto dans le csv toutes les x musiques (update)
# TODO : Faire une boucle de vérification pour checker si tout a bien été téléchargé (avec un fuzz ?)
# TODO : Ne pas télécharger des trop longues musiques (DJ Set en entier)

INPUT_CSV = download.input_csv
OUTPUT_METADATA = download.output_metadata
DOWNLOAD_DIR = download.download_dir
SPOTDL_CMD = download.spotdl_cmd
FFMPEG_PATH = download.ffmpeg_path


def download_tracks(df):
    """
    Download spotify tracks via csv
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        title = str(row["title"])
        artist = str(row["artist"])
        subgenres = str(row["subgenres"])
        url = str(row["spotify_url"])

        filename = f"{artist} - {title}.mp3"
        #filepath = os.path.join(DOWNLOAD_DIR, filename)
        filepath = Path(DOWNLOAD_DIR, filename).as_posix()
        print(f"Downloading {filename}, {filepath}")

        # Exists
        if os.path.exists(filepath):
            print("Already downloaded")
            metadata.append({
                "path": filepath,
                "title": title,
                "artist": artist,
                "subgenres": subgenres
            })
            continue

        try:
            # SpotDL
            cmd = [
                SPOTDL_CMD,
                "download", url,
                "--ffmpeg", FFMPEG_PATH,
                "--format", "mp3",
                "--output", DOWNLOAD_DIR,
                "--overwrite", "skip"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            files = [f for f in os.listdir(DOWNLOAD_DIR) if artist.lower() in f.lower() and title.lower() in f.lower()]
            if files:
                filepath = os.path.join(DOWNLOAD_DIR, files[0])

            metadata.append({
                "path": filepath,
                "title": title,
                "artist": artist,
                "subgenres": subgenres
            })
        except Exception as e:
            print(f"Error downloading {title} - {artist}: {e}")

    return metadata


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} Spotify tracks from {INPUT_CSV}")

    metadata = download_tracks(df)

    df_meta = pd.DataFrame(metadata)
    df_meta["path"] = df_meta["path"].apply(lambda p: p.replace("\\", "/") if isinstance(p, str) else p)
    df_meta.to_csv(OUTPUT_METADATA, index=False)
    print(f"\nSaved metadata for {len(df_meta)} tracks to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
