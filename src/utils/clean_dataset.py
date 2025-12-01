import logging
import os
import pandas as pd
import librosa
import argparse
import glob

from src.utils.utils import setup_logger
from src.config import audio, download, sanity_check, fetch

MAX_DURATION = audio.max_track_duration
INPUT_CSV = download.output_metadata_csv
OUTPUT_CSV = INPUT_CSV
LOG_FILE_DIR = sanity_check.sanity_output_dir
RAW_AUDIO_DIR = download.download_dir


def clean_dataset(input_csv, output_csv, max_duration):
    """
    1. Supprime les doublons dans le CSV.
    2. Supprime physiquement et du CSV les fichiers trop longs.
    3. Supprime du CSV les fichiers introuvables.
    """
    if not os.path.exists(input_csv):
        logging.error(f"Input CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    initial_count = len(df)
    logging.info(f"Loaded {initial_count} entries from {input_csv}")

    subset_col = "path"

    if subset_col in df.columns:
        df.drop_duplicates(subset=[subset_col], inplace=True)
        dedup_count = len(df)
        duplicates_removed = initial_count - dedup_count

        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} duplicate entries based on '{subset_col}'.")
    else:
        # Fallback
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} exact duplicate rows.")


    kept_rows = []
    removed_duration_count = 0
    missing_files_count = 0

    for idx, row in df.iterrows():

        audio_path = os.path.normpath(row["path"])

        if not os.path.exists(audio_path):
            logging.warning(f"File missing on disk (removing from CSV) → {audio_path}")
            missing_files_count += 1
            continue

        try:

            duration = librosa.get_duration(path=audio_path)
        except Exception as e:
            logging.error(f"[ERROR] Could not read {audio_path} → {e}")

            continue

        if duration > max_duration:

            try:
                os.remove(audio_path)
                logging.info(f"[DELETE] Too long ({duration:.2f}s > {max_duration}s) → {audio_path}")
            except Exception as e:
                logging.error(f"[ERROR] Failed to delete {audio_path} → {e}")
            removed_duration_count += 1
        else:

            row["path"] = audio_path
            kept_rows.append(row)

    df_clean = pd.DataFrame(kept_rows)
    df_clean.to_csv(output_csv, index=False)

    logging.info("\n--- CLEANING SUMMARY ---")
    logging.info(f"Initial entries      : {initial_count}")
    logging.info(f"Duplicates removed   : {duplicates_removed}")
    logging.info(f"Files missing/error  : {missing_files_count}")
    logging.info(f"Files too long (rm)  : {removed_duration_count}")
    logging.info(f"Final dataset size   : {len(df_clean)}")
    logging.info(f"Saved cleaned CSV    : {output_csv}")


def update_raw_folder(csv_path):
    """Supprime les fichiers du dossier RAW qui ne sont plus dans le CSV."""
    logging.info("\n--- UPDATING RAW FOLDER ---")

    if not os.path.exists(csv_path):
        logging.error("CSV not found, cannot synchronize folder.")
        return

    df = pd.read_csv(csv_path)
    valid_paths = set(os.path.normpath(p) for p in df["path"].dropna().tolist())

    logging.info(f"Valid files referenced in CSV: {len(valid_paths)}")

    extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a']
    physical_files = []

    for ext in extensions:
        for root, dirs, files in os.walk(RAW_AUDIO_DIR):
            for file in files:
                if file.endswith(tuple([e.replace('*', '') for e in extensions])):
                    full_path = os.path.join(root, file)
                    physical_files.append(os.path.normpath(full_path))

    logging.info(f"Physical files found in {RAW_AUDIO_DIR}: {len(physical_files)}")

    deleted_count = 0
    for p_file in physical_files:
        if p_file not in valid_paths:
            try:
                os.remove(p_file)
                logging.info(f"[PRUNE] Orphan file removed → {p_file}")
                deleted_count += 1
            except Exception as e:
                logging.error(f"[ERROR] Could not remove orphan {p_file} → {e}")

    logging.info(f"Total orphan files removed: {deleted_count}")


if __name__ == "__main__":
    # Setup Arguments
    parser = argparse.ArgumentParser(description="Clean dataset and optionally prune raw folder.")
    parser.add_argument(
        "--update-raw-folder",
        action="store_true",
        help="If set, deletes files in raw folder that are not in the CSV."
    )
    args = parser.parse_args()

    setup_logger(LOG_FILE_DIR, "clean_dataset")

    clean_dataset(INPUT_CSV, OUTPUT_CSV, MAX_DURATION)
    if args.update_raw_folder:
        update_raw_folder(OUTPUT_CSV)