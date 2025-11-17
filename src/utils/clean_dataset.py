import logging
import os
import pandas as pd
import librosa

from src.utils.utils import setup_logger
from src.config import audio, download, sanity_check

MAX_DURATION = audio.max_track_duration
INPUT_CSV = download.output_metadata_csv
OUTPUT_CSV = INPUT_CSV
LOG_FILE_DIR = sanity_check.sanity_output_dir

def clean_dataset(input_csv, output_csv, max_duration):
    df = pd.read_csv(input_csv)

    logging.info(f"Loaded {len(df)} entries from {input_csv}")

    kept_rows = []
    removed_count = 0
    missing_files = 0

    for idx, row in df.iterrows():
        audio_path = row["path"]

        if not os.path.exists(audio_path):
            logging.info(f"File missing → {audio_path}")
            missing_files += 1
            # don't keep row
            continue

        try:
            duration = librosa.get_duration(path=audio_path)
        except Exception as e:
            logging.info(f"[ERROR] Could not read {audio_path} → {e}")
            continue

        if duration > max_duration:
            # delete file
            try:
                os.remove(audio_path)
                logging.info(f"[DELETE] {audio_path} (duration={duration:.2f}s > {max_duration}s)")
            except Exception as e:
                logging.info(f"[ERROR] Failed to delete {audio_path} → {e}")
            removed_count += 1
        else:
            kept_rows.append(row)

    df_clean = pd.DataFrame(kept_rows)
    df_clean.to_csv(output_csv, index=False)

    logging.info("\n--- CLEANING SUMMARY ---")
    logging.info(f"Kept tracks       : {len(df_clean)}")
    logging.info(f"Removed tracks    : {removed_count}")
    logging.info(f"Missing files     : {missing_files}")
    logging.info(f"Saved cleaned CSV : {output_csv}")

if __name__ == "__main__":
    setup_logger(LOG_FILE_DIR, "clean_dataset")
    clean_dataset(INPUT_CSV, OUTPUT_CSV, MAX_DURATION)
