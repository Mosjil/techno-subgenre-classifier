import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from src.config import download, preprocess, audio

RAW_DIR = download.download_dir
SEGMENT_DIR = preprocess.segments_dir
PROCESSED_CSV = preprocess.preprocessed_csv
METADATA_CSV = download.output_metadata_csv

SEGMENT_DURATION = audio.segment_duration
PADDING = audio.audio_padding   # sec to ignore before after
SR_TARGET = audio.sample_rate_target
MAX_MUSIC_DURATION = audio.max_track_duration
MAX_SEGMENTS = audio.max_segments

os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)

def segments_already_exist(base_name, expected_num):
    """
    Vérifie si tous les segments .wav existent déjà.
    expected_num = nombre de segments attendus
    """
    for seg_id in range(expected_num):
        seg_filename = f"{base_name}_seg{seg_id}.wav"
        seg_path = os.path.join(SEGMENT_DIR, seg_filename)
        if not os.path.exists(seg_path):
            return False
    return True

def create_segments():
    df_meta = pd.read_csv(METADATA_CSV)
    processed_rows = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Processing tracks"):
        audio_path = row["path"]
        artist = row["artist"]
        title = row["title"]
        subgenres = row["subgenres"]

        if not os.path.exists(audio_path):
            print(f"Missing file: {audio_path}")
            continue

        try:
            y, sr_in = librosa.load(audio_path, sr=None)
            if sr_in != SR_TARGET:
                y = librosa.resample(y=y, orig_sr=sr_in, target_sr=SR_TARGET)
            sr = SR_TARGET
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        total_duration = librosa.get_duration(y=y, sr=sr)
        start_time = PADDING
        end_time = total_duration - PADDING

        if end_time <= start_time:
            print(f"{audio_path}: too short (< {2 * PADDING}s), skipped.")
            continue

        expected_segments = min(MAX_SEGMENTS, int((end_time - start_time) // SEGMENT_DURATION))
        if expected_segments == 0:
            print(f"{audio_path}: shorter than one segment, skipped.")
            continue

        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # SKIP SI TOUS LES SEGMENTS EXISTENT DEJA
        if segments_already_exist(base_name, expected_segments):
            print(f"Skipping {base_name} — all {expected_segments} segments already exist.")

            # Mais on doit quand même remplir le CSV !
            for seg_id in range(expected_segments):
                seg_filename = f"{base_name}_seg{seg_id}.wav"
                seg_path = os.path.join(SEGMENT_DIR, seg_filename)

                spec_filename = seg_filename.replace(".wav", ".npy")
                spec_path = os.path.join("data/specs", spec_filename)

                processed_rows.append({
                    "path_audio": seg_path.replace("\\", "/"),
                    "path_spec": spec_path.replace("\\", "/"),
                    "artist": artist,
                    "title": title,
                    "segment_id": seg_id,
                    "subgenres": subgenres
                })

            continue

    df_out = pd.DataFrame(processed_rows)
    df_out["path_audio"] = df_out["path_audio"].apply(lambda p: p.replace("\\", "/") if isinstance(p, str) else p)
    df_out.to_csv(PROCESSED_CSV, index=False)

    print(f"\nDone! {len(df_out)} segments saved at {SR_TARGET} Hz in '{SEGMENT_DIR}'")
    print(f"Processed CSV saved to '{PROCESSED_CSV}'")

if __name__ == "__main__":
    create_segments()