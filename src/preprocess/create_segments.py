import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

RAW_DIR = "data/raw"
SEGMENT_DIR = "data/segments"
PROCESSED_CSV = "data/processed.csv"
METADATA_CSV = "data/metadata.csv"

SEGMENT_DURATION = 30
PADDING = 5             # sec to ignore before after
SR_TARGET = 44100
MAX_MUSIC_DURATION = 10*60 # 10 mins
MAX_SEGMENTS = MAX_MUSIC_DURATION/SEGMENT_DURATION

os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)

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
            # Chargement à SR natif
            y, sr_in = librosa.load(audio_path, sr=None)
            # Resampling vers SR_TARGET si nécessaire
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
            print(f"{audio_path}: too short (< {2*PADDING}s), skipped.")
            continue

        num_segments = min(MAX_SEGMENTS, int((end_time - start_time) // SEGMENT_DURATION))
        if num_segments == 0:
            print(f"{audio_path}: shorter than one segment, skipped.")
            continue

        for seg_id in range(num_segments):
            seg_start = start_time + seg_id * SEGMENT_DURATION
            seg_end = seg_start + SEGMENT_DURATION
            seg_y = y[int(seg_start * sr):int(seg_end * sr)]

            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            seg_filename = f"{base_name}_seg{seg_id}.wav"
            seg_path = os.path.join(SEGMENT_DIR, seg_filename)

            # Save
            sf.write(seg_path, seg_y, sr)

            # Link for Mel_Spec
            spec_filename = seg_filename.replace(".wav", ".npy")
            spec_path = os.path.join("data/specs", spec_filename)

            processed_rows.append({
                "path_audio": seg_path,
                "path_spec": spec_path,
                "artist": artist,
                "title": title,
                "segment_id": seg_id,
                "subgenres": subgenres
            })

    df_out = pd.DataFrame(processed_rows)
    df_out["path_audio"] = df_out["path_audio"].apply(lambda p: p.replace("\\", "/") if isinstance(p, str) else p)
    df_out.to_csv(PROCESSED_CSV, index=False)

    print(f"\nDone! {len(df_out)} segments saved at {SR_TARGET} Hz in '{SEGMENT_DIR}'")
    print(f"Processed CSV saved to '{PROCESSED_CSV}'")

if __name__ == "__main__":
    create_segments()