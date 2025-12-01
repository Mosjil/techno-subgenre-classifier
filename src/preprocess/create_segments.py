import os
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
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

def split_audio(y, sr, segment_duration, overlap=0):
    seglen = int(segment_duration * sr)
    step = int(seglen * (1 - overlap))
    segments = []
    for start in range(0, len(y) - seglen + 1, step):
        end = start + seglen
        segments.append(y[start:end])
    if not segments:
        segments = [np.pad(y, (0, max(0, seglen - len(y))), "constant")]
    return segments

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
                y = librosa.resample(y, orig_sr=sr_in, target_sr=SR_TARGET)
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

        num_segments = min(MAX_SEGMENTS, int((end_time - start_time) // SEGMENT_DURATION))
        if num_segments == 0:
            print(f"{audio_path}: shorter than one segment, skipped.")
            continue

        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Découpage effectif dans la fenêtre utile (padding inclus)
        effective_y = y[int(start_time * sr):int(end_time * sr)]
        segments = split_audio(effective_y, sr, SEGMENT_DURATION, overlap=0)[:num_segments]

        # Skip global si tout existe déjà
        if segments_already_exist(base_name, num_segments):
            print(f"Skipping {base_name} — all {num_segments} segments already exist.")
        else:
            print(f"Processing {base_name} — creating missing segments if needed.")

        for seg_id, seg_y in enumerate(segments):
            seg_filename = f"{base_name}_seg{seg_id}.wav"
            seg_path = os.path.join(SEGMENT_DIR, seg_filename)

            # Crée juste le segment s'il n'existe pas déjà
            if not os.path.exists(seg_path):
                sf.write(seg_path, seg_y, sr)

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

    df_out = pd.DataFrame(processed_rows)
    df_out["path_audio"] = df_out["path_audio"].apply(lambda p: p.replace("\\", "/") if isinstance(p, str) else p)
    df_out.to_csv(PROCESSED_CSV, index=False)

    print(f"\nDone! {len(df_out)} segments saved at {SR_TARGET} Hz in '{SEGMENT_DIR}'")
    print(f"Processed CSV saved to '{PROCESSED_CSV}'")


if __name__ == "__main__":
    create_segments()