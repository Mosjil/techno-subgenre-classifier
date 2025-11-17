import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from src.config import preprocess, audio

PROCESSED_CSV = preprocess.preprocessed_csv
SPECS_DIR = preprocess.spectrogram_dir

SR = audio.sample_rate_target
N_FFT = audio.n_fft
HOP_LENGTH = audio.hop_length
N_MELS = audio.n_mels
FMIN = audio.fmin
POWER = audio.power

os.makedirs(SPECS_DIR, exist_ok=True)

def generate_mel_spectrogram(audio_path, sr=SR):
    """Load an audio file and generate mel-spectrogram"""
    y, _ = librosa.load(audio_path, sr=sr)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=sr//2, power=POWER
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    S_norm = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)

    return S_norm


def generate_all_specs(plots=False):
    df = pd.read_csv(PROCESSED_CSV)
    rows_ok = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating mel spectrograms"):
        audio_path = row["path_audio"]
        spec_path = row["path_spec"]

        os.makedirs(os.path.dirname(spec_path), exist_ok=True)

        # Exists
        if os.path.exists(spec_path):
            rows_ok.append(row)
            continue

        try:
            mel = generate_mel_spectrogram(audio_path)
            np.save(spec_path, mel)
            rows_ok.append(row)

            if plots:
                visualize_mel_spectrogram(mel, save_path=spec_path)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    df_out = pd.DataFrame(rows_ok).drop_duplicates(
        subset=["path_audio", "segment_id"]
    )
    df_out.to_csv(PROCESSED_CSV, index=False)
    df_out["path_spec"] = df_out["path_spec"].apply(lambda p: p.replace("\\", "/") if isinstance(p, str) else p)
    df_out.to_csv(PROCESSED_CSV, index=False)
    print(f"\n{len(rows_ok)} spectrograms saved in '{SPECS_DIR}'")
    print(f"CSV updated: {PROCESSED_CSV}")

import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def visualize_mel_spectrogram(mel_spectrogram, sr=44100, hop_length=512, fmin=20, save_path=None, show=False):

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        fmin=fmin,
        fmax=sr // 2,
        cmap='magma'
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()

    if save_path is not None:
        if save_path.endswith(".npy"):
            img_path = save_path.replace(".npy", ".png")
        else:
            img_path = save_path + ".png"

        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram plot at: {img_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    generate_all_specs(plots=False)
