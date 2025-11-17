import argparse
import torch
import librosa
import numpy as np
from pathlib import Path
from src.models.cnn_bigru import ParallelCNNBiGRU
from src.config import audio

# TODO : Utiliser les mêmes fonction que dans le preprocessing. Ici on génère le spect avant de le couper. Alors que dans le
# preprocessing on coupe l'audio avant de générer les specs.

def extract_mel_tensor(y, sr, n_fft=audio.n_fft, hop_length=audio.hop_length, n_mels=audio.n_mels):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=audio.fmin, fmax=sr//2, power=audio.power
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
    return torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def split_audio(y, sr, segment_duration=audio.segment_duration, overlap=0.5):
    seg_len = int(segment_duration * sr)
    step = int(seg_len * (1 - overlap))
    segments = []
    for start in range(0, len(y) - seg_len + 1, step):
        end = start + seg_len
        segments.append(y[start:end])
    # cas où la piste est plus courte
    if not segments:
        segments = [np.pad(y, (0, max(0, seg_len - len(y))))[:seg_len]]
    return segments

def load_model(checkpoint_path, device):
    model = ParallelCNNBiGRU(num_classes=len(audio.subgenres))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def predict_audio(path_audio, model, device, sr=audio.sample_rate_target, threshold=None):
    y, _ = librosa.load(path_audio, sr=sr, mono=True)
    segments = split_audio(y, sr, segment_duration=audio.segment_duration, overlap=0.5)

    all_probs = []
    for seg in segments:
        x = extract_mel_tensor(seg, sr).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.append(probs)

    mean_probs = np.mean(all_probs, axis=0)
    if threshold is not None:
        active = mean_probs >= threshold
    else:
        active = mean_probs >= 0.5

    return mean_probs, active


def display_results(path, mean_probs):
    probs = mean_probs.copy()
    probs = np.maximum(probs, 0)  # clamp
    total = probs.sum()
    if total > 0:
        norm_probs = probs / total
    else:
        norm_probs = probs

    print(f"\nAudio : {Path(path).name}")
    print("Probabilités par sous-genre :")
    for label, p, n in sorted(zip(audio.subgenres, probs, norm_probs), key=lambda x: -x[1]):
        print(f"  {label:<20} {p*100:5.1f}%  (normalisé {n*100:5.1f}%)")

    top = sorted(zip(audio.subgenres, probs), key=lambda x: -x[1])[:3]
    mix = [f"{lbl} {p*100:.1f}%" for lbl, p in top]
    print("\nMix dominant :", ", ".join(mix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-label inference for techno subgenres")
    parser.add_argument("audio_path", type=str, help="Chemin vers le fichier audio (.mp3/.wav)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--threshold", type=float, default=None, help="Seuil pour activer les classes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    mean_probs, _ = predict_audio(args.audio_path, model, device, threshold=args.threshold)
    display_results(args.audio_path, mean_probs)
