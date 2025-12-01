import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import torch
import soundfile as sf
import matplotlib.pyplot as plt

from src.preprocess.create_segments import split_audio
from src.preprocess.generate_mel_specs import generate_mel_spectrogram, visualize_mel_spectrogram
from src.models.cnn_bigru import ParallelCNNBiGRU
from src.config import audio

def load_model(checkpoint_path, device):
    model = ParallelCNNBiGRU(num_classes=len(audio.subgenres))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def predict_audio(path_audio, model, device, segment_duration, overlap, save_outputs=False, out_dir=None):
    y, sr = librosa.load(path_audio, sr=audio.sample_rate_target, mono=True)
    base_name = Path(path_audio).stem
    out_base = None
    if save_outputs and out_dir is not None:
        out_base = Path(out_dir) / base_name
        os.makedirs(out_base / "segments", exist_ok=True)
        os.makedirs(out_base / "spectrograms", exist_ok=True)
        os.makedirs(out_base / "results", exist_ok=True)

    segments = split_audio(y, sr, segment_duration=segment_duration, overlap=overlap)
    all_probs = []

    for i, seg in enumerate(segments):
        # Sauvegarde segment audio si demandé
        if out_base is not None:
            seg_path = out_base / "segments" / f"{base_name}_seg{i}.wav"
            sf.write(str(seg_path), seg, sr)
        # Génère et sauvegarde mel spectrogramme
        mel = generate_mel_spectrogram(seg, sr)

        if out_base is not None:
            spec_path = out_base / "spectrograms" / f"{base_name}_seg{i}.npy"
            visualize_mel_spectrogram(mel_spectrogram=mel, save_path=spec_path)

        # Préparation pour le modèle
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)

    mean_probs = np.mean(all_probs, axis=0)
    if out_base is not None:
        results_path = out_base / "results" / f"{base_name}_results.csv"
        print(sorted(audio.subgenres))
        df = pd.DataFrame({"subgenre": sorted(audio.subgenres), "proba": mean_probs})
        df.to_csv(results_path, index=False)
    return mean_probs

def display_results(path, mean_probs):
    probs = mean_probs.copy()
    probs = np.maximum(probs, 0)  # clamp
    total = probs.sum()
    if total > 0:
        norm_probs = probs / total
    else:
        norm_probs = probs
    print(f"\nAudio : {Path(path).name}")
    print("Probabilités par sous-genre :", sorted(audio.subgenres))
    for label, p, n in sorted(zip(sorted(audio.subgenres), probs, norm_probs), key=lambda x: -x[1]):
        print(f"  {label:<20} {p*100:5.1f}% (normalisé {n*100:5.1f}%)")
    top = sorted(zip(sorted(audio.subgenres), probs), key=lambda x: -x[1])[:3]
    mix = [f"{lbl} {p*100:.1f}%" for lbl, p in top]
    print("\nMix dominant :", ", ".join(mix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-label inference for techno subgenres")
    parser.add_argument("audio_path", type=str, help="Chemin vers le fichier audio (.mp3/.wav)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--threshold", type=float, default=None, help="Seuil pour activer les classes")
    parser.add_argument("--save-outputs", action="store_true",
                        help="Sauvegarder les segments, spectrogrammes, et plots")
    parser.add_argument("--output-dir", type=str, default="inference",
                        help="Répertoire de sortie (dossier parent)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    mean_probs = predict_audio(
        args.audio_path, model, device,
        segment_duration=audio.segment_duration,
        overlap=0.5,
        save_outputs=args.save_outputs,
        out_dir=args.output_dir if args.save_outputs else None
    )
    display_results(args.audio_path, mean_probs)
