from dataclasses import dataclass, field
from typing import List

@dataclass
class Fetch:
    output_spotify_tracks: str = "spotify_tracks.csv"
    tracks_per_subgenre: int = 100
    sleep_between_spotify_api_calls: float = 0.5
    fuzz_score_ratio: float = 80

@dataclass
class Download:
    input_csv: str = "spotify_tracks.csv"
    output_metadata_csv: str = "data/metadata.csv"
    download_dir: str = "data/raw"
    spotdl_cmd: str = "spotdl"
    ffmpeg_path: str = r"C:\ffmpeg\bin\ffmpeg.exe"

@dataclass
class Preprocess:
    segments_dir: str = "data/segments"
    preprocessed_csv: str = "data/processed.csv"
    spectrogram_dir: str = "data/specs"

@dataclass
class Audio:
    subgenres: List[str] = field(default_factory=lambda: [
        "Tech House",
        "Hard Techno",
        "Melodic Techno",
        "Minimal Techno",
        "Trance",
    ])
    max_track_duration: int = 15*60 # 15mins
    segment_duration: int = 30
    max_segments = max_track_duration/segment_duration
    audio_padding:int = 5
    sample_rate_target: int = 44100
    n_fft: int = 4096
    hop_length: int = 512
    n_mels: int = 256
    fmin: int = 20
    power: float = 2.0

@dataclass
class SanityCheck:
    sanity_output_dir: str = "outputs/health"

@dataclass
class Train:
    generic_output_dir: str = "outputs/"
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    model: str = "cnn_bigru"
    num_workers: int = 2
    val_split: float = 0.2

fetch = Fetch()
download = Download()
preprocess = Preprocess()
audio = Audio()
sanity_check = SanityCheck()
train_config = Train()