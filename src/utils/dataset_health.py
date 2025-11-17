import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import seaborn as sns
import logging
from datetime import datetime

from src.utils.utils import setup_logger
from src.config import sanity_check

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded CSV with {len(df)} rows")
    return df


def basic_checks(df: pd.DataFrame):
    """Simple sanity checks."""
    print("\n*** Basic Sanity Checks ***")

    print(f"- Rows: {len(df)}")
    print(f"- Columns: {df.columns.tolist()}")

    # NaN count
    nan_counts = df.isna().sum()
    print(f"\nNaN per column:\n{nan_counts.to_string()}")

    # Duplicate rows
    dup_rows = df.duplicated().sum()
    print(f"\n- Duplicate rows: {dup_rows}")

    return {
        "rows": len(df),
        "nan_counts": nan_counts.to_dict(),
        "duplicate_rows": int(dup_rows),
    }

def explode_atomic_labels(series):
    """
    Transforme une colonne subgenres (string) en une liste de sous-genres atomiques.
    Gère :
        - "Tech House, Minimal Techno"
        - "Tech House + Minimal Techno"
        - "Tech House | Minimal Techno"
        - "Tech House;Minimal Techno"
    """
    out = []

    for s in series.astype(str):
        # Remove quotes
        s = s.strip().strip('"').strip("'")

        # Split on multiple separators
        parts = re.split(r"[+,;|]", s)

        # Cleanup
        labels = [p.strip() for p in parts if p.strip() != ""]
        out.append(labels)

    return out

def label_distribution_atomic(df, label_col="subgenres"):
    """
    Retourne un tableau label | count basé sur les sous-genres atomiques.
    """
    exploded = explode_atomic_labels(df[label_col])

    # flatten
    flat = []
    for lst in exploded:
        flat.extend(lst)

    # count
    counts = (
        pd.Series(flat)
        .value_counts()
        .reset_index()
    )
    counts.columns = ["label", "count"]

    return counts


def imbalance_metrics(counts: pd.DataFrame):
    """Compute max/min ratio + entropy + simple metrics."""
    c = counts["count"].astype(float)
    total = c.sum()
    k = len(c)

    if total == 0 or k == 0:
        return {}

    p = c / total

    majority = c.max()
    minority = c.min()

    ir = majority / minority if minority > 0 else np.inf
    entropy_bits = float(-(p * np.log2(p + 1e-12)).sum())
    entropy_max = np.log2(k)
    entropy_ratio = entropy_bits / entropy_max if entropy_max > 0 else 1.0

    print("\n*** Imbalance Metrics ***")
    print(f"- classes: {k}")
    print(f"- majority count: {majority}")
    print(f"- minority count: {minority}")
    print(f"- IR (max/min): {ir:.2f}")
    print(f"- entropy bits: {entropy_bits:.3f} / max={entropy_max:.3f}")
    print(f"- entropy ratio: {entropy_ratio:.3f}")

    return {
        "classes": k,
        "majority": float(majority),
        "minority": float(minority),
        "IR_max_min": float(ir),
        "entropy_bits": float(entropy_bits),
        "entropy_ratio": float(entropy_ratio),
    }


def plot_label_distribution(dist_df, outpath=None, top=None):

    df = dist_df.copy()

    df = df.sort_values("count", ascending=False)

    if top is not None:
        df = df.head(top)

    plt.figure(figsize=(10, 6))
    plt.bar(df["label"], df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Label distribution (atomic subgenres)")
    plt.tight_layout()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150)
        plt.close()
        logging.info(f"[saved] {outpath}")
    else:
        plt.show()


def segments_per_track(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count segments per (artist, title).
    Works for the POST-segmentation CSV.
    """
    per_track = (
        df.groupby(["artist", "title"])
        .size()
        .rename("segments_per_track")
        .reset_index()
        .sort_values("segments_per_track", ascending=False)
    )

    logging.info("\n=== Segments per Track (Top 10) ===")
    logging.info(per_track.head(10).to_string(index=False))

    return per_track


#TODO :

from pathlib import Path

def check_files_exist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie l'existence des fichiers audio et spectrogrammes.
    Retourne un DataFrame :
        path_audio | audio_exists | path_spec | spec_exists
    """

    # Vérification simple
    def exists(p):
        try:
            return Path(p).exists()
        except Exception:
            return False

    out = pd.DataFrame()
    out["path_audio"] = df["path_audio"] if "path_audio" in df.columns else None
    out["path_spec"]  = df["path_spec"]  if "path_spec"  in df.columns else None

    # Colonnes d'existence
    if "path_audio" in df.columns:
        out["audio_exists"] = out["path_audio"].apply(exists)
    else:
        out["audio_exists"] = False

    if "path_spec" in df.columns:
        out["spec_exists"] = out["path_spec"].apply(exists)
    else:
        out["spec_exists"] = False

    # Stats globales
    missing_audio = (~out["audio_exists"]).sum()
    missing_spec  = (~out["spec_exists"]).sum()

    logging.info(f"- Missing audio files: {missing_audio}")
    logging.info(f"- Missing spec files : {missing_spec}")

    if missing_audio == 0 and missing_spec == 0:
        logging.info("All files exist.")
    else:
        logging.info("Some files are missing. Check the output CSV for details.")

    return out


def class_cooccurrence(df: pd.DataFrame, label_col="subgenres") -> pd.DataFrame:

    atomic_lists = explode_atomic_labels(df[label_col])

    # Toutes les classes
    labels = sorted(set(label for lst in atomic_lists for label in lst))

    # Init matrice
    mat = pd.DataFrame(0, index=labels, columns=labels)

    # Remplir matrice
    for lst in atomic_lists:
        unique = set(lst)
        for a in unique:
            for b in unique:
                mat.loc[a, b] += 1

    return mat

def plot_cooccurrence_matrix(mat: pd.DataFrame, outpath=None):

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        mat,
        annot=True,
        cmap="Blues",
        square=True,
        cbar=True,
        linewidths=0.5,
        linecolor="white"
    )

    plt.title("Label Co-occurrence Matrix")
    plt.tight_layout()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[saved] {outpath}")
    else:
        plt.show()


def preview_samples(df: pd.DataFrame):
    """TODO: show some spectrograms or audios."""
    print("\nSample preview not implemented yet.")
    return None


def save_table(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser(description="Dataset Health Checker (v2)")
    ap.add_argument("--csv", required=True, help="Path to CSV (metadata or processed).")
    ap.add_argument("--outdir", default=sanity_check.sanity_output_dir,
                    help="Directory to save outputs.")
    args = ap.parse_args()

    outdir = args.outdir

    # INITIALIZE LOGGER
    setup_logger(outdir=outdir, filename="health")
    logging.info("=== DATASET HEALTH CHECK STARTED ===")

    # LOAD CSV
    df = read_csv(args.csv)
    logging.info(f"Loaded CSV with {len(df)} rows")
    logging.info(df.head().to_string(index=False))

    # BASIC CHECKS
    logging.info("\n=== BASIC SANITY CHECKS ===")
    basic_info = basic_checks(df)
    logging.info(str(basic_info))

    # ATOMIC LABEL DISTRIBUTION
    logging.info("\n=== ATOMIC LABEL DISTRIBUTION ===")
    dist = label_distribution_atomic(df, label_col="subgenres")
    logging.info("\n" + dist.to_string(index=False))
    save_table(dist, f"{outdir}/label_distribution_atomic.csv")
    plot_label_distribution(dist, outpath=f"{outdir}/label_distribution.png")

    # IMBALANCE METRICS
    logging.info("\n=== IMBALANCE METRICS ===")
    imb = imbalance_metrics(dist)
    logging.info(str(imb))

    pd.DataFrame([imb]).to_csv(f"{outdir}/imbalance_metrics.csv", index=False)

    # SEGMENTS PER TRACK (POST-SEG CSV ONLY)
    if {"path_spec", "path_audio", "segment_id"}.issubset(df.columns):
        logging.info("\nDetected segmented CSV → Computing segments per track...")
        per_track = segments_per_track(df)
        logging.info("\n" + per_track.head().to_string(index=False))
        save_table(per_track, f"{outdir}/segments_per_track.csv")
    else:
        logging.info("\nPre-segmentation CSV detected → skipping segments per track.")

    # EXISTING FILES CHECK
    logging.info("\n*** FILE EXISTENCE CHECK ***")
    exists_df = check_files_exist(df)
    save_table(exists_df, f"{outdir}/file_existence.csv")

    #COOCCURRENCE MATRIX
    logging.info("\n*** COOCCURRENCE MATRIX CALC ***")
    cooc = class_cooccurrence(df, label_col="subgenres")
    save_table(cooc, f"{outdir}/cooccurrence_matrix.csv")
    plot_cooccurrence_matrix(cooc, outpath=f"{outdir}/cooccurrence_matrix.png")

    logging.info("\n*** HEALTH CHECK COMPLETED ***")
    logging.info(f"All results saved in: {outdir}")



if __name__ == "__main__":
    main()
