#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Health Checker for audio segment/spec CSV

Expected columns:
- path_audio
- path_spec
- artist
- title
- segment_id
- subgenres   (single label OR multi-label with '|' or ';' separator)

Examples:
    python dataset_health.py --csv data/metadata.csv
    python dataset_health.py --csv data/metadata.csv --check-files --plots --outdir output/stats
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv(csv_path: str) -> pd.DataFrame:
    """Read the dataset CSV with robust defaults for quoted fields and commas in text."""
    df = pd.read_csv(csv_path)
    expected = {"path_audio", "path_spec", "artist", "title", "segment_id", "subgenres"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
    return df


def detect_subgenre_sep(series: pd.Series) -> Optional[str]:
    """Detect a likely multi-label separator ('|' or ';'), else None."""
    sample = series.dropna().astype(str).head(200).tolist()
    counts = {"|": 0, ";": 0}
    for s in sample:
        counts["|"] += ("|" in s)
        counts[";"] += (";" in s)
    if counts["|"] > counts[";"] and counts["|"] > 0:
        return "|"
    if counts[";"] > counts["|"] and counts[";"] > 0:
        return ";"
    return None  # assume single-label


def explode_subgenres(df: pd.DataFrame, sep: Optional[str]) -> pd.DataFrame:
    """
    Explode multi-labels: 1 segment compte pour 1 dans CHAQUE sous-genre.
    En plus, scinde les combos 'A + B' en ['A','B'] et ne garde jamais
    une étiquette contenant un '+' telle quelle.
    """
    def split_clean(cell: str) -> List[str]:
        if not isinstance(cell, str) or cell.strip() == "":
            return []

        s = cell.strip()
        # 1) normaliser les séparateurs principaux si besoin (auto/none -> '|')
        main_sep = sep if sep in ("|", ";") else "|"

        # 2) découper une première fois sur le séparateur principal s’il existe
        parts = [p.strip() for p in (s.split(sep) if sep else [s])]

        # 3) pour chaque part, découper aussi les combos 'A + B' (on peut étendre plus tard)
        expanded = []
        for p in parts:
            # coupe sur '+' entouré d'espaces (tolérant)
            subparts = [x.strip() for x in re.split(r"\s*\+\s*", p) if x.strip() != ""]
            expanded.extend(subparts)

        # 4) retirer tout token résiduel contenant '+' (au cas où)
        expanded = [x for x in expanded if "+" not in x]

        # 5) dédup insensible à la casse en préservant l’ordre et l’orthographe d’origine
        seen, uniq = set(), []
        for it in expanded:
            key = it.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(it)
        return uniq

    df_ex = df.copy()
    df_ex["__subgenre_list"] = df_ex["subgenres"].fillna("").astype(str).apply(split_clean)
    df_ex = df_ex.explode("__subgenre_list", ignore_index=True)
    df_ex = df_ex.rename(columns={"__subgenre_list": "subgenre"})
    df_ex["subgenre"] = df_ex["subgenre"].fillna("").astype(str).str.strip()
    df_ex = df_ex[df_ex["subgenre"] != ""]
    return df_ex


def basic_quality_checks(df: pd.DataFrame) -> Dict[str, any]:
    """Compute basic sanity checks (nulls, duplicates, types)."""
    report = {}
    report["n_rows"] = len(df)
    report["n_null_by_col"] = df.isna().sum().to_dict()
    # Duplicats potentiels
    report["n_dup_rows"] = int(df.duplicated().sum())
    report["n_dup_path_audio"] = int(df["path_audio"].duplicated().sum())
    report["n_dup_path_spec"] = int(df["path_spec"].duplicated().sum())
    # Types
    try:
        seg = pd.to_numeric(df["segment_id"], errors="coerce")
        report["segment_id_non_numeric"] = int(seg.isna().sum())
    except Exception:
        report["segment_id_non_numeric"] = "error"
    return report


def class_distribution(df_exploded: pd.DataFrame) -> pd.DataFrame:
    """Counts per subgenre (multi-label aware)."""
    counts = (
        df_exploded.groupby("subgenre", dropna=False)
        .size()
        .sort_values(ascending=False)
        .rename("count")
        .to_frame()
    )
    return counts


def imbalance_metrics(counts: pd.Series) -> Dict[str, float]:
    """Compute a few imbalance indicators."""
    c = counts[counts > 0].astype(float)
    total = c.sum()
    k = len(c)
    if k == 0 or total == 0:
        return {"classes": float(k), "majority": 0.0, "minority": 0.0, "IR_max_min": np.nan,
                "gini_simpson": np.nan, "entropy_bits": np.nan}
    p = c / total
    majority = float(c.max())
    minority = float(c.min())
    ir = float(majority / minority) if minority > 0 else np.inf
    gini_simpson = float(1.0 - np.sum(p**2))  # 0 balanced? (ranges 0..1-1/k); higher = more diverse
    entropy_bits = float(-(p * np.log2(p + 1e-12)).sum())  # max = log2(k) if perfectly balanced
    return {
        "classes": float(k),
        "majority": majority,
        "minority": minority,
        "IR_max_min": ir,
        "gini_simpson": gini_simpson,
        "entropy_bits": entropy_bits,
        "entropy_bits_max": float(np.log2(k)),
        "entropy_balance_ratio": float(entropy_bits / np.log2(k)) if k > 1 else 1.0,
    }


def segments_per_track(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of segment counts per (artist,title) track."""
    per_track = (
        df.groupby(["artist", "title"])
        .size()
        .rename("segments_per_track")
        .reset_index()
        .sort_values("segments_per_track", ascending=False)
    )
    return per_track


def segments_per_artist(df: pd.DataFrame) -> pd.DataFrame:
    """Total segments per artist (before exploding subgenres)."""
    per_artist = (
        df.groupby("artist")
        .size()
        .rename("segments")
        .sort_values(ascending=False)
        .to_frame()
    )
    return per_artist


def crosstab_artist_subgenre(df_exploded: pd.DataFrame, top_n_artists: int = 20) -> pd.DataFrame:
    """Pivot of top artists × subgenres."""
    top_artists = (
        df_exploded.groupby("artist").size().sort_values(ascending=False).head(top_n_artists).index
    )
    slim = df_exploded[df_exploded["artist"].isin(top_artists)]
    tab = pd.crosstab(slim["artist"], slim["subgenre"]).sort_index()
    return tab


def check_files_exist(df: pd.DataFrame) -> pd.DataFrame:
    """Check whether paths in path_audio and path_spec exist on disk."""
    def exists(p):
        try:
            return Path(p).exists()
        except Exception:
            return False

    out = df[["path_audio", "path_spec"]].copy()
    out["audio_exists"] = out["path_audio"].apply(exists)
    out["spec_exists"] = out["path_spec"].apply(exists)
    return out


def ensure_outdir(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: str, index: bool = True):
    ensure_outdir(Path(path).parent.as_posix())
    df.to_csv(path, index=index, encoding="utf-8")
    print(f"[saved] {path}")


def plot_bar(counts: pd.DataFrame, title: str, outpath: Optional[str] = None, top: int = 30):
    """Simple bar plot (top-N)."""
    head = counts.head(top)
    plt.figure(figsize=(10, 6))
    head["count"].plot(kind="bar")
    plt.title(title)
    plt.xlabel("subgenre")
    plt.ylabel("segments (multi-label)")
    plt.tight_layout()
    if outpath:
        ensure_outdir(Path(outpath).parent.as_posix())
        plt.savefig(outpath, dpi=150)
        print(f"[saved] {outpath}")
        plt.close()
    else:
        plt.show()


def describe_series(x: pd.Series) -> pd.DataFrame:
    """Return a compact describe for a numeric series."""
    d = x.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame(name="value")
    return d


def main():
    ap = argparse.ArgumentParser(description="Dataset health checker (audio/spec CSV).")
    ap.add_argument("--csv", required=True, help="Path to your dataset CSV.")
    ap.add_argument("--subgenre-sep", default="auto",
                    help="Multi-label separator for subgenres: 'auto' (default), '|', ';', or 'none'.")
    ap.add_argument("--check-files", action="store_true", help="Verify that path_audio/path_spec exist on disk.")
    ap.add_argument("--plots", action="store_true", help="Export a bar plot of class distribution.")
    ap.add_argument("--outdir", default="output/stats", help="Directory to save tables/plots.")
    args = ap.parse_args()

    df = read_csv(args.csv)
    print("\n=== Aperçu CSV ===")
    print(df.head(5).to_string(index=False))

    # Quality checks
    qc = basic_quality_checks(df)
    print("\n=== Checks de base ===")
    for k, v in qc.items():
        print(f"- {k}: {v}")

    # Detect / set subgenre separator
    if args.subgenre_sep == "auto":
        sep = detect_subgenre_sep(df["subgenres"])
    elif args.subgenre_sep.lower() in ("|", ";"):
        sep = args.subgenre_sep
    elif args.subgenre_sep.lower() == "none":
        sep = None
    else:
        raise ValueError("--subgenre-sep doit être 'auto', '|', ';' ou 'none'.")

    print(f"\nSéparateur de sous-genres: {sep if sep else '[aucun – mono-label]'}")

    # Explode multi-labels for class stats
    df_ex = explode_subgenres(df, sep)

    # Class distribution (multi-label aware)
    dist = class_distribution(df_ex)
    imb = imbalance_metrics(dist["count"])
    print("\n=== Répartition des sous-genres (multi-label) ===")
    print(dist.head(30).to_string())

    print("\n=== Indicateurs d'imbalance ===")
    for k, v in imb.items():
        print(f"- {k}: {v}")

    # Segments per track / artist
    per_track = segments_per_track(df)
    per_artist = segments_per_artist(df)
    print("\n=== Segments par morceau (top 10) ===")
    print(per_track.head(10).to_string(index=False))

    print("\n=== Segments par artiste (top 10) ===")
    print(per_artist.head(10).to_string())

    # Numeric distribution of segments per track
    print("\n=== Stats segments par morceau ===")
    sp = describe_series(per_track["segments_per_track"])
    print(sp.to_string())

    # Crosstab artist x subgenre
    xtab = crosstab_artist_subgenre(df_exploded=df_ex, top_n_artists=20)
    print("\n=== Artist × Subgenre (top artistes) ===")
    print(xtab.head(20).to_string())

    # Optional file existence check
    if args.check_files:
        exists_df = check_files_exist(df)
        n_audio_missing = int((~exists_df["audio_exists"]).sum())
        n_spec_missing = int((~exists_df["spec_exists"]).sum())
        print("\n=== Présence des fichiers ===")
        print(f"- audio manquants: {n_audio_missing}")
        print(f"- specs manquants: {n_spec_missing}")
        save_table(exists_df, Path(args.outdir, "files_existence.csv").as_posix(), index=False)

    # Save tables
    save_table(dist, Path(args.outdir, "class_distribution.csv").as_posix())
    save_table(per_track, Path(args.outdir, "segments_per_track.csv").as_posix(), index=False)
    save_table(per_artist, Path(args.outdir, "segments_per_artist.csv").as_posix())
    save_table(xtab, Path(args.outdir, "artist_x_subgenre.csv").as_posix())

    # Optional plot
    if args.plots:
        plot_bar(dist, title="Subgenre distribution (multi-label)", outpath=Path(args.outdir, "subgenre_bar_top30.png").as_posix(), top=30)

    print("\n✅ Terminé.")


if __name__ == "__main__":
    main()
