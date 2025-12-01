import pandas as pd
import argparse
from src.config import fetch, download


FETCH_CSV_PATH = fetch.output_soundcloud_tracks
METADATA_CSV_PATH = download.output_metadata_csv
KEY_COLUMNS = ['title', 'artist']


def update_csv_by_intersection(fetch_csv_path: str, metadata_csv_path: str, output_csv_path: str):

    print(f"Starting synchronization...")
    print(f"   - File getting update (CSV1) : {fetch_csv_path}")
    print(f"   - Reference file (CSV2) : {metadata_csv_path}")

    try:
        df_fetch = pd.read_csv(fetch_csv_path)
        df_metadata = pd.read_csv(metadata_csv_path)

    except FileNotFoundError as e:
        print(f"ERROR: One of the CSVs wasn't found. Verify path : {e}")
        return
    except Exception as e:
        print(f"ERROR while loading files : {e}")
        return

    print(f"\nInitial dimensions of {fetch_csv_path} (Row, Cols): {df_fetch.shape}")
    print(f"\nInitial dimensions of {metadata_csv_path} (Row, Cols): {df_metadata.shape}")

    metadata_keys = set(
        df_metadata[KEY_COLUMNS].astype(str).apply(lambda row: tuple(row.str.lower()), axis=1)
    )

    df_fetch['key_tuple'] = df_fetch[KEY_COLUMNS].astype(str).apply(lambda row: tuple(row.str.lower()), axis=1)

    df_fetch_filtered = df_fetch[df_fetch['key_tuple'].isin(metadata_keys)].copy()

    df_fetch_filtered = df_fetch_filtered.drop(columns=['key_tuple'])

    df_fetch_filtered.to_csv(output_csv_path, index=False)

    rows_before = df_fetch.shape[0]
    rows_after = df_fetch_filtered.shape[0]
    rows_deleted = rows_before - rows_after

    print("\nSynchronization completed successfully !")
    print(f"   - Rows keept {output_csv_path} : {rows_after}")
    print(f"   - Rows deleted : {rows_deleted}")
    print(f"   - New file path : **{output_csv_path}**")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Program to synchronize between the metadate CSV and the fetch CSV')
    parser.add_argument('--fetch-csv', type=str, help='Path to fetch CSV', default=FETCH_CSV_PATH)
    parser.add_argument('--metadata-csv', type=str, help='Path to metadata CSV', default=METADATA_CSV_PATH)
    parser.add_argument('--output-csv', type=str, help='Path to output CSV', default=FETCH_CSV_PATH)

    args = parser.parse_args()

    update_csv_by_intersection(fetch_csv_path=args.fetch_csv, metadata_csv_path=args.metadata_csv, output_csv_path=args.output_csv)