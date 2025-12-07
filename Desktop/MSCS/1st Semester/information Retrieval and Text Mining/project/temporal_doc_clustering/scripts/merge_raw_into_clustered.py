"""
Merge original/raw email messages into clustered_emails.csv

This script streams the large `data/enron_extracted/emails.csv` file in chunks
and copies the `message` into a new `raw_body` column in the clustered CSV
matching on the `file` column. It writes a new output file to avoid overwriting
the original until you're satisfied: `data/clustered_emails_raw.csv`.

Usage (from project root):
    python scripts/merge_raw_into_clustered.py

Note: This reads the large raw CSV in chunks and only keeps messages for files
present in `data/clustered_emails.csv` to limit memory usage.
"""
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_CSV = DATA_DIR / 'enron_extracted' / 'emails.csv'
CLUSTERED_CSV = DATA_DIR / 'clustered_emails.csv'
OUT_CSV = DATA_DIR / 'clustered_emails_raw.csv'


def main():
    if not RAW_CSV.exists():
        print(f"Raw CSV not found: {RAW_CSV}")
        return
    if not CLUSTERED_CSV.exists():
        print(f"Clustered CSV not found: {CLUSTERED_CSV}")
        return

    print("Loading clustered CSV (small)")
    df_cluster = pd.read_csv(CLUSTERED_CSV)
    needed_files = set(df_cluster['file'].dropna().astype(str).unique())
    print(f"Need raw messages for {len(needed_files)} files")

    # Map: file -> message
    file_to_message = {}

    # Stream the large raw CSV in chunks and extract only needed rows
    chunksize = 20000
    print(f"Scanning raw CSV in chunks ({chunksize})...")
    for chunk in pd.read_csv(RAW_CSV, chunksize=chunksize):
        if 'file' not in chunk.columns or 'message' not in chunk.columns:
            # Try common alternate names
            cols = chunk.columns.tolist()
            print(f"Raw CSV missing expected columns. Found: {cols}")
            break

        # Filter rows where file is in needed_files
        subset = chunk[chunk['file'].isin(needed_files)][['file', 'message']]
        for _, row in subset.iterrows():
            key = str(row['file'])
            if key not in file_to_message:
                # store the original message text
                file_to_message[key] = row['message']

        # early exit when we have all
        if len(file_to_message) >= len(needed_files):
            break

    print(f"Collected {len(file_to_message)} raw messages")

    # Add raw_body column to clustered df
    print("Merging raw messages into clustered dataframe...")
    df_cluster['raw_body'] = df_cluster['file'].astype(str).map(file_to_message)

    # Where raw_body is missing, keep cleaned_text as fallback (optional)
    # (Don't overwrite cleaned_text; just ensure UI can access raw_body)

    print(f"Writing output to: {OUT_CSV}")
    df_cluster.to_csv(OUT_CSV, index=False)
    print("Done. You can now point the app at `data/clustered_emails_raw.csv` or replace the original file.")


if __name__ == '__main__':
    main()
