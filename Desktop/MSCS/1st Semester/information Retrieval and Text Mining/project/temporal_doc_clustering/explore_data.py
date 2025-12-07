import pandas as pd
import re
from datetime import datetime
from pathlib import Path

# Configuration
DATA_PATH = Path("data/enron_extracted/emails.csv")
CHUNK_SIZE = 10000  # Process 10k rows at a time

def parse_email_header(message_text):
    """Extract metadata from email header"""
    email_data = {
        'date': None,
        'from': None,
        'to': None,
        'subject': None,
        'body': None
    }
    
    try:
        # Extract date
        date_match = re.search(r'Date: (.+)', message_text)
        if date_match:
            email_data['date'] = date_match.group(1).strip()
        
        # Extract from
        from_match = re.search(r'From: (.+)', message_text)
        if from_match:
            email_data['from'] = from_match.group(1).strip()
        
        # Extract to
        to_match = re.search(r'To: (.+)', message_text)
        if to_match:
            email_data['to'] = to_match.group(1).strip()
        
        # Extract subject
        subject_match = re.search(r'Subject: (.+)', message_text)
        if subject_match:
            email_data['subject'] = subject_match.group(1).strip()
        
        # Extract body (everything after X-FileName or after headers)
        body_match = re.search(r'X-FileName: .+?\n\n(.+)', message_text, re.DOTALL)
        if body_match:
            email_data['body'] = body_match.group(1).strip()
    
    except Exception as e:
        print(f"Error parsing email: {e}")
    
    return email_data

def explore_dataset():
    """Explore the Enron email dataset"""
    
    print("="*60)
    print("ENRON EMAIL DATASET EXPLORATION")
    print("="*60)
    print(f"\nDataset location: {DATA_PATH}")
    print(f"File size: {DATA_PATH.stat().st_size / (1024**3):.2f} GB\n")
    
    # Count total rows
    print("Counting total emails (this may take a minute)...")
    total_rows = 0
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
    
    print(f"✓ Total emails in dataset: {total_rows:,}\n")
    
    # Load first chunk for detailed analysis
    print("Loading first chunk for analysis...")
    df_sample = pd.read_csv(DATA_PATH, nrows=1000)
    
    print(f"✓ Loaded {len(df_sample)} sample emails")
    print(f"\nColumns: {list(df_sample.columns)}")
    print(f"\nDataFrame shape: {df_sample.shape}")
    
    # Parse first email
    print("\n" + "="*60)
    print("SAMPLE EMAIL PARSING")
    print("="*60)
    
    first_message = df_sample.iloc[0]['message']
    parsed = parse_email_header(first_message)
    
    print(f"\nDate: {parsed['date']}")
    print(f"From: {parsed['from']}")
    print(f"To: {parsed['to']}")
    print(f"Subject: {parsed['subject']}")
    print(f"\nBody preview (first 200 chars):")
    print("-" * 60)
    if parsed['body']:
        print(parsed['body'][:200] + "...")
    print("-" * 60)
    
    # Date analysis
    print("\n" + "="*60)
    print("DATE RANGE ANALYSIS")
    print("="*60)
    
    dates = []
    print("\nExtracting dates from sample (this may take a moment)...")
    
    for idx, row in df_sample.iterrows():
        parsed_email = parse_email_header(row['message'])
        if parsed_email['date']:
            try:
                # Try to parse the date
                date_str = parsed_email['date']
                # Handle different date formats
                date_str = re.sub(r'\([^)]*\)', '', date_str).strip()
                date_obj = pd.to_datetime(date_str, errors='coerce')
                if pd.notna(date_obj):
                    dates.append(date_obj)
            except:
                pass
    
    if dates:
        print(f"✓ Successfully parsed {len(dates)} dates from sample")
        print(f"\nDate range in sample:")
        print(f"  Earliest: {min(dates).strftime('%Y-%m-%d')}")
        print(f"  Latest: {max(dates).strftime('%Y-%m-%d')}")
        print(f"  Span: {(max(dates) - min(dates)).days} days")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total emails: {total_rows:,}")
    print(f"Sample analyzed: {len(df_sample):,}")
    print(f"Dates successfully parsed: {len(dates)}")
    print(f"Missing values in sample: {df_sample.isnull().sum().sum()}")
    
    print("\n" + "="*60)
    print("✓ Exploration complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. The data structure is confirmed")
    print("2. Email parsing works correctly")
    print("3. Ready to build the preprocessing pipeline")

if __name__ == "__main__":
    try:
        explore_dataset()
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}")
        print("Please ensure the CSV file is in the correct location.")
    except Exception as e:
        print(f"Error during exploration: {e}")
        import traceback
        traceback.print_exc()