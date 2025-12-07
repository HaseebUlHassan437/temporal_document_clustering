"""
Data Preprocessing Module
Author: Abdul Mateen (MSCS25001)

This module handles:
- Email parsing and metadata extraction
- Text cleaning and normalization
- Stopword removal and stemming
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pathlib import Path
from tqdm import tqdm

# Download required NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EmailPreprocessor:
    """Preprocesses Enron email dataset for clustering"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom email-related stopwords
        self.stop_words.update([
            'enron', 'ect', 'subject', 'cc', 'bcc', 'forwarded', 
            'original', 'message', 'email', 'com', 'http', 'www'
        ])
    
    def parse_email(self, message_text):
        """
        Extract metadata from email message
        
        Args:
            message_text (str): Raw email message with headers
            
        Returns:
            dict: Parsed email data
        """
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
            
            # Extract body (everything after X-FileName or last header)
            body_match = re.search(r'X-FileName: .+?\n\n(.+)', message_text, re.DOTALL)
            if body_match:
                email_data['body'] = body_match.group(1).strip()
            else:
                # Fallback: get text after double newline
                parts = message_text.split('\n\n', 1)
                if len(parts) > 1:
                    email_data['body'] = parts[1].strip()
        
        except Exception as e:
            print(f"Error parsing email: {e}")
        
        return email_data
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text):
        """
        Tokenize text and apply stemming
        
        Args:
            text (str): Cleaned text
            
        Returns:
            str: Stemmed text
        """
        if not text:
            return ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words (< 3 characters)
        tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        # Apply stemming
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Join back into string
        return ' '.join(stemmed_tokens)
    
    def preprocess_email(self, message_text):
        """
        Complete preprocessing pipeline for one email
        
        Args:
            message_text (str): Raw email message
            
        Returns:
            dict: Preprocessed email data
        """
        # Parse email
        parsed = self.parse_email(message_text)
        
        # Combine subject and body for text content
        text_content = ""
        if parsed['subject']:
            text_content += parsed['subject'] + " "
        if parsed['body']:
            text_content += parsed['body']
        
        # Clean text
        cleaned_text = self.clean_text(text_content)
        
        # Tokenize and stem
        processed_text = self.tokenize_and_stem(cleaned_text)
        
        return {
            'date': parsed['date'],
            'from': parsed['from'],
            'to': parsed['to'],
            'subject': parsed['subject'],
            'cleaned_text': cleaned_text,
            'processed_text': processed_text
        }
    
    def process_dataset(self, csv_path, output_path, sample_size=None, chunk_size=10000):
        """
        Process entire dataset
        
        Args:
            csv_path (str): Path to input CSV file
            output_path (str): Path to save processed CSV
            sample_size (int, optional): Number of emails to process. If None, process all
            chunk_size (int): Number of rows to process at a time
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        print(f"Loading data from: {csv_path}")
        
        processed_data = []
        total_processed = 0
        
        # Read CSV in chunks
        chunks = pd.read_csv(csv_path, chunksize=chunk_size)
        
        for chunk in tqdm(chunks, desc="Processing emails"):
            for idx, row in chunk.iterrows():
                try:
                    # Preprocess email
                    processed = self.preprocess_email(row['message'])
                    
                    # Only keep emails with meaningful content
                    if processed['processed_text'] and len(processed['processed_text'].split()) > 5:
                        processed['file'] = row['file']
                        processed_data.append(processed)
                        total_processed += 1
                    
                    # Stop if we've reached sample size
                    if sample_size and total_processed >= sample_size:
                        break
                
                except Exception as e:
                    print(f"Error processing email {idx}: {e}")
                    continue
            
            # Stop if we've reached sample size
            if sample_size and total_processed >= sample_size:
                break
        
        print(f"\n✓ Processed {total_processed} emails")
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # Parse dates
        print("Parsing dates...")
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        
        # Convert to timezone-naive datetime
        df['date'] = df['date'].dt.tz_localize(None)
        
        # Remove emails with invalid dates or dates before 1990 (likely errors)
        valid_date_mask = df['date'].notna()
        df = df[valid_date_mask].copy()
        
        # Further filter for reasonable date range (1990-2005)
        df = df[(df['date'].dt.year >= 1990) & (df['date'].dt.year <= 2005)].copy()
        
        print(f"✓ {len(df)} emails with valid dates (1990 onwards)")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Save to CSV
        print(f"\nSaving processed data to: {output_path}")
        df.to_csv(output_path, index=False)
        print("✓ Saved successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total emails processed: {len(df):,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Average words per email: {df['processed_text'].str.split().str.len().mean():.1f}")
        print("="*60)
        
        return df


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = EmailPreprocessor()
    
    # Process a small sample
    # Get project root directory (parent of src/)
    project_root = Path(__file__).parent.parent
    DATA_PATH = project_root / "data/enron_extracted/emails.csv"
    OUTPUT_PATH = project_root / "data/processed_emails.csv"
    
    # Process first 10,000 emails as a test
    df = preprocessor.process_dataset(
        csv_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        sample_size=10000  # Start with 10k emails
    )
    
    print("\n✓ Preprocessing module test complete!")