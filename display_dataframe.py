#!/usr/bin/env python3
"""
Display DataFrame in a clean, readable format
"""

import pandas as pd
import sys
from pathlib import Path

def display_clean_dataframe():
    """Display the processed data in a clean, readable format."""
    
    # Read the CSV file
    df = pd.read_csv('processed_data_complete.csv')
    
    print("🔍 PROCESSED DATA REVIEW - FIRST 50 RECORDS")
    print("=" * 120)
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    # Select key columns for display
    key_columns = [
        'source_file', 'quarter_period', 'sentence_id', 'speaker_norm', 
        'analyst_utterance', 'text', 'word_count', 'source_type'
    ]
    
    # Display first 50 records with key columns
    display_df = df[key_columns].head(50)
    
    # Clean up text for better display
    display_df = display_df.copy()
    display_df['text'] = display_df['text'].str.replace('\n', ' ').str.strip()
    display_df['text'] = display_df['text'].str[:100] + '...' if display_df['text'].str.len().max() > 100 else display_df['text']
    
    print(display_df.to_string(index=True))
    
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 120)
    print(f"Total Records: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Date Range: {df['processing_date'].min()} to {df['processing_date'].max()}")
    
    print(f"\n📁 RECORDS BY SOURCE FILE:")
    print(df['source_file'].value_counts().to_string())
    
    print(f"\n🎤 SPEAKER DISTRIBUTION:")
    print(df['speaker_norm'].value_counts().head(15).to_string())
    
    print(f"\n📅 RECORDS BY QUARTER:")
    print(df['quarter_period'].value_counts().to_string())
    
    print(f"\n🔍 ANALYST VS MANAGEMENT STATEMENTS:")
    print(df['analyst_utterance'].value_counts().to_string())
    
    print(f"\n📊 WORD COUNT STATISTICS:")
    print(df['word_count'].describe().to_string())
    
    print(f"\n📋 SAMPLE TRANSCRIPT RECORDS (with speakers):")
    transcript_records = df[
        (df['source_file'] == 'transcript.pdf') & 
        (df['speaker_norm'] != 'UNKNOWN')
    ].head(10)
    
    for idx, row in transcript_records.iterrows():
        speaker_type = "ANALYST" if row['analyst_utterance'] else "MANAGEMENT"
        print(f"  [{speaker_type}] {row['speaker_norm']}: {row['text'][:100]}...")
    
    print(f"\n✅ Complete dataset available in: processed_data_complete.csv")
    print(f"📄 Summary report available in: processed_data_summary.txt")

if __name__ == "__main__":
    display_clean_dataframe()