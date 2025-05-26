#!/usr/bin/env python3
"""
Create Human-in-the-Loop Validation Export

This script creates a clean, human-readable CSV export of the processed data
specifically designed for manual validation and quality assurance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

def create_validation_export():
    """Create a human-readable validation export."""
    
    print("📋 CREATING HUMAN-IN-THE-LOOP VALIDATION EXPORT")
    print("=" * 70)
    
    # Load the processed data
    df = pd.read_csv('processed_data_complete.csv')
    
    # Create validation-focused dataset
    validation_df = df.copy()
    
    # Clean and prepare data for human review
    print("🧹 Cleaning data for human review...")
    
    # 1. Clean text field - remove excessive whitespace and newlines
    validation_df['text_clean'] = validation_df['text'].str.replace('\n', ' ').str.replace('\r', ' ')
    validation_df['text_clean'] = validation_df['text_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 2. Truncate very long text for readability
    validation_df['text_preview'] = validation_df['text_clean'].apply(
        lambda x: x[:200] + '...' if len(str(x)) > 200 else x
    )
    
    # 3. Create human-readable source type
    source_type_mapping = {
        'earnings_presentation': 'Presentation',
        'earnings_call': 'Transcript',
        'financial_supplement': 'Supplement',
        'financial_results': 'Results',
        'other': 'Other'
    }
    validation_df['document_type'] = validation_df['source_type'].map(source_type_mapping).fillna(validation_df['source_type'])
    
    # 4. Create speaker category
    def categorize_speaker(speaker, is_analyst):
        if speaker == 'UNKNOWN':
            return 'Document Text'
        elif is_analyst:
            return 'Analyst'
        elif speaker in ['Operator', 'OPERATOR']:
            return 'Operator'
        elif speaker in ['Jane Fraser', 'Mark Mason']:
            return 'Management'
        else:
            return 'Other Speaker'
    
    validation_df['speaker_category'] = validation_df.apply(
        lambda row: categorize_speaker(row['speaker_norm'], row['analyst_utterance']), axis=1
    )
    
    # 5. Create quality flags
    validation_df['quality_flags'] = ''
    
    # Flag very short text
    validation_df.loc[validation_df['word_count'] < 3, 'quality_flags'] += 'SHORT_TEXT; '
    
    # Flag very long text
    validation_df.loc[validation_df['word_count'] > 100, 'quality_flags'] += 'LONG_TEXT; '
    
    # Flag potential parsing issues
    validation_df.loc[validation_df['text_clean'].str.contains(r'[^\w\s\.\,\!\?\;\:\-\(\)\$\%]', regex=True, na=False), 'quality_flags'] += 'SPECIAL_CHARS; '
    
    # Clean up quality flags
    validation_df['quality_flags'] = validation_df['quality_flags'].str.rstrip('; ')
    
    # Select columns for validation export
    validation_columns = [
        'source_file',
        'quarter_period', 
        'document_type',
        'sentence_id',
        'speaker_category',
        'speaker_norm',
        'analyst_utterance',
        'text_preview',
        'word_count',
        'quality_flags',
        'call_id',
        'processing_date'
    ]
    
    validation_export = validation_df[validation_columns].copy()
    
    # Rename columns for clarity
    column_renames = {
        'source_file': 'Source_File',
        'quarter_period': 'Quarter',
        'document_type': 'Document_Type',
        'sentence_id': 'Sentence_ID',
        'speaker_category': 'Speaker_Category',
        'speaker_norm': 'Speaker_Name',
        'analyst_utterance': 'Is_Analyst',
        'text_preview': 'Text_Content',
        'word_count': 'Word_Count',
        'quality_flags': 'Quality_Flags',
        'call_id': 'Call_ID',
        'processing_date': 'Processing_Date'
    }
    
    validation_export = validation_export.rename(columns=column_renames)
    
    # Sort for logical review order
    validation_export = validation_export.sort_values([
        'Quarter', 'Source_File', 'Sentence_ID'
    ]).reset_index(drop=True)
    
    # Add validation row numbers
    validation_export.insert(0, 'Review_ID', range(1, len(validation_export) + 1))
    
    # Create summary statistics
    print("📊 Generating validation summary...")
    
    summary_stats = {
        'Total Records': len(validation_export),
        'Unique Source Files': validation_export['Source_File'].nunique(),
        'Quarters Covered': ', '.join(sorted(validation_export['Quarter'].unique())),
        'Document Types': ', '.join(sorted(validation_export['Document_Type'].unique())),
        'Speaker Categories': ', '.join(sorted(validation_export['Speaker_Category'].unique())),
        'Records with Quality Flags': len(validation_export[validation_export['Quality_Flags'] != '']),
        'Average Word Count': f"{validation_export['Word_Count'].mean():.1f}",
        'Processing Date Range': f"{validation_export['Processing_Date'].min()} to {validation_export['Processing_Date'].max()}"
    }
    
    # Create sample records for each document type
    print("📋 Creating sample records by document type...")
    
    sample_records = []
    for doc_type in validation_export['Document_Type'].unique():
        type_records = validation_export[validation_export['Document_Type'] == doc_type]
        
        # Get diverse samples
        samples = []
        
        # Sample from different speakers if available
        for speaker_cat in type_records['Speaker_Category'].unique():
            speaker_records = type_records[type_records['Speaker_Category'] == speaker_cat]
            if len(speaker_records) > 0:
                samples.append(speaker_records.iloc[0])
            if len(samples) >= 3:  # Max 3 samples per doc type
                break
        
        # If we don't have enough samples, add more
        if len(samples) < 3:
            remaining = type_records[~type_records.index.isin([s.name for s in samples])]
            additional = remaining.head(3 - len(samples))
            samples.extend([additional.iloc[i] for i in range(len(additional))])
        
        for sample in samples:
            sample_records.append({
                'Document_Type': doc_type,
                'Sample_Text': sample['Text_Content'],
                'Speaker': sample['Speaker_Name'],
                'Word_Count': sample['Word_Count'],
                'Quality_Flags': sample['Quality_Flags']
            })
    
    # Save validation export
    validation_file = 'validation_export_human_review.csv'
    validation_export.to_csv(validation_file, index=False, encoding='utf-8-sig')
    
    # Save summary report
    summary_file = 'validation_summary_report.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("HUMAN-IN-THE-LOOP VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 30 + "\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nDOCUMENT TYPE BREAKDOWN:\n")
        f.write("-" * 30 + "\n")
        doc_counts = validation_export['Document_Type'].value_counts()
        for doc_type, count in doc_counts.items():
            percentage = (count / len(validation_export)) * 100
            f.write(f"{doc_type}: {count} records ({percentage:.1f}%)\n")
        
        f.write(f"\nSPEAKER CATEGORY BREAKDOWN:\n")
        f.write("-" * 30 + "\n")
        speaker_counts = validation_export['Speaker_Category'].value_counts()
        for speaker_cat, count in speaker_counts.items():
            percentage = (count / len(validation_export)) * 100
            f.write(f"{speaker_cat}: {count} records ({percentage:.1f}%)\n")
        
        f.write(f"\nQUALITY FLAGS ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        flagged_records = validation_export[validation_export['Quality_Flags'] != '']
        if len(flagged_records) > 0:
            flag_counts = {}
            for flags in flagged_records['Quality_Flags']:
                for flag in flags.split('; '):
                    if flag:
                        flag_counts[flag] = flag_counts.get(flag, 0) + 1
            
            for flag, count in sorted(flag_counts.items()):
                f.write(f"{flag}: {count} records\n")
        else:
            f.write("No quality flags detected.\n")
        
        f.write(f"\nSAMPLE RECORDS BY DOCUMENT TYPE:\n")
        f.write("-" * 30 + "\n")
        for sample in sample_records:
            f.write(f"\nDocument Type: {sample['Document_Type']}\n")
            f.write(f"Speaker: {sample['Speaker']}\n")
            f.write(f"Word Count: {sample['Word_Count']}\n")
            f.write(f"Quality Flags: {sample['Quality_Flags'] or 'None'}\n")
            f.write(f"Text: {sample['Sample_Text']}\n")
            f.write("-" * 50 + "\n")
    
    # Create focused validation subsets
    print("📂 Creating focused validation subsets...")
    
    # 1. High-priority validation (speakers, analysts, quality issues)
    high_priority = validation_export[
        (validation_export['Speaker_Category'].isin(['Management', 'Analyst'])) |
        (validation_export['Quality_Flags'] != '')
    ].copy()
    
    high_priority_file = 'validation_high_priority.csv'
    high_priority.to_csv(high_priority_file, index=False, encoding='utf-8-sig')
    
    # 2. Random sample for spot checking
    sample_size = min(100, len(validation_export))
    random_sample = validation_export.sample(n=sample_size, random_state=42)
    
    random_sample_file = 'validation_random_sample.csv'
    random_sample.to_csv(random_sample_file, index=False, encoding='utf-8-sig')
    
    # Print completion summary
    print(f"\n✅ VALIDATION EXPORT COMPLETE")
    print("=" * 70)
    print(f"📄 Main validation file: {validation_file}")
    print(f"   • {len(validation_export):,} records ready for human review")
    print(f"   • Columns optimized for validation workflow")
    print(f"   • Quality flags highlight potential issues")
    
    print(f"\n📊 Summary report: {summary_file}")
    print(f"   • Statistical overview and breakdowns")
    print(f"   • Sample records for each document type")
    print(f"   • Quality analysis and recommendations")
    
    print(f"\n🎯 High-priority subset: {high_priority_file}")
    print(f"   • {len(high_priority):,} records requiring focused review")
    print(f"   • Management/analyst statements and quality issues")
    
    print(f"\n🎲 Random sample: {random_sample_file}")
    print(f"   • {len(random_sample):,} records for spot checking")
    print(f"   • Representative sample across all document types")
    
    print(f"\n💡 VALIDATION WORKFLOW RECOMMENDATIONS:")
    print("   1. Start with high-priority subset for critical validation")
    print("   2. Review random sample for overall quality assessment")
    print("   3. Use main file for comprehensive review if needed")
    print("   4. Focus on speaker identification and text quality")
    print("   5. Check document type classification accuracy")
    
    return validation_export, summary_stats

if __name__ == "__main__":
    validation_df, stats = create_validation_export()
    print(f"\n🎉 Validation export ready for human review!")