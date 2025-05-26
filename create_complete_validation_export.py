#!/usr/bin/env python3
"""
Create Complete Validation Export with All Schema Fields

This script creates a comprehensive validation export that includes:
1. All prescribed schema fields (bank_name, quarter, call_id, etc.)
2. Enhanced NLP fields (sentiment, topics, entities, etc.)
3. Human-readable formatting for validation
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_complete_validation_export():
    """Create a complete validation export with all schema fields."""
    
    print("📋 CREATING COMPLETE VALIDATION EXPORT WITH ALL SCHEMA FIELDS")
    print("=" * 80)
    
    # Load the processed data
    df = pd.read_csv('processed_data_complete.csv')
    
    print(f"📊 Original data: {len(df)} records with {len(df.columns)} columns")
    print(f"Available columns: {list(df.columns)}")
    
    # Create complete validation dataset
    validation_df = df.copy()
    
    # Clean text for better readability
    print("🧹 Preparing data for validation...")
    
    validation_df['text_clean'] = validation_df['text'].str.replace('\n', ' ').str.replace('\r', ' ')
    validation_df['text_clean'] = validation_df['text_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Create readable text preview (but keep full text available)
    validation_df['text_preview'] = validation_df['text_clean'].apply(
        lambda x: x[:150] + '...' if len(str(x)) > 150 else x
    )
    
    # Map source_type to human-readable format
    source_type_mapping = {
        'earnings_presentation': 'Presentation',
        'earnings_call': 'Transcript', 
        'financial_supplement': 'Supplement',
        'financial_results': 'Results',
        'other': 'Other'
    }
    validation_df['document_type_readable'] = validation_df['source_type'].map(source_type_mapping).fillna(validation_df['source_type'])
    
    # Create speaker category for readability
    def categorize_speaker(speaker, is_analyst):
        if pd.isna(speaker) or speaker == 'UNKNOWN':
            return 'Document Text'
        elif is_analyst:
            return 'Analyst'
        elif speaker in ['Operator', 'OPERATOR']:
            return 'Operator'
        elif speaker in ['Jane Fraser', 'Mark Mason']:
            return 'Management'
        else:
            return 'Named Speaker'
    
    validation_df['speaker_category'] = validation_df.apply(
        lambda row: categorize_speaker(row['speaker_norm'], row['analyst_utterance']), axis=1
    )
    
    # Add quality flags
    validation_df['quality_flags'] = ''
    
    # Flag very short text
    validation_df.loc[validation_df['word_count'] < 3, 'quality_flags'] += 'SHORT_TEXT; '
    
    # Flag very long text  
    validation_df.loc[validation_df['word_count'] > 100, 'quality_flags'] += 'LONG_TEXT; '
    
    # Flag potential parsing issues
    validation_df.loc[validation_df['text_clean'].str.contains(r'[^\w\s\.\,\!\?\;\:\-\(\)\$\%\&]', regex=True, na=False), 'quality_flags'] += 'SPECIAL_CHARS; '
    
    # Flag missing timestamps
    validation_df.loc[validation_df['timestamp_epoch'].isna(), 'quality_flags'] += 'NO_TIMESTAMP; '
    
    # Clean up quality flags
    validation_df['quality_flags'] = validation_df['quality_flags'].str.rstrip('; ')
    
    # Define complete column set for validation export
    # Core prescribed schema fields (11 required fields)
    core_schema_columns = [
        'bank_name',           # string - Bank identifier
        'quarter',             # string - Quarter period  
        'call_id',             # string - Unique call identifier
        'source_type',         # string - Document type
        'timestamp_epoch',     # long - Unix timestamp
        'timestamp_iso',       # string - ISO timestamp
        'speaker_norm',        # string - Normalized speaker
        'analyst_utterance',   # boolean - Is analyst speaking
        'sentence_id',         # int32 - Sentence sequence
        'text',                # string - Full sentence text
        'file_path'            # string - Source file path
    ]
    
    # Enhanced NLP schema fields (additional 13 fields)
    enhanced_schema_columns = [
        'word_count',          # int - Word count per sentence
        'sentence_length',     # int - Character length
        'sentiment_score',     # float - Sentiment analysis score
        'sentiment_label',     # string - Sentiment classification
        'topic_labels',        # string - Topic assignments
        'topic_scores',        # string - Topic confidence scores
        'named_entities',      # string - Extracted entities
        'key_phrases',         # string - Important phrases
        'processing_date',     # datetime - When processed
        'processing_version',  # string - Pipeline version
        'page_number',         # float - Source page number
        'source_file',         # string - Original filename
        'quarter_period'       # string - Readable quarter
    ]
    
    # Human-readable helper columns for validation
    validation_helper_columns = [
        'document_type_readable',  # Human-readable document type
        'speaker_category',        # Human-readable speaker category
        'text_preview',           # Truncated text for readability
        'quality_flags'           # Quality assessment flags
    ]
    
    # Combine all columns
    all_validation_columns = (
        ['Review_ID'] +  # Add review ID for tracking
        core_schema_columns + 
        enhanced_schema_columns + 
        validation_helper_columns
    )
    
    # Add Review_ID
    validation_df.insert(0, 'Review_ID', range(1, len(validation_df) + 1))
    
    # Select columns that exist in the dataframe
    available_columns = [col for col in all_validation_columns if col in validation_df.columns]
    missing_columns = [col for col in all_validation_columns if col not in validation_df.columns]
    
    if missing_columns:
        print(f"⚠️  Missing columns: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            if col in ['timestamp_epoch', 'timestamp_iso']:
                validation_df[col] = np.nan
            elif col in ['sentiment_score', 'topic_scores', 'page_number']:
                validation_df[col] = np.nan
            elif col in ['sentiment_label', 'topic_labels', 'named_entities', 'key_phrases']:
                validation_df[col] = ''
            else:
                validation_df[col] = ''
    
    # Create the complete validation export
    complete_validation = validation_df[all_validation_columns].copy()
    
    # Sort for logical review order
    complete_validation = complete_validation.sort_values([
        'quarter_period', 'source_file', 'sentence_id'
    ]).reset_index(drop=True)
    
    # Update Review_ID after sorting
    complete_validation['Review_ID'] = range(1, len(complete_validation) + 1)
    
    # Rename columns for better readability while keeping original names
    column_display_names = {
        'bank_name': 'Bank_Name',
        'quarter': 'Quarter_Code', 
        'call_id': 'Call_ID',
        'source_type': 'Source_Type',
        'timestamp_epoch': 'Timestamp_Epoch',
        'timestamp_iso': 'Timestamp_ISO',
        'speaker_norm': 'Speaker_Name',
        'analyst_utterance': 'Is_Analyst',
        'sentence_id': 'Sentence_ID',
        'text': 'Full_Text',
        'file_path': 'File_Path',
        'word_count': 'Word_Count',
        'sentence_length': 'Sentence_Length',
        'sentiment_score': 'Sentiment_Score',
        'sentiment_label': 'Sentiment_Label',
        'topic_labels': 'Topic_Labels',
        'topic_scores': 'Topic_Scores',
        'named_entities': 'Named_Entities',
        'key_phrases': 'Key_Phrases',
        'processing_date': 'Processing_Date',
        'processing_version': 'Processing_Version',
        'page_number': 'Page_Number',
        'source_file': 'Source_File',
        'quarter_period': 'Quarter_Period',
        'document_type_readable': 'Document_Type',
        'speaker_category': 'Speaker_Category',
        'text_preview': 'Text_Preview',
        'quality_flags': 'Quality_Flags'
    }
    
    complete_validation = complete_validation.rename(columns=column_display_names)
    
    # Save complete validation export
    complete_file = 'complete_validation_export.csv'
    complete_validation.to_csv(complete_file, index=False, encoding='utf-8-sig')
    
    # Create schema compliance report
    print("📊 Creating schema compliance analysis...")
    
    prescribed_fields = [
        'Bank_Name', 'Quarter_Code', 'Call_ID', 'Source_Type', 'Timestamp_Epoch',
        'Timestamp_ISO', 'Speaker_Name', 'Is_Analyst', 'Sentence_ID', 'Full_Text', 'File_Path'
    ]
    
    enhanced_fields = [
        'Word_Count', 'Sentence_Length', 'Sentiment_Score', 'Sentiment_Label',
        'Topic_Labels', 'Topic_Scores', 'Named_Entities', 'Key_Phrases',
        'Processing_Date', 'Processing_Version', 'Page_Number', 'Source_File', 'Quarter_Period'
    ]
    
    # Create focused validation subsets
    print("📂 Creating focused validation subsets...")
    
    # 1. Schema compliance validation - focus on core fields
    schema_validation = complete_validation[
        ['Review_ID'] + prescribed_fields + ['Document_Type', 'Speaker_Category', 'Quality_Flags']
    ].copy()
    
    schema_validation_file = 'schema_compliance_validation.csv'
    schema_validation.to_csv(schema_validation_file, index=False, encoding='utf-8-sig')
    
    # 2. NLP features validation - focus on enhanced fields
    nlp_validation = complete_validation[
        ['Review_ID', 'Source_File', 'Quarter_Period', 'Speaker_Name', 'Text_Preview'] + 
        enhanced_fields + ['Quality_Flags']
    ].copy()
    
    nlp_validation_file = 'nlp_features_validation.csv'
    nlp_validation.to_csv(nlp_validation_file, index=False, encoding='utf-8-sig')
    
    # 3. High-priority validation (management/analyst + quality issues)
    high_priority = complete_validation[
        (complete_validation['Speaker_Category'].isin(['Management', 'Analyst'])) |
        (complete_validation['Quality_Flags'] != '')
    ].copy()
    
    high_priority_file = 'high_priority_complete_validation.csv'
    high_priority.to_csv(high_priority_file, index=False, encoding='utf-8-sig')
    
    # Create comprehensive summary report
    summary_file = 'complete_validation_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("COMPLETE VALIDATION EXPORT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("SCHEMA COMPLIANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Prescribed Schema Fields: {len(prescribed_fields)}\n")
        f.write(f"Enhanced NLP Fields: {len(enhanced_fields)}\n")
        f.write(f"Total Fields in Export: {len(complete_validation.columns)}\n")
        f.write(f"Total Records: {len(complete_validation):,}\n\n")
        
        f.write("PRESCRIBED SCHEMA FIELDS:\n")
        for field in prescribed_fields:
            null_count = complete_validation[field].isnull().sum()
            null_pct = (null_count / len(complete_validation)) * 100
            f.write(f"  {field}: {null_count} nulls ({null_pct:.1f}%)\n")
        
        f.write(f"\nENHANCED NLP FIELDS:\n")
        for field in enhanced_fields:
            if field in complete_validation.columns:
                null_count = complete_validation[field].isnull().sum()
                null_pct = (null_count / len(complete_validation)) * 100
                f.write(f"  {field}: {null_count} nulls ({null_pct:.1f}%)\n")
        
        f.write(f"\nDATA QUALITY SUMMARY:\n")
        f.write("-" * 30 + "\n")
        quality_counts = complete_validation['Quality_Flags'].value_counts()
        clean_records = len(complete_validation[complete_validation['Quality_Flags'] == ''])
        f.write(f"Clean Records: {clean_records} ({(clean_records/len(complete_validation)*100):.1f}%)\n")
        f.write(f"Records with Issues: {len(complete_validation) - clean_records}\n\n")
        
        f.write("VALIDATION FILES CREATED:\n")
        f.write("-" * 30 + "\n")
        f.write(f"1. {complete_file} - Complete dataset with all fields\n")
        f.write(f"2. {schema_validation_file} - Core schema compliance validation\n")
        f.write(f"3. {nlp_validation_file} - NLP features validation\n")
        f.write(f"4. {high_priority_file} - High-priority records for review\n")
    
    # Print completion summary
    print(f"\n✅ COMPLETE VALIDATION EXPORT FINISHED")
    print("=" * 80)
    print(f"📄 Complete validation: {complete_file}")
    print(f"   • {len(complete_validation):,} records")
    print(f"   • {len(prescribed_fields)} prescribed schema fields")
    print(f"   • {len(enhanced_fields)} enhanced NLP fields")
    print(f"   • {len(complete_validation.columns)} total columns")
    
    print(f"\n📋 Schema compliance: {schema_validation_file}")
    print(f"   • Focus on core prescribed fields")
    print(f"   • {len(schema_validation):,} records")
    
    print(f"\n🧠 NLP features: {nlp_validation_file}")
    print(f"   • Focus on enhanced NLP fields")
    print(f"   • {len(nlp_validation):,} records")
    
    print(f"\n🎯 High priority: {high_priority_file}")
    print(f"   • Management/analyst statements + quality issues")
    print(f"   • {len(high_priority):,} records")
    
    print(f"\n📊 Summary report: {summary_file}")
    print(f"   • Complete field-by-field analysis")
    print(f"   • Data quality assessment")
    
    return complete_validation

if __name__ == "__main__":
    complete_df = create_complete_validation_export()
    print(f"\n🎉 Complete validation export ready with all schema fields!")