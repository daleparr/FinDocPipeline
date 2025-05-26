#!/usr/bin/env python3
"""
Schema Validation - Compare actual output against prescribed schema design
"""

import pandas as pd
import numpy as np
from datetime import datetime

def validate_schema_compliance():
    """Validate that our ETL output matches the prescribed schema design."""
    
    print("🔍 SCHEMA VALIDATION REPORT")
    print("=" * 80)
    
    # Load the processed data
    df = pd.read_csv('processed_data_complete.csv')
    
    # Prescribed schema from requirements
    prescribed_schema = {
        'bank_name': 'string',
        'quarter': 'string', 
        'call_id': 'string',
        'source_type': 'string',
        'timestamp_epoch': 'long',
        'timestamp_iso': 'string',
        'speaker_norm': 'string',
        'analyst_utterance': 'boolean',
        'sentence_id': 'int32',
        'text': 'string',
        'file_path': 'string'
    }
    
    print("📋 PRESCRIBED SCHEMA vs ACTUAL IMPLEMENTATION")
    print("-" * 80)
    
    compliance_report = []
    
    for field, expected_type in prescribed_schema.items():
        if field in df.columns:
            actual_type = str(df[field].dtype)
            
            # Check type compliance
            type_compliant = False
            if expected_type == 'string' and 'object' in actual_type:
                type_compliant = True
            elif expected_type == 'boolean' and 'bool' in actual_type:
                type_compliant = True
            elif expected_type in ['int32', 'long'] and 'int' in actual_type:
                type_compliant = True
            elif expected_type == 'long' and 'float' in actual_type:
                type_compliant = True  # Can be converted
            
            status = "✅ COMPLIANT" if type_compliant else "❌ NON-COMPLIANT"
            
            compliance_report.append({
                'field': field,
                'prescribed_type': expected_type,
                'actual_type': actual_type,
                'status': status,
                'present': True
            })
            
            print(f"{field:20} | {expected_type:10} | {actual_type:15} | {status}")
        else:
            compliance_report.append({
                'field': field,
                'prescribed_type': expected_type,
                'actual_type': 'MISSING',
                'status': '❌ MISSING',
                'present': False
            })
            print(f"{field:20} | {expected_type:10} | {'MISSING':15} | ❌ MISSING")
    
    # Check for extra fields
    extra_fields = [col for col in df.columns if col not in prescribed_schema.keys()]
    
    print(f"\n📊 SCHEMA COMPLIANCE SUMMARY")
    print("-" * 80)
    
    compliant_fields = len([r for r in compliance_report if '✅' in r['status']])
    total_prescribed = len(prescribed_schema)
    compliance_rate = (compliant_fields / total_prescribed) * 100
    
    print(f"Compliant Fields: {compliant_fields}/{total_prescribed} ({compliance_rate:.1f}%)")
    print(f"Extra Fields: {len(extra_fields)}")
    print(f"Total Fields in Output: {len(df.columns)}")
    
    if extra_fields:
        print(f"\n📎 EXTRA FIELDS (beyond prescribed schema):")
        for field in extra_fields:
            print(f"  • {field} ({df[field].dtype})")
    
    # Sample data validation
    print(f"\n📋 SAMPLE DATA VALIDATION")
    print("-" * 80)
    
    # Check required field patterns
    sample_record = df.iloc[0]
    
    print("Sample Record Analysis:")
    for field in prescribed_schema.keys():
        if field in df.columns:
            value = sample_record[field]
            print(f"  {field}: {value}")
    
    # Validate specific field formats
    print(f"\n🔍 FIELD FORMAT VALIDATION")
    print("-" * 80)
    
    # Check call_id format
    if 'call_id' in df.columns:
        call_id_pattern = df['call_id'].iloc[0]
        expected_pattern = "BankName_Quarter_timestamp"
        print(f"call_id format: {call_id_pattern}")
        print(f"Expected pattern: {expected_pattern}")
        
    # Check quarter format
    if 'quarter' in df.columns:
        quarter_values = df['quarter'].unique()
        print(f"Quarter values: {quarter_values}")
        
    # Check speaker normalization
    if 'speaker_norm' in df.columns:
        speaker_values = df['speaker_norm'].value_counts().head(10)
        print(f"Top speakers: {list(speaker_values.index)}")
        
    # Check analyst_utterance distribution
    if 'analyst_utterance' in df.columns:
        analyst_dist = df['analyst_utterance'].value_counts()
        print(f"Analyst utterance distribution: {analyst_dist.to_dict()}")
    
    # Data quality checks
    print(f"\n📈 DATA QUALITY METRICS")
    print("-" * 80)
    
    print(f"Total records: {len(df):,}")
    print(f"Null values per field:")
    for field in prescribed_schema.keys():
        if field in df.columns:
            null_count = df[field].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"  {field}: {null_count} ({null_pct:.1f}%)")
    
    # Check sentence-level granularity
    if 'text' in df.columns:
        avg_sentence_length = df['text'].str.len().mean()
        avg_word_count = df['word_count'].mean() if 'word_count' in df.columns else 'N/A'
        print(f"Average sentence length: {avg_sentence_length:.1f} characters")
        print(f"Average word count: {avg_word_count}")
    
    # Generate compliance score
    print(f"\n🎯 OVERALL COMPLIANCE ASSESSMENT")
    print("-" * 80)
    
    if compliance_rate >= 90:
        grade = "🟢 EXCELLENT"
    elif compliance_rate >= 75:
        grade = "🟡 GOOD"
    elif compliance_rate >= 50:
        grade = "🟠 NEEDS IMPROVEMENT"
    else:
        grade = "🔴 POOR"
    
    print(f"Schema Compliance: {compliance_rate:.1f}% - {grade}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    missing_fields = [r['field'] for r in compliance_report if not r['present']]
    if missing_fields:
        print("Missing required fields:")
        for field in missing_fields:
            print(f"  • Add {field} field")
    
    non_compliant = [r for r in compliance_report if '❌' in r['status'] and r['present']]
    if non_compliant:
        print("Type conversion needed:")
        for r in non_compliant:
            print(f"  • Convert {r['field']} from {r['actual_type']} to {r['prescribed_type']}")
    
    # Check if we should save as Parquet
    print(f"\n💾 STORAGE FORMAT RECOMMENDATION")
    print("-" * 80)
    print("Current format: CSV")
    print("Prescribed format: Parquet/Arrow for performance")
    print("Recommendation: Convert to Parquet for production use")
    
    return compliance_report, compliance_rate

def convert_to_parquet():
    """Convert the CSV data to Parquet format with proper schema."""
    
    print(f"\n🔄 CONVERTING TO PARQUET FORMAT")
    print("-" * 80)
    
    try:
        # Load CSV
        df = pd.read_csv('processed_data_complete.csv')
        
        # Apply schema corrections
        schema_corrections = {
            'sentence_id': 'int32',
            'analyst_utterance': 'bool',
            'timestamp_epoch': 'Int64',  # Nullable integer
        }
        
        for field, dtype in schema_corrections.items():
            if field in df.columns:
                if dtype == 'int32':
                    df[field] = df[field].astype('int32')
                elif dtype == 'bool':
                    df[field] = df[field].astype('bool')
                elif dtype == 'Int64':
                    df[field] = pd.to_numeric(df[field], errors='coerce').astype('Int64')
        
        # Save as Parquet
        parquet_file = 'processed_data_complete.parquet'
        df.to_parquet(parquet_file, index=False, engine='pyarrow')
        
        print(f"✅ Successfully converted to Parquet: {parquet_file}")
        print(f"File size reduction: CSV vs Parquet")
        
        import os
        csv_size = os.path.getsize('processed_data_complete.csv') / 1024 / 1024
        parquet_size = os.path.getsize(parquet_file) / 1024 / 1024
        reduction = ((csv_size - parquet_size) / csv_size) * 100
        
        print(f"  CSV: {csv_size:.2f} MB")
        print(f"  Parquet: {parquet_size:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to Parquet: {e}")
        return False

if __name__ == "__main__":
    compliance_report, compliance_rate = validate_schema_compliance()
    
    # Try to convert to Parquet
    if compliance_rate >= 75:
        convert_to_parquet()
    
    print(f"\n✅ Schema validation complete!")